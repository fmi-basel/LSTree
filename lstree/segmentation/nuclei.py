import tensorflow as tf
import numpy as np

from skimage.segmentation import watershed
from scipy.ndimage.morphology import distance_transform_edt
from skimage.transform import resize

from lstree.luigi_utils import thread_lock_wrapper
from improc.morphology import fill_holes_sliced, clean_up_labels
from improc.resample import match_spacing
from dlutils.postprocessing.voting import embeddings_to_labels, seeded_embeddings_to_labels


def expand_labels(labels, distance, spacing=1):
    '''expands annotations by assigning neighboring background pixels to the closest instance'''

    dist_transform = distance_transform_edt(labels == 0, spacing)
    return watershed(dist_transform,
                     markers=labels,
                     mask=dist_transform < distance)


def seeds_to_annot(seeds, shape, spacing, radius=2, sequential=True):
    '''creates weak annotation with a sphere placed on each seed
    
    args:
        seeds: tree props dataframe with z,y,x image coordinates column
        shape: raw image shape
        spacing: voxel size
        radius: radius of each sphere
        
    '''

    annot = np.zeros(shape, dtype=np.int16)
    if sequential:
        annot[tuple(seeds[['z', 'y',
                           'x']].values.T)] = np.arange(1,
                                                        len(seeds) + 1)
    else:
        annot[tuple(seeds[['z', 'y', 'x']].values.T)] = seeds.mamut_id
    annot = expand_labels(annot, distance=radius,
                          spacing=spacing).astype(np.int16)
    annot[annot == 0] = -1

    return annot


def add_conservative_bg_to_weak_annot(annot, raw, thresholds=(0.3, 0.7)):
    '''Adds background to weak annot with a conservative threshold.
    Also delete labels close to background level (i.e. in case spheres too large to fit in nuclei)'''

    bg_level = raw < np.quantile(raw, 0.10)
    fg_level = np.median(raw[annot > 0])
    bg_threshold = bg_level + (fg_level - bg_level) * thresholds[0]
    fg_threshold = bg_level + (fg_level - bg_level) * thresholds[1]

    fg_mask = raw >= fg_threshold
    fg_mask = fill_holes_sliced(fg_mask)

    annot_mask = annot > 0

    annot[(raw < fg_threshold) & (~fg_mask)] = -1
    annot[(raw < bg_threshold) & (~annot_mask) & (~fg_mask)] = 0

    return annot


def split_samples(data):
    '''Generate input image and output targets for embeddings instance segmentation'''

    fg = tf.minimum(tf.cast(data['segm'], tf.int32), 1)
    fg = tf.squeeze(fg, axis=-1)
    fg_hot = tf.one_hot(fg, depth=2)
    return data['image'], {
        'embeddings': data['segm'],
        'semantic_class': fg_hot
    }


def get_nuclei_inference_fun(model, spacing, peak_min_distance):
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None), dtype=tf.uint16)
    ])
    def serve(img):
        '''preprocesses image, runs model on it, and postprocesses the models output ... 
        '''
        #preprocess
        img = tf.cast(img, tf.float32)
        img = img - tf.reduce_mean(img)
        img = img / tf.math.reduce_std(img)
        img = img[None, ..., None]

        # apply the actual model.
        embeddings, classes = model(img, training=False)

        # postprocess
        embeddings = embeddings[0]
        fg_mask = classes[0, ..., 1] > 0.5

        return embeddings_to_labels(embeddings,
                                    fg_mask,
                                    peak_min_distance=peak_min_distance,
                                    spacing=spacing,
                                    min_count=5)

    return serve


def get_seeded_nuclei_inference_fun(model, spacing, dist_threshold):
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None), dtype=tf.uint16),
        tf.TensorSpec(shape=(None, 3), dtype=tf.int32)
    ])
    def serve(img, seeds):
        '''preprocesses image, runs model on it, and postprocesses the models output ... 
        '''
        #preprocess
        img = tf.cast(img, tf.float32)
        img = img - tf.reduce_mean(img)
        img = img / tf.math.reduce_std(img)
        img = img[None, ..., None]

        # apply the actual model.
        embeddings, classes = model(img, training=False)

        # postprocess
        embeddings = embeddings[0]
        fg_mask = classes[0, ..., 1] > 0.5

        return seeded_embeddings_to_labels(embeddings,
                                           fg_mask,
                                           seeds,
                                           dist_threshold=dist_threshold)

    return serve


def segment_nuclei(img,
                   inference_model,
                   model_spacing,
                   image_spacing,
                   lock=None):
    '''Segments nuclei'''

    predict_fn = inference_model.signatures['serve']

    if lock is not None:
        predict_fn = thread_lock_wrapper(predict_fn, lock)

    image_spacing = np.asarray(image_spacing)
    model_spacing = np.asarray(model_spacing)

    if (model_spacing != image_spacing).any():
        img_shape = img.shape
        img = match_spacing(img,
                            image_spacing,
                            model_spacing,
                            image_type='greyscale')

    pred = predict_fn(img=tf.convert_to_tensor(img))['output_0'].numpy()

    pred = clean_up_labels(pred,
                           fill_holes=True,
                           radius=0.5,
                           size_threshold=None,
                           keep_largest=True,
                           spacing=model_spacing)

    if (model_spacing != image_spacing).any():
        pred = resize(pred.astype(np.float32),
                      output_shape=img_shape,
                      order=0,
                      preserve_range=True)

    return pred.astype(np.uint16)


def segment_nuclei_seeded(img,
                          seeds,
                          inference_model,
                          model_spacing,
                          image_spacing,
                          lock=None):
    '''Segments nuclei given tracked seeds'''

    predict_fn = inference_model.signatures['serve_seeded']

    if lock is not None:
        predict_fn = thread_lock_wrapper(predict_fn, lock)

    image_spacing = np.asarray(image_spacing)
    model_spacing = np.asarray(model_spacing)

    if (model_spacing != image_spacing).any():
        img_shape = img.shape
        img = match_spacing(img,
                            image_spacing,
                            model_spacing,
                            image_type='greyscale')

        seeds = seeds * image_spacing[None] / model_spacing[None]
        seeds = np.round(seeds).astype(np.int32)

    pred = predict_fn(img=tf.convert_to_tensor(img),
                      seeds=tf.convert_to_tensor(seeds))['output_0'].numpy()

    pred = clean_up_labels(pred,
                           fill_holes=True,
                           radius=0.5,
                           size_threshold=None,
                           keep_largest=True,
                           spacing=model_spacing)

    if (model_spacing != image_spacing).any():
        pred = resize(pred.astype(np.float32),
                      output_shape=img_shape,
                      order=0,
                      preserve_range=True)

    return pred.astype(np.uint16)
