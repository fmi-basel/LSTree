import numpy as np
import tensorflow as tf

from skimage.transform import resize
from scipy.ndimage import label as nd_label

from improc.morphology import clean_up_labels
from improc.resample import match_spacing
from dlutils.postprocessing.voting import embeddings_to_labels, seeded_embeddings_to_labels
from dlutils.improc import gaussian_filter

from lstree.luigi_utils import thread_lock_wrapper


def merge_double_nuclei(annot, seeds):
    '''Merges nuclei ids that belong to the same cell'''

    nuclei_to_cell_lut = np.zeros(seeds.timepoint_id.max() + 2, dtype=np.int16)
    nuclei_to_cell_lut[seeds.timepoint_id + 1] = seeds.timepoint_cell_id + 1
    annot = nuclei_to_cell_lut[annot]
    annot[annot == 0] = -1

    return annot


def split_samples_lumen(data):
    '''Generate input image and output targets for embeddings instance segmentation from lumen annot'''

    data['segm'] = tf.cast(data['segm'], tf.int32)

    classes = tf.minimum(data['segm'], 2)
    classes = tf.squeeze(classes, axis=-1)
    classes_hot = tf.one_hot(classes, depth=3)

    segm = tf.zeros_like(data['segm']) - 1

    return data['image'], {'embeddings': segm, 'semantic_class': classes_hot}


def split_samples_cell(data):
    '''Generate input image and output targets for embeddings instance segmentation from partial cell annot'''

    data['segm'] = tf.cast(data['segm'], tf.int32)

    classes = tf.zeros_like(data['segm']) - 1
    classes = tf.squeeze(classes, axis=-1)
    classes_hot = tf.one_hot(classes, depth=3)

    return data['image'], {
        'embeddings': data['segm'],
        'semantic_class': classes_hot
    }


def get_cell_inference_fun(model, spacing, peak_min_distance):
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

        # smooth embedding/probmaps to reduces checkerboard
        # NOTE gaussial_filter support only 1 channel in 3D
        # NOTE might not be necessary if transposed conv is replaced by linear interp + conv or trained longer
        iso_sigma = np.array([2., 2., 2.], dtype=np.float32)
        smooth_filter = gaussian_filter(iso_sigma / spacing,
                                        spatial_rank=3,
                                        truncate=2)
        embeddings = tf.concat([
            smooth_filter(embeddings[..., i:i + 1])
            for i in range(embeddings.shape[-1])
        ],
                               axis=-1)
        classes = tf.concat([
            smooth_filter(classes[..., i:i + 1])
            for i in range(classes.shape[-1])
        ],
                            axis=-1)

        # postprocess
        embeddings = embeddings[0]
        classes = tf.cast(tf.argmax(classes[0], axis=-1), tf.uint16)
        epithelium_mask = classes == 2

        labels = embeddings_to_labels(embeddings,
                                      epithelium_mask,
                                      peak_min_distance=peak_min_distance,
                                      spacing=spacing,
                                      min_count=5)

        return {'lumen_segmentation': classes, 'cell_segmentation': labels}

    return serve


# ~def refine_embeddings(embeddings, spacing=1, n_iter=2):
# ~'''Iteratively resamples embeddings at voted location'''

# ~emb_ndim = embeddings.shape[-1]
# ~spacing = np.broadcast_to(np.asarray(spacing), emb_ndim)

# ~embeddings = tf.cast(tf.round(embeddings / spacing), tf.int32)
# ~embeddings = tf.stack([tf.clip_by_value(embeddings[...,d], 0, tf.shape(embeddings)[d+1]-1) for d in range(emb_ndim)], axis=-1)

# ~for i in range(n_iter):
# ~resampled_embeddings = tf.gather_nd(embeddings[0], tf.reshape(embeddings, (embeddings.shape[0], -1, emb_ndim)))
# ~embeddings = tf.reshape(resampled_embeddings, tf.shape(embeddings))

# ~return tf.cast(embeddings, tf.float32) * spacing

# ~def mean_embeddings(embeddings, labels):
# ~n_classes = tf.maximum(0, tf.reduce_max(labels)) + 1

# ~new_seeds = tf.TensorArray(tf.float32, n_classes-1, element_shape=(3,))
# ~for i in tf.range(1,n_classes):
# ~mask = labels == i
# ~new_seeds = new_seeds.write(i-1, tf.reduce_mean(tf.boolean_mask(embeddings, mask), axis=0))

# ~return new_seeds.stack()


def get_seeded_cell_inference_fun(model, spacing, dist_threshold):
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

        # ~embeddings = refine_embeddings(embeddings, spacing=spacing, n_iter=5)

        # smooth embedding/probmaps to reduces checkerboard
        # NOTE gaussial_filter support only 1 channel in 3D
        # NOTE might not be necessary if transposed conv is replaced by linear interp + conv or trained longer
        iso_sigma = np.array([2., 2., 2.], dtype=np.float32)
        smooth_filter = gaussian_filter(iso_sigma / spacing,
                                        spatial_rank=3,
                                        truncate=2)
        embeddings = tf.concat([
            smooth_filter(embeddings[..., i:i + 1])
            for i in range(embeddings.shape[-1])
        ],
                               axis=-1)
        classes = tf.concat([
            smooth_filter(classes[..., i:i + 1])
            for i in range(classes.shape[-1])
        ],
                            axis=-1)

        # postprocess
        embeddings = embeddings[0]
        classes = tf.cast(tf.argmax(classes[0], axis=-1), tf.uint16)
        epithelium_mask = classes == 2

        # "refine" centers (iteratively resample at voted location)
        # ~for _ in range(10):
        # ~seeds = tf.cast(tf.gather_nd(embeddings, seeds) / spacing[None], tf.int32)

        labels = seeded_embeddings_to_labels(embeddings,
                                             epithelium_mask,
                                             seeds,
                                             dist_threshold=dist_threshold)

        # ~for i in range(10):
        # ~seeds = mean_embeddings(embeddings, labels)
        # ~seeds = tf.cast(seeds  / np.array(spacing)[None], tf.int32)
        # ~labels = seeded_embeddings_to_labels(embeddings,
        # ~epithelium_mask,
        # ~seeds,
        # ~dist_threshold=dist_threshold)
        # ~assert 0

        return {'lumen_segmentation': classes, 'cell_segmentation': labels}

    return serve


def cleanup_lumen_seeded(lumen, seeded_cells):
    '''Keep the "organoid" as the segmented object having the largest 
    overlap with seeded cell segmentation
    '''

    # label organoid disjoint parts
    organoid, _ = nd_label(lumen > 0)

    # keep the one having largest overlap with seeded cell segmentation
    unique_l, counts = np.unique(organoid[seeded_cells > 0],
                                 return_counts=True)
    if len(counts) > 0:
        organoid_label = unique_l[np.argmax(counts)]
        lumen[organoid != organoid_label] = 0
    return lumen


def cleanup_lumen(lumen):
    '''Keep the "organoid" as the segmented object that touches the crop center (if any)'''

    # label organoid disjoint parts
    organoid, _ = nd_label(lumen > 0)
    central_organoid_label = organoid[lumen.shape[0] // 2, lumen.shape[1] // 2,
                                      lumen.shape[2] // 2]
    lumen[organoid != central_organoid_label] = 0

    return lumen


def segment_cells_seeded(img,
                         tree_props,
                         inference_model,
                         model_spacing,
                         image_spacing,
                         lock=None):
    '''Segments lumen and individual cell based on tracked nuclei seeds.
    
    Note: if a cell contain multiple nuclei, they are merged in post-processing
    based on timepoint_cell_id columns
    '''

    predict_fn = inference_model.signatures['serve_seeded']

    if lock is not None:
        predict_fn = thread_lock_wrapper(predict_fn, lock)

    image_spacing = np.asarray(image_spacing)
    model_spacing = np.asarray(model_spacing)

    seeds = tree_props[['z', 'y', 'x']].values.astype(np.int32)

    if (model_spacing != image_spacing).any():
        img_shape = img.shape
        img = match_spacing(img,
                            image_spacing,
                            model_spacing,
                            image_type='greyscale')

        seeds = seeds * image_spacing[None] / model_spacing[None]
        seeds = np.round(seeds).astype(np.int32)

    pred = predict_fn(img=tf.convert_to_tensor(img),
                      seeds=tf.convert_to_tensor(seeds))

    lumen_seg = pred['lumen_segmentation'].numpy()
    cell_seg = pred['cell_segmentation'].numpy()

    # used nuclei seeds --> merge label if multiple nuclei in same cell
    tree_props = tree_props.set_index('timepoint_id').sort_index()
    nuclei_to_cell_lut = tree_props['timepoint_cell_id'].values.astype(
        np.uint16)
    positive_mask = cell_seg > 0
    cell_seg[positive_mask] = nuclei_to_cell_lut[cell_seg[positive_mask] -
                                                 1] + 1

    cell_seg = clean_up_labels(cell_seg,
                               fill_holes=False,
                               radius=0.5,
                               size_threshold=None,
                               keep_largest=False,
                               spacing=model_spacing)

    lumen_seg = cleanup_lumen_seeded(lumen_seg, cell_seg)

    if (model_spacing != image_spacing).any():
        lumen_seg = resize(lumen_seg.astype(np.float32),
                           output_shape=img_shape,
                           order=0,
                           preserve_range=True)

        cell_seg = resize(cell_seg.astype(np.float32),
                          output_shape=img_shape,
                          order=0,
                          preserve_range=True)

    return lumen_seg.astype(np.uint16), cell_seg.astype(np.uint16)


def segment_cells(img,
                  inference_model,
                  model_spacing,
                  image_spacing,
                  lock=None):
    '''Segments lumen and individual cell.'''

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

    pred = predict_fn(img=tf.convert_to_tensor(img))

    lumen_seg = pred['lumen_segmentation'].numpy()
    cell_seg = pred['cell_segmentation'].numpy()

    cell_seg = clean_up_labels(cell_seg,
                               fill_holes=False,
                               radius=0.5,
                               size_threshold=None,
                               keep_largest=True,
                               spacing=model_spacing)

    lumen_seg = cleanup_lumen(lumen_seg)

    if (model_spacing != image_spacing).any():
        lumen_seg = resize(lumen_seg.astype(np.float32),
                           output_shape=img_shape,
                           order=0,
                           preserve_range=True)

        cell_seg = resize(cell_seg.astype(np.float32),
                          output_shape=img_shape,
                          order=0,
                          preserve_range=True)

    return lumen_seg.astype(np.uint16), cell_seg.astype(np.uint16)
