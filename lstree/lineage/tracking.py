import tensorflow as tf
import numpy as np

from skimage.segmentation import relabel_sequential
from improc.segmentation import segment_from_projections
from improc.morphology import fill_holes_sliced, clean_up_labels
from improc.resample import match_spacing
from skimage.transform import resize
from lstree.luigi_utils import thread_lock_wrapper
from dlutils.models.rdcnet import GenericRDCnetBase
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from dlutils.layers.semi_conv import generate_coordinate_grid
from dlutils.postprocessing.voting import embeddings_to_labels, seeded_embeddings_to_labels


def split_output_into_instance_tracking_heads(model, n_classes, spacing=1.):

    spatial_dims = len(model.inputs[0].shape) - 2
    spacing = tuple(
        float(val) for val in np.broadcast_to(spacing, spatial_dims))
    y_preds = model.outputs[0]

    if y_preds.shape[-1] < 2 * (n_classes + spatial_dims):
        raise ValueError(
            'model has less than n_classes + n_spatial_dims channels: {} < {} + {}'
            .format(y_preds.shape[-1], n_classes, spatial_dims))

    # reshape_pseudo_dim
    orig_shape = tf.shape(y_preds)
    new_shape = tf.concat([
        orig_shape[:-1],
        tf.constant([2]),
        tf.constant([n_classes + spatial_dims])
    ],
                          axis=0)
    y_preds = tf.reshape(y_preds, new_shape)

    vfield = y_preds[..., 0:spatial_dims]
    coords = generate_coordinate_grid(tf.shape(vfield), spatial_dims) * spacing
    coords = coords[None, :, :, :, None]

    embeddings = coords + vfield

    semantic_class = y_preds[..., spatial_dims:spatial_dims + n_classes]

    # we used hinged jaccard loss, no need to apply activation
    # ~semantic_class = tf.nn.softmax(semantic_class, axis=-1)

    # rename outputs
    embeddings = Lambda(lambda x: x, name='embeddings')(embeddings)
    semantic_class = Lambda(lambda x: x, name='semantic_class')(semantic_class)

    return Model(inputs=model.inputs,
                 outputs=[embeddings, semantic_class],
                 name=model.name)


def build_instancetracking_model(input_shape, downsampling_factor,
                                 n_downsampling_channels, n_output_channels,
                                 n_groups, dilation_rates, channels_per_group,
                                 n_steps, n_classes, spacing, dropout):
    '''special case of an instance segmentation model with final reshapping of pseudo 4D output (time as channel) into 4D'''

    model = GenericRDCnetBase(input_shape=input_shape,
                              downsampling_factor=downsampling_factor,
                              n_downsampling_channels=n_downsampling_channels,
                              n_output_channels=n_output_channels,
                              n_groups=n_groups,
                              dilation_rates=dilation_rates,
                              channels_per_group=channels_per_group,
                              n_steps=n_steps,
                              dropout=dropout)

    model = split_output_into_instance_tracking_heads(model,
                                                      n_classes=n_classes,
                                                      spacing=spacing)

    return model


def split_samples(data):

    # add channel dimension to target (4th dim is time)
    data['segm'] = data['segm'][..., None]

    fg = tf.minimum(tf.cast(data['segm'], tf.int32), 1)
    fg = tf.squeeze(fg, axis=-1)
    fg_hot = tf.one_hot(fg, depth=2)
    return data['image'], {
        'embeddings': data['segm'],
        'semantic_class': fg_hot
    }


def plot_instance_tracking_dataset(path, tf_dataset, n_samples=10):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    import inter_view.color

    def create_single_page(raw, y_true):
        raw = raw.numpy().squeeze()
        labels = y_true['embeddings'].numpy().squeeze().astype(np.int16)
        one_hot_classes = y_true['semantic_class'].numpy().squeeze()
        classes = np.argmax(one_hot_classes,
                            axis=-1) + one_hot_classes.sum(axis=-1) - 1
        classes = classes.astype(np.int16)

        zslice = raw.shape[0] // 2
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))

        axs[0, 0].imshow(raw[zslice, ..., 0], cmap='Greys_r')
        axs[1, 0].imshow(raw[zslice, ..., 1], cmap='Greys_r')
        axs[0, 1].imshow(labels[zslice, ..., 0],
                         cmap='blk_glasbey_hv',
                         interpolation='nearest',
                         vmin=-1,
                         vmax=254)
        axs[1, 1].imshow(labels[zslice, ..., 1],
                         cmap='blk_glasbey_hv',
                         interpolation='nearest',
                         vmin=-1,
                         vmax=254)
        axs[0, 2].imshow(classes[zslice, ..., 0],
                         cmap='blk_glasbey_hv',
                         interpolation='nearest',
                         vmin=-1,
                         vmax=254)
        axs[1, 2].imshow(classes[zslice, ..., 1],
                         cmap='blk_glasbey_hv',
                         interpolation='nearest',
                         vmin=-1,
                         vmax=254)

        axs[0, 0].set_xlabel('image T-1')
        axs[0, 1].set_xlabel('embeddings labels T-1')
        axs[0, 2].set_xlabel('classes labels T-1')
        axs[1, 0].set_xlabel('image T')
        axs[1, 1].set_xlabel('embeddings labels T')
        axs[1, 2].set_xlabel('classes labels T')

        axs[0, 0].set_title('Min: {:4.1f}, Max: {:4.1f}'.format(
            raw.min(), raw.max()))
        axs[0, 1].set_title('Min: {:4.1f}, Max: {:4.1f}'.format(
            labels.min(), labels.max()))
        axs[0, 2].set_title('Min: {:4.1f}, Max: {:4.1f}'.format(
            classes.min(), classes.max()))
        plt.tight_layout()

    with PdfPages(path) as pdf:
        for raw, y_true in tf_dataset.unbatch().take(n_samples):
            create_single_page(raw, y_true)
            pdf.savefig(bbox_inches='tight')
            plt.close()


def mean_embeddings(embeddings, labels, n_classes=None):
    '''Return the mean embeddings over each label.'''

    if n_classes is None:
        n_classes = tf.maximum(0, tf.reduce_max(labels)) + 1

    means = tf.TensorArray(tf.float32, n_classes, element_shape=(3, ))
    for i in tf.range(n_classes):
        mask = labels == i + 1
        means = means.write(
            i, tf.reduce_mean(tf.boolean_mask(embeddings, mask), axis=0))

    return means.stack()


def get_tracking_inference_fun(model, spacing, peak_min_distance):
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, 2), dtype=tf.uint16)
    ])
    def serve(img):
        '''preprocesses image, runs model on it, and postprocesses the models output ... 
        
        
        returns nuclei labels, linking labels in next frame, linking distance in embedding space
        '''
        #preprocess
        # normalize per frame (i.e. per channel)
        img = tf.cast(img, tf.float32)
        img = img - tf.reduce_mean(img, axis=(0, 1, 2), keepdims=True)
        img = img / tf.math.reduce_std(img, axis=(0, 1, 2), keepdims=True)
        img = img[None]

        # apply the actual model.
        embeddings, classes = model(img, training=False)

        # postprocess
        embeddings = embeddings[0]
        fg_mask = classes[0, ..., 1] > 0.5

        labels, centers = embeddings_to_labels(
            embeddings[..., 0, :],
            fg_mask[..., 0],
            peak_min_distance=peak_min_distance,
            spacing=spacing,
            min_count=5,
            return_centers=True)

        labels_link = seeded_embeddings_to_labels(embeddings[..., 1, :],
                                                  fg_mask[..., 1],
                                                  seeds=None,
                                                  dist_threshold=None,
                                                  centers=centers)

        centers_link = mean_embeddings(embeddings[..., 1, :],
                                       labels_link,
                                       n_classes=tf.shape(centers)[0])

        link_dist = tf.math.reduce_euclidean_norm(centers - centers_link,
                                                  axis=-1)

        return {
            'labels': labels,
            'labels_link': labels_link,
            'linking_distance': link_dist
        }

    return serve


def remove_background_label(labels, labels_link, imgs, spacing):
    '''Rough organoid segmentation from MIPs to remove labels far away 
    (e.g. miss predicted nuclei on the FEP foil)'''

    mask = segment_from_projections(imgs.max(axis=-1),
                                    spacing,
                                    sigma=5,
                                    threshold_method='otsu')
    labels[~mask] = 0
    labels_link[~mask] = 0

    return labels, labels_link


def track_nuclei(img_t,
                 img_tp,
                 inference_model,
                 model_spacing,
                 image_spacing,
                 lock=None):
    '''Segments nuclei given tracked seeds'''

    predict_fn = inference_model.signatures['serve']

    if lock is not None:
        predict_fn = thread_lock_wrapper(predict_fn, lock)

    image_spacing = np.asarray(image_spacing)
    model_spacing = np.asarray(model_spacing)

    imgs = [img_t, img_tp]
    if (model_spacing != image_spacing).any():
        img_shape = img_t.shape
        imgs = [
            match_spacing(img,
                          image_spacing,
                          model_spacing,
                          image_type='greyscale') for img in imgs
        ]

    img = np.stack(imgs, axis=-1)

    pred = predict_fn(img=tf.convert_to_tensor(img))
    labels = pred['labels'].numpy()
    labels_link = pred['labels_link'].numpy()
    linking_distance = pred['linking_distance'].numpy()

    labels, labels_link = remove_background_label(labels, labels_link, img,
                                                  model_spacing)

    # same post proc as normal nuclei
    labels = clean_up_labels(labels,
                             fill_holes=True,
                             radius=None,
                             size_threshold=20,
                             keep_largest=True,
                             spacing=model_spacing)

    # keep all because split cells have the same label
    labels_link = clean_up_labels(
        labels_link,
        fill_holes=True,
        radius=None,
        size_threshold=20,
        keep_largest=False,  # !!
        spacing=model_spacing)

    # update distances
    unique_l = np.unique(labels)
    unique_l = unique_l[unique_l != 0]
    linking_distance = linking_distance[unique_l - 1]

    # jointly relabel sequential
    (labels,
     labels_link) = relabel_sequential(np.stack([labels, labels_link],
                                                axis=0))[0]

    if (model_spacing != image_spacing).any():
        labels = resize(labels.astype(np.float32),
                        output_shape=img_shape,
                        order=0,
                        preserve_range=True)
        labels_link = resize(labels_link.astype(np.float32),
                             output_shape=img_shape,
                             order=0,
                             preserve_range=True)

    return labels.astype(np.uint16), labels_link.astype(
        np.uint16), linking_distance
