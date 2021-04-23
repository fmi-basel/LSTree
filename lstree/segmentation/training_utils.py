import os
import tensorflow as tf
from numpy import ceil

from dlutils.losses.embedding.embedding_loss import MarginInstanceEmbeddingLoss, relabel_sequential
from dlutils.models.rdcnet import GenericRDCnetBase
from dlutils.models.heads import split_output_into_instance_seg

from dlutils.training.callbacks import ModelConfigSaver
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import TerminateOnNaN
from tensorflow.keras.callbacks import EarlyStopping

from dlutils.training.scheduler import CosineAnnealingSchedule


def build_instance_model(input_shape, downsampling_factor,
                         n_downsampling_channels, n_output_channels, n_groups,
                         dilation_rates, channels_per_group, n_steps,
                         n_classes, spacing, dropout):

    model = GenericRDCnetBase(input_shape=input_shape,
                              downsampling_factor=downsampling_factor,
                              n_downsampling_channels=n_downsampling_channels,
                              n_output_channels=n_output_channels,
                              n_groups=n_groups,
                              dilation_rates=dilation_rates,
                              channels_per_group=channels_per_group,
                              n_steps=n_steps,
                              dropout=dropout)

    model = split_output_into_instance_seg(model,
                                           n_classes=n_classes,
                                           spacing=spacing,
                                           class_activation=False)

    return model


#TODO move to dlutils
def mix_datasets_with_reps(d1, d2, batch_size=None, drop_remainder=True):
    '''Repeats 2 dataset and mixes them with interleaving. 
    
    e.g. repeat a short fully annotated dataset and mix it with weak annotations'''

    if batch_size:
        d1 = d1.unbatch()
        d2 = d2.unbatch()

    d1 = d1.repeat()
    d2 = d2.repeat()

    dataset = tf.data.Dataset.zip((d1, d2))
    dataset = dataset.flat_map(lambda d1, d2: tf.data.Dataset.from_tensors(d1).
                               concatenate(tf.data.Dataset.from_tensors(d2)))

    if batch_size:
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset


#TODO move to dlutils and replace old version. breaking changes?
def common_callbacks(output_folder,
                     lr_min,
                     lr_max,
                     epochs,
                     n_restarts=1,
                     epoch_to_restart_growth=1.0,
                     restart_decay=1.,
                     patience=None):
    '''creates several keras callbacks to be used in model.fit or
    model.fit_generator.

    '''

    n_restarts_factor = sum(epoch_to_restart_growth**x
                            for x in range(n_restarts))

    epochs_to_restart = (epochs + 1) / n_restarts_factor
    if epochs_to_restart < 1:
        raise ValueError(
            'Initial epoch_to_restart ({}) < 1. Decrease n_restarts ({}) or epoch_to_restart_growth ({})'
            .format(epochs_to_restart, n_restarts, epoch_to_restart_growth))

    epochs_to_restart = int(ceil(epochs_to_restart))

    callbacks = []

    if lr_max != lr_min:
        callbacks.append(
            LearningRateScheduler(
                CosineAnnealingSchedule(
                    lr_max=lr_max,
                    lr_min=lr_min,
                    epoch_max=epochs_to_restart,
                    epoch_max_growth=epoch_to_restart_growth,
                    reset_decay=restart_decay)))

    callbacks.extend([
        TerminateOnNaN(),
        TensorBoard(os.path.join(output_folder, 'tensorboard-logs'),
                    write_graph=True,
                    write_grads=False,
                    write_images=False,
                    histogram_freq=0),
        ModelCheckpoint(os.path.join(output_folder, 'weights_best.h5'),
                        save_best_only=True,
                        save_weights_only=True),
        ModelCheckpoint(os.path.join(output_folder, 'weights_latest.h5'),
                        save_best_only=False,
                        save_weights_only=True),
    ])

    if patience is not None and patience >= 1:
        callbacks.append(EarlyStopping(patience=patience))

    return callbacks


class RegMarginInstanceEmbeddingLoss(MarginInstanceEmbeddingLoss):
    '''discretization regularization. Forces embedding px 
    that are unannotated to "choose" an instance'''
    def _unbatched_loss(self, packed):
        '''
        '''

        y_true, y_pred = packed
        y_true = relabel_sequential(y_true)  # on random patch level
        one_hot = self._unbatched_label_to_hot(y_true)

        centers = self._unbatched_embedding_center(one_hot, y_pred)
        center_dist = self._unbatched_embeddings_to_center_dist(
            y_pred, centers)

        probs = self._center_dist_to_probs(one_hot, center_dist)

        emb_loss = tf.reduce_mean(self._unbatched_soft_jaccard(one_hot, probs))

        # push all px embeddings to "choose" one of the instance
        unannot_mask = tf.less(y_true, 0)
        bg_probs = tf.where(unannot_mask, probs, 1.)
        # hinged outputs, can be geater than 1 in unsupervised/wrong regions
        bg_probs = tf.minimum(bg_probs, 1.)
        reg_loss = 1. - tf.reduce_mean(tf.reduce_max(bg_probs, axis=-1))

        return emb_loss + 0.2 * reg_loss


def plot_instance_dataset(path, tf_dataset, n_samples=10):
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
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].imshow(raw[zslice], cmap='Greys_r')
        axs[1].imshow(labels[zslice],
                      cmap='blk_glasbey_hv',
                      interpolation='nearest',
                      vmin=-1,
                      vmax=254)
        axs[2].imshow(classes[zslice],
                      cmap='blk_glasbey_hv',
                      interpolation='nearest',
                      vmin=-1,
                      vmax=254)

        axs[0].set_xlabel('image')
        axs[1].set_xlabel('embeddings labels')
        axs[2].set_xlabel('classes labels')

        axs[0].set_title('Min: {:4.1f}, Max: {:4.1f}'.format(
            raw.min(), raw.max()))
        axs[1].set_title('Min: {:4.1f}, Max: {:4.1f}'.format(
            labels.min(), labels.max()))
        axs[2].set_title('Min: {:4.1f}, Max: {:4.1f}'.format(
            classes.min(), classes.max()))
        plt.tight_layout()

    with PdfPages(path) as pdf:
        for raw, y_true in tf_dataset.unbatch().take(n_samples):
            create_single_page(raw, y_true)
            pdf.savefig(bbox_inches='tight')
            plt.close()
