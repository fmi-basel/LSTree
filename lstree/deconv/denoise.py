import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from lstree.luigi_utils import thread_lock_wrapper
from dlutils.models.rdcnet import GenericRDCnetBase
from dlutils.dataset.tfrecords import RecordParserBase, _bytes_feature, _float_feature, _int64_feature


# TODO move to dlutils
class AutoEncoderRecordParser(RecordParserBase):
    '''defines the serialize and parse function for a typical autoencoder dataset.

    '''

    image_key = 'image'
    img_shape_key = 'img_shape'

    def __init__(self, image_dtype, fixed_ndim=None):
        '''
        Parameters
        ----------
        image_dtype : tf.dtype
            data type of the input image. Note that the output of parse()
            still casts to tf.float32.
        fixed_ndim : int
            length of the image shape. Needed for training loop
            for determining the tensor shapes downstream.

        '''
        self.image_dtype = image_dtype
        self.fixed_ndim = fixed_ndim
        assert fixed_ndim is None or fixed_ndim >= 2

    def serialize(self, image):
        '''serializes a training tuple such that it can be written as tfrecord example.

        Parameters
        ----------
        image : array-like
            image to be serialized. Needs to have image_dtype.

        '''

        features = tf.train.Features(
            feature={
                self.img_shape_key: _int64_feature(list(image.shape)),
                self.image_key: _bytes_feature(image.tostring()),
            })
        return tf.train.Example(features=features)

    def parse(self, example):
        '''parse a tfrecord example.

        Returns
        -------
        sample : dict of tensors
            sample containing image. Note that the image is always converted 
            to tf.float32.

        '''
        features = {
            # Extract features using the keys set during creation
            self.img_shape_key:
            tf.io.FixedLenSequenceFeature([], tf.int64, True),
            self.image_key:
            tf.io.FixedLenFeature([], tf.string),
        }
        sample = tf.io.parse_single_example(example, features)

        # Fixed shape appears to be necessary for training with keras.
        shapes = {
            key: tf.ensure_shape(sample[key], (self.fixed_ndim, ))
            if self.fixed_ndim is not None else sample[key]
            for key in [self.img_shape_key]
        }

        def _reshape_and_cast(val, shape, dtype):
            '''this ensures that tensorflow "knows" the shape of the resulting
            tensors.
            '''
            return tf.reshape(tf.io.decode_raw(val, dtype), shape)

        parsed = {
            self.image_key:
            _reshape_and_cast(sample[self.image_key],
                              shapes[self.img_shape_key], self.image_dtype)
        }
        parsed[self.image_key] = tf.cast(parsed[self.image_key], tf.float32)
        return parsed


def noise2void_training_inputs(noise_fraction=0.002):
    '''Creates 2 images from "image" input key: 
    "image_in": image where noise_fraction pixels have been replaced by noise
    "packed_target": image concatenated with the binary mask of the replaced pixels (to be compatible with keras loss API)
    '''
    def _distorter(input_dict):

        loss_mask = tf.random.uniform(
            shape=tf.shape(input_dict['image'])) < noise_fraction
        noise = tf.random.normal(shape=tf.shape(input_dict['image']),
                                 mean=tf.reduce_mean(input_dict['image'],
                                                     axis=(1, 2),
                                                     keepdims=True),
                                 stddev=tf.math.reduce_std(input_dict['image'],
                                                           axis=(1, 2),
                                                           keepdims=True))
        noise = tf.clip_by_value(
            noise,
            tf.reduce_min(input_dict['image'], axis=(1, 2), keepdims=True),
            tf.reduce_max(input_dict['image'], axis=(1, 2), keepdims=True))

        # replace masked pixels by noise
        masked_img = tf.where(loss_mask, noise, input_dict['image'])
        packed_target = tf.concat(
            [input_dict['image'],
             tf.cast(loss_mask, tf.float32)], axis=-1)

        return masked_img, packed_target

    return _distorter


def fourier_clip(y_true, y_pred):
    '''"clips" frequencies with amplitude higher than y_true. implemented for single channel'''

    y_true = y_true[..., 0]
    y_pred = y_pred[..., 0]

    fft_y_true = tf.signal.fft2d(tf.cast(y_true, tf.complex64))
    fft_y_pred = tf.signal.fft2d(tf.cast(y_pred, tf.complex64))

    fft_y_true_abs = tf.abs(fft_y_true)
    fft_y_pred_abs = tf.abs(fft_y_pred)

    clipped_ratio = tf.maximum(1., fft_y_pred_abs / (fft_y_true_abs + 1e-12))
    fft_y_pred = fft_y_pred / tf.cast(clipped_ratio, tf.complex64)
    return tf.cast(tf.signal.ifft2d(fft_y_pred)[..., None], tf.float32)


def build_model(input_shape, downsampling_factor, n_downsampling_channels,
                n_output_channels, n_groups, dilation_rates,
                channels_per_group, n_steps, dropout):
    model = GenericRDCnetBase(input_shape=input_shape,
                              downsampling_factor=downsampling_factor,
                              n_downsampling_channels=n_downsampling_channels,
                              n_output_channels=n_output_channels,
                              n_groups=n_groups,
                              dilation_rates=dilation_rates,
                              channels_per_group=channels_per_group,
                              n_steps=n_steps,
                              dropout=dropout,
                              up_method='upsample')

    return model


def reconstruction_loss(packed_y_true, y_pred):

    mask = tf.cast(packed_y_true[..., 1:2], tf.bool)
    y_true = packed_y_true[..., 0:1]
    masked_loss = tf.boolean_mask(tf.square(y_true - y_pred), mask)

    return tf.reduce_mean(masked_loss)


denoise_record_parser = AutoEncoderRecordParser(tf.float32, fixed_ndim=3)


def get_denoise2D_inference_fun(model):
    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.float32)])
    def serve(img):
        '''preprocesses image, runs model on it, and postprocesses the models output ... 
        '''
        x = img[None, ..., None]
        y = model(x, training=False)
        y = fourier_clip(x, y)

        return tf.squeeze(y)

    return serve


# NOTE this steps can be performed in tensorflow and save as part of the graph.
# However, even when placing all operations on CPU except the slice prediction,
# it is slower and requires more VRAM ?? (all slices on GPU requires > 8GB VRAM for typical stacks)
def denoise_3Dimg(img, inference_model, bounds=None, lock=None):
    '''Applies a 2D denoising model on each slice of a 3D stack'''

    predict_fn = inference_model.signatures['serve']

    if lock is not None:
        predict_fn = thread_lock_wrapper(predict_fn, lock)

    # mask indiciating slice that are not zero padded
    zmask = img.max(axis=(1, 2)) > 0

    # replace black padding from old crops
    crop_mask = img != 0
    img[~crop_mask] = np.percentile(img[crop_mask], 10)

    img = img.astype(np.float32)
    img_mean = img.mean()
    img_std = img.std()
    img -= img_mean
    img /= img_std

    pred = np.stack([
        predict_fn(tf.convert_to_tensor(sl))['output_0'].numpy() for sl in img
    ],
                    axis=0)

    pred = pred * img_std  #* 50
    pred = pred + img_mean

    # subtract background estimation
    # NOTE
    # ideally background correction should be done before cropping
    # with a background estimated over the entire movie
    #
    # NOTE
    # offset by 2 and clip --> remove background speckles --> ~10x compression
    bg = pred[zmask].min(axis=0)
    pred -= (bg + 2)
    pred = pred.clip(0., None)

    if bounds is not None:
        # rescale intensity to use ~90% of uint16 range
        pred *= 0.9 * (2**16 - 1) / (bounds[1] - bounds[0])

    return pred.astype(np.uint16)
