import luigi
import os
import logging
import json
import numpy as np
from glob import glob
import threading

from concurrent.futures import ThreadPoolExecutor
from skimage.io import imread, imsave

from lstree.config import RDCNetParams, TrainingParams, AugmentationTrainingParams
from lstree.luigi_utils import ExternalInputFile, BuildTrainingRecordBaseTask, stable_hash, GPUTask, _format_tuple, monitor_futures


class BuildDenoiseTrainingRecordTask(BuildTrainingRecordBaseTask):
    '''Builds a training record to train a denoising model
    
    Picks random images from a timelapse folder and create a training record from 2D slices
    '''

    base_dir = luigi.Parameter(description='Base images/data directory')
    images_dir = luigi.Parameter(description='raw images directory')
    n_images = luigi.IntParameter(
        description='number of images to use (randomly sampled)')
    min_patch_size = luigi.TupleParameter(
        description='minimum patch size. Pads with reflection if smaller')

    record_name = None  # delete param, infered from images_dir on init

    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        # concat the name of last 2 directory levels (movie,channel)
        self.record_name = '_'.join(self.images_dir.split(os.sep)[-2:])

    def requires(self):

        paths = sorted(
            glob(os.path.join(self.base_dir, self.images_dir, '*.tif')))

        np.random.seed(self.seed + 1)
        idxs = np.random.choice(len(paths), self.n_images, replace=False)
        paths = [ExternalInputFile(path=paths[idx]) for idx in idxs]

        return paths

    def _get_serialization_fun(self):
        import tensorflow as tf
        from lstree.deconv.denoise import AutoEncoderRecordParser

        return AutoEncoderRecordParser(tf.float32, fixed_ndim=3).serialize

    def _data_gen(self, inputs):
        from dlutils.preprocessing.normalization import standardize

        logger = logging.getLogger('luigi-interface')

        for raw_target in inputs:
            logger.info('adding {} to training record: {}'.format(
                raw_target.path, self.record_name))
            raw = imread(raw_target.path)

            # replace black padding from old crops
            crop_mask = raw != 0
            raw[~crop_mask] = np.percentile(raw[crop_mask], 10)

            raw = standardize(raw).astype(np.float32)

            z_profile = raw.mean(axis=(1, 2))

            for z, raw_slice in enumerate(raw):

                # exclude slices without content
                if z_profile[z] > np.median(z_profile):
                    pad_size = [
                        (max(0, (min_size - shape) // 2),
                         max(0,
                             (min_size - shape) // 2 + (min_size - shape) % 2))
                        for min_size, shape in zip(self.min_patch_size,
                                                   raw_slice.shape)
                    ]
                    raw_slice = np.pad(raw_slice, pad_size, 'reflect')
                    yield raw_slice[..., None],


class DenoiseTrainingTask(RDCNetParams, TrainingParams,
                          AugmentationTrainingParams, GPUTask):

    training_base_dir = luigi.Parameter(
        description='Directory to read the training records and save the model'
    )
    base_dir = luigi.Parameter(description='Base images/data directory')
    images_dirs = luigi.ListParameter(
        description=
        'List of image directories or glob pattern to include in training set')

    def _expand_image_dirs(self):
        images_dirs = []
        for pattern in self.images_dirs:

            expanded_dirs = sorted(glob(os.path.join(self.base_dir, pattern)))
            images_dirs.extend(expanded_dirs)

        self.images_dirs = images_dirs

    def requires(self):
        self._expand_image_dirs()
        return [
            BuildDenoiseTrainingRecordTask(images_dir=d,
                                           seed=stable_hash(d) % 2**31)
            for d in self.images_dirs
        ]

    @property
    def model_name(self):
        '''
        '''
        # do not build the model here to avoid tensorflow import/init in the scheduler process
        model_name = 'RDCNet-F{}-DC{}-OC1-G{}-DR{}-GC{}-S{}-D{}'.format(
            _format_tuple(self.downsampling_factor),
            self.n_downsampling_channels, self.n_groups,
            _format_tuple(self.dilation_rates), self.channels_per_group,
            self.n_steps, self.dropout)

        if len(self.suffix) > 0:
            model_name += '_' + self.suffix

        return model_name

    def output(self):
        return luigi.LocalTarget(
            os.path.join(self.training_base_dir, 'out',
                         self.model_name + '_inference'))

    def _build_model(self):
        from lstree.deconv.denoise import build_model

        model = build_model(
            input_shape=(None, None, 1),
            downsampling_factor=self.downsampling_factor,
            n_downsampling_channels=self.n_downsampling_channels,
            n_output_channels=1,
            n_groups=self.n_groups,
            dilation_rates=self.dilation_rates,
            channels_per_group=self.channels_per_group,
            n_steps=self.n_steps,
            dropout=self.dropout)

        return model

    def _get_transforms(self):
        from dlutils.dataset.augmentations import random_axis_flip, random_intensity_scaling, random_gaussian_offset
        from lstree.deconv.denoise import noise2void_training_inputs

        transforms = [
            random_axis_flip(axis=0, flip_prob=0.5),
            random_axis_flip(axis=1, flip_prob=0.5),
            random_intensity_scaling(self.intensity_scaling_bounds, ['image']),
            random_gaussian_offset(self.intensity_offset_sigma, ['image']),
            noise2void_training_inputs(noise_fraction=0.002)
        ]

        return transforms

    def gpu_run(self):
        import tensorflow as tf
        from dlutils.preprocessing.normalization import standardize
        from dlutils.dataset.dataset import create_dataset
        from dlutils.training.callbacks import create_callbacks
        from lstree.deconv.denoise import get_denoise2D_inference_fun, AutoEncoderRecordParser, reconstruction_loss
        from lstree.segmentation.training_utils import common_callbacks

        record_parser = AutoEncoderRecordParser(tf.float32, fixed_ndim=3)

        model_dir = self.output().path.replace('_inference', '')

        trainset_paths = [inp['train'].path for inp in self.input()]
        validset_paths = [inp['valid'].path for inp in self.input()]

        trainset = create_dataset(trainset_paths,
                                  batch_size=self.train_batch_size,
                                  parser_fn=record_parser.parse,
                                  transforms=self._get_transforms(),
                                  shuffle_buffer=5000,
                                  shuffle=True,
                                  drop_remainder=True,
                                  cache_after_parse=False,
                                  patch_size=self.patch_size)

        validset = create_dataset(validset_paths,
                                  batch_size=self.valid_batch_size,
                                  parser_fn=record_parser.parse,
                                  transforms=self._get_transforms(),
                                  drop_remainder=False,
                                  cache_after_parse=False,
                                  patch_size=self.patch_size)

        model = self._build_model()
        model.save(model_dir)

        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate),
                      loss=reconstruction_loss,
                      metrics=None)

        if self.resume_weights:
            model.load_weights(self.resume_weights)

        callbacks = common_callbacks(model_dir,
                                     lr_max=self.learning_rate,
                                     lr_min=self.learning_rate / 100.,
                                     epochs=self.epochs,
                                     n_restarts=self.n_restarts,
                                     epoch_to_restart_growth=2.,
                                     patience=None)

        history = model.fit(trainset,
                            validation_data=validset,
                            epochs=self.epochs,
                            callbacks=callbacks)

        serve = get_denoise2D_inference_fun(model)
        with self.output().temporary_path() as temp_output_path:
            tf.saved_model.save(model,
                                export_dir=temp_output_path,
                                signatures={'serve': serve})


class ChannelBoundsTask(luigi.Task):
    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')
    ch_subdir = luigi.Parameter(
        description='Input subdirectory containing tif files to denoise')

    def requires(self):
        paths = sorted(
            glob(os.path.join(self.movie_dir, self.ch_subdir, '*.tif')))
        return [ExternalInputFile(p) for p in paths]

    def output(self):

        return luigi.LocalTarget(
            os.path.join(self.movie_dir, self.ch_subdir + '_bounds.json'))

    def run(self):
        background = None
        counts = None
        bounds = {'min': 2**16, 'max_q0.99999': 0., 'max': 0.}

        for input_target in self.input():
            raw = imread(input_target.path)

            # ignore 0 intensity pixels (old padding)
            raw = raw[raw != 0]

            bounds['min'] = min(int(raw.min()), bounds['min'])
            bounds['max_q0.99999'] = max(int(np.quantile(raw, 0.99999)),
                                         bounds['max_q0.99999'])
            bounds['max'] = max(int(raw.max()), bounds['max'])

        with self.output().temporary_path() as temp_output_path:
            with open(temp_output_path, 'w') as json_file:
                json.dump(bounds, json_file, sort_keys=True, indent=4)


class DenoiseTask(GPUTask):
    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')
    ch_subdir = luigi.Parameter(
        description='Input subdirectory containing tif files to denoise')
    out_suffix = luigi.Parameter(
        description='Suffix to add to create the output subdirectory')
    n_threads = luigi.IntParameter(
        6, description='max number of threads for pre/post processing')
    avg_active_threads = luigi.IntParameter(
        3, description='estimated average number of active threads')

    @property
    def resources(self):
        # gpu bound, estimated effective load
        resources = super().resources
        resources.update({'pool_workers': self.avg_active_threads})
        return resources

    def requires(self):
        paths = sorted(
            glob(os.path.join(self.movie_dir, self.ch_subdir, '*.tif')))

        return {
            'model':
            DenoiseTrainingTask(),
            'images': [ExternalInputFile(p) for p in paths],
            'bounds':
            ChannelBoundsTask(movie_dir=self.movie_dir,
                              ch_subdir=self.ch_subdir)
        }

    def output(self):
        paths = [
            inp.path.replace(
                os.sep + self.ch_subdir + os.sep,
                os.sep + self.ch_subdir + self.out_suffix + os.sep)
            for inp in self.input()['images']
        ]
        return [luigi.LocalTarget(p) for p in paths]

    def _predict_single_frame(self, img_target, output, model, bounds, lock):
        from lstree.deconv.denoise import denoise_3Dimg

        logger = logging.getLogger('luigi-interface')

        logger.info('denoising: {}'.format(img_target.path))

        raw = imread(img_target.path)
        pred = denoise_3Dimg(raw, model,
                             (bounds['min'], bounds['max_q0.99999']), lock)

        output.makedirs()
        with output.temporary_path() as temp_output_path:

            imsave(
                temp_output_path,
                pred,
                plugin='tifffile',
                check_contrast=False,
                compress=6,
                bigtiff=True,
                # ~imagej=True,
                # ~metadata={'mode': 'composite'}
            )

    def gpu_run(self):
        import tensorflow as tf

        logger = logging.getLogger('luigi-interface')

        model = tf.saved_model.load(self.input()['model'].path)
        lock = threading.Lock()

        with open(self.input()['bounds'].path, 'r') as f:
            bounds = json.load(f)

        # use multithreading to write images (bottleneck when compression enabled)
        # gpu almost fully loaded without multithreaded read --> keep it simple
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            for img_target, output in zip(self.input()['images'],
                                          self.output()):
                if output.exists():
                    logger.info('denoised output already exist: {}'.format(
                        output.path))
                    continue

                futures.append(
                    executor.submit(self._predict_single_frame, img_target,
                                    output, model, bounds, lock))

            monitor_futures(futures)
