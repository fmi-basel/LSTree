import luigi
import os
import json
import re
import logging
import numpy as np
from glob import glob
import pandas as pd
import threading

from concurrent.futures import ThreadPoolExecutor
from skimage.io import imread, imsave
from skimage.segmentation import relabel_sequential

from improc.roi import crop_object
from improc.resample import match_spacing

from lstree.config import RDCNetParams, TrainingParams, AugmentationTrainingParams, InstanceTrainingParams, InstanceHeadParams, ExperimentParams
from lstree.luigi_utils import ExternalInputFile, BuildTrainingRecordBaseTask, stable_hash, GPUTask, _format_tuple, monitor_futures
from lstree.deconv.deconv_tasks import DeconvolutionTask
from lstree.lineage.tree_tasks import TreePropsTask


class BuildNucleiTrainingRecordTask(ExperimentParams,
                                    BuildTrainingRecordBaseTask):
    '''Builds a training record to train a segmentation model
    
    '''

    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')
    ch_subdir = luigi.Parameter(
        description='Input subdirectory containing original tif files')
    annot_subdir = luigi.Parameter(description='annotations subdirectory')
    min_patch_size = luigi.TupleParameter(
        description='minimum patch size. Pads with reflection if smaller')
    patch_margins = luigi.TupleParameter(
        description='cropping margins around labeled region')
    spacing = luigi.TupleParameter(
        description=
        'voxel size of the training record. (resample image if different)')

    record_name = None  # delete param, infered from movie_dir on init

    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        self.record_name = os.path.basename(self.movie_dir)

    def requires(self):
        annot_paths = sorted(
            glob(os.path.join(self.movie_dir, self.annot_subdir, '*.tif')))

        return {
            'experiment_config':
            ExternalInputFile(
                os.path.join(self.movie_dir, self.movie_config_name)),
            'annots': [ExternalInputFile(path=p) for p in annot_paths],
            'images':
            DeconvolutionTask(movie_dir=self.movie_dir,
                              ch_subdir=self.ch_subdir)
        }

    def _get_serialization_fun(self):
        import tensorflow as tf
        from dlutils.dataset.tfrecords import ImageToSegmentationRecordParser

        return ImageToSegmentationRecordParser(tf.float32,
                                               tf.int16,
                                               fixed_ndim=4).serialize

    def _data_gen(self, inputs):
        from dlutils.preprocessing.normalization import standardize

        logger = logging.getLogger('luigi-interface')

        for annot_target in inputs:
            annot_path = annot_target.path
            raw_path = os.path.join(self.images_dir,
                                    os.path.basename(annot_path))

            logger.info(
                'adding raw|annot pair to training record: {} - {} to {}'.
                format(raw_path, annot_path, self.record_name))
            annot = imread(annot_path)
            annot[annot >= 0] = relabel_sequential(annot[annot >= 0])[0]

            raw = imread(raw_path)
            raw = standardize(raw).astype(np.float32)

            # resample to match voxel size of training record if necessary
            if self.spacing != self.img_spacing:
                raw = match_spacing(raw,
                                    self.img_spacing,
                                    self.spacing,
                                    image_type='greyscale')
                annot = match_spacing(annot,
                                      self.img_spacing,
                                      self.spacing,
                                      image_type='label_nearest')

            # tighter crop based on anntoations
            raw = crop_object([raw],
                              annot,
                              min_shape=self.min_patch_size,
                              margins=self.patch_margins,
                              mode='reflect')[0]

            annot = crop_object([annot],
                                annot,
                                min_shape=self.min_patch_size,
                                margins=self.patch_margins,
                                constant_values=-1)[0]

            yield raw[..., None], annot[..., None]

    def run(self):

        self.images_dir = os.path.dirname(self.input()['images'][0].path)

        self.record_input = list(self.input()['annots'])

        # read image spacing from config file before building the record as usual
        with open(self.input()['experiment_config'].path, 'r') as file:
            experiment_config = json.load(file)
        self.img_spacing = tuple(experiment_config['spacing'])

        super().run()


class NucleiWeakAnnotTask(ExperimentParams, luigi.Task):
    '''Generates partial annotations (spheres) from nuclei seed coordinate'''

    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')
    ch_subdir = luigi.Parameter(
        description='Input subdirectory containing original tif files')
    out_subdir = luigi.Parameter(description='Output subdirectory')

    sphere_radius = luigi.FloatParameter(
        3., 'Radius [um] of the shpere palce at each seed location')

    n_threads = luigi.IntParameter(
        8,
        description='max number of threads (e.g. for compressed tif export)')

    @property
    def resources(self):
        return {'pool_workers': self.n_threads}

    def requires(self):
        return {
            'experiment_config':
            ExternalInputFile(
                os.path.join(self.movie_dir, self.movie_config_name)),
            'tree_props':
            TreePropsTask(movie_dir=self.movie_dir),
            'images':
            DeconvolutionTask(movie_dir=self.movie_dir,
                              ch_subdir=self.ch_subdir)
        }

    def output(self):
        paths = [
            inp.path.replace(inp.path.split(os.sep)[-2], self.out_subdir)
            for inp in self.input()['images']
        ]
        return [luigi.LocalTarget(p) for p in paths]

    def _build_weak_annot(self, img_target, annot_target):

        from lstree.segmentation.nuclei import seeds_to_annot, add_conservative_bg_to_weak_annot

        logger = logging.getLogger('luigi-interface')

        if annot_target.exists():
            logger.info('Weak annotation already exist: {}'.format(
                img_target.path))
        else:

            logger.info('Building weak annotation from seed for: {}'.format(
                img_target.path))

            img = imread(img_target.path)

            img_name = os.path.basename(img_target.path)
            mamut_t = int(re.search('T[0-9]{4}', img_name)[0][1:])
            seed_path = os.path.join(self.input()['tree_props'].path,
                                     'T{:04d}.csv'.format(mamut_t))

            if not os.path.isfile(seed_path):
                logger.info('no seed available for {}'.format(img_target.path))
                annot = np.zeros_like(img, dtype=np.int16) - 1

            else:
                seeds = pd.read_csv(seed_path, index_col='timepoint_id')
                seeds.sort_index(inplace=True)

                annot = seeds_to_annot(seeds,
                                       img.shape,
                                       self.spacing,
                                       radius=self.sphere_radius,
                                       sequential=True)
                annot = add_conservative_bg_to_weak_annot(annot,
                                                          img,
                                                          thresholds=(0.1,
                                                                      0.6))

            annot_target.makedirs()

            with annot_target.temporary_path() as temp_output_path:

                imsave(temp_output_path,
                       annot,
                       plugin='tifffile',
                       check_contrast=False,
                       compress=9)

    def run(self):
        with open(self.input()['experiment_config'].path, 'r') as f:
            experiment_config = json.load(f)
            self.spacing = experiment_config['spacing']

        # use multithreading
        # This task is either io bound or numpy/scipy backend release GIL
        # --> use multithreading with "avg_active_threads" ressource managment
        # to maximize cpu usage, even if this task is the only one running
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            for img_target, annot_target in zip(self.input()['images'],
                                                self.output()):

                futures.append(
                    executor.submit(self._build_weak_annot, img_target,
                                    annot_target))

            monitor_futures(futures)


class BuildWeakNucleiTrainingRecordTask(ExperimentParams,
                                        BuildTrainingRecordBaseTask):
    '''Builds a training record to train a nuclei segmentation model
    
    '''

    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')
    ch_subdir = luigi.Parameter(
        description='Input subdirectory containing original tif files')
    min_patch_size = luigi.TupleParameter(
        description='minimum patch size. Pads with reflection if smaller')
    patch_margins = luigi.TupleParameter(
        description='cropping margins around labeled region')
    spacing = luigi.TupleParameter(
        description=
        'voxel size of the training record. (resample image if different)')

    record_name = None  # delete param, infered from images_dir on init

    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        self.record_name = 'weak_' + os.path.basename(self.movie_dir)

    def requires(self):
        return {
            'experiment_config':
            ExternalInputFile(
                os.path.join(self.movie_dir, self.movie_config_name)),
            'weak_annots':
            NucleiWeakAnnotTask(movie_dir=self.movie_dir,
                                ch_subdir=self.ch_subdir),
            'images':
            DeconvolutionTask(movie_dir=self.movie_dir,
                              ch_subdir=self.ch_subdir)
        }

    def _get_serialization_fun(self):
        import tensorflow as tf
        from dlutils.dataset.tfrecords import ImageToSegmentationRecordParser

        return ImageToSegmentationRecordParser(tf.float32,
                                               tf.int16,
                                               fixed_ndim=4).serialize

    def _data_gen(self, inputs):
        from dlutils.preprocessing.normalization import standardize

        logger = logging.getLogger('luigi-interface')

        for raw_target, annot_target in inputs:
            raw_path = raw_target.path
            annot_path = annot_target.path

            annot = imread(annot_path)
            if np.unique(annot).max() < 1:
                # empty annot
                continue

            if os.path.basename(raw_path) != os.path.basename(annot_path):
                raise ValueError(
                    "image and annot names don't match:\n\t{}\n\t{}\n".format(
                        raw_path, annot_path))

            logger.info(
                'adding raw|annot pair to training record: {} - {} to {}'.
                format(raw_path, annot_path, self.record_name))

            raw = imread(raw_path)
            raw = standardize(raw).astype(np.float32)

            # resample to match voxel size of training record if necessary
            if self.spacing != self.img_spacing:
                raw = match_spacing(raw,
                                    self.img_spacing,
                                    self.spacing,
                                    image_type='greyscale')
                annot = match_spacing(annot,
                                      self.img_spacing,
                                      self.spacing,
                                      image_type='label_nearest')

            # tighter crop based on annotations
            raw = crop_object([raw],
                              annot,
                              min_shape=self.min_patch_size,
                              margins=self.patch_margins,
                              mode='reflect')[0]

            annot = crop_object([annot],
                                annot,
                                min_shape=self.min_patch_size,
                                margins=self.patch_margins,
                                constant_values=-1)[0]

            yield raw[..., None], annot[..., None]

    def run(self):

        self.record_input = list(
            zip(self.input()['images'],
                self.input()['weak_annots']))

        # read image spacing from config file before building the record as usual
        with open(self.input()['experiment_config'].path, 'r') as file:
            experiment_config = json.load(file)
        self.img_spacing = tuple(experiment_config['spacing'])

        super().run()


class NucleiSegmentationTrainingTask(RDCNetParams, TrainingParams,
                                     AugmentationTrainingParams,
                                     InstanceHeadParams,
                                     InstanceTrainingParams, GPUTask):

    training_base_dir = luigi.Parameter(
        description='Directory to read the training records and save the model'
    )
    movie_dirs = luigi.ListParameter(
        description=
        'List of movie directories or glob patterns to include in training set'
    )
    spacing = luigi.TupleParameter(
        description=
        'voxel size of the training record. (resample image if different)')
    train_batches_per_epoch = luigi.IntParameter(
        description=
        'Number of training batches per epoch (infinitely repeating dataset)')

    def _expand_movie_dirs(self):
        movie_dirs = []
        for pattern in self.movie_dirs:
            expanded_dirs = sorted(glob(pattern))
            movie_dirs.extend(expanded_dirs)

        self.movie_dirs = movie_dirs

    def requires(self):
        self._expand_movie_dirs()
        return {
            'records': [
                BuildNucleiTrainingRecordTask(movie_dir=d,
                                              seed=stable_hash(d) % 2**31)
                for d in self.movie_dirs
            ],
            'weak_records': [
                BuildWeakNucleiTrainingRecordTask(movie_dir=d,
                                                  seed=stable_hash(d) % 2**31 +
                                                  1) for d in self.movie_dirs
            ]
        }

    @property
    def model_name(self):
        '''
        '''
        # do not build the model here to avoid tensorflow import/init in the scheduler process
        model_name = 'RDCNet-F{}-DC{}-OC{}-G{}-DR{}-GC{}-S{}-D{}'.format(
            _format_tuple(self.downsampling_factor),
            self.n_downsampling_channels, 3 + self.n_classes, self.n_groups,
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
        from lstree.segmentation.training_utils import build_instance_model

        model = build_instance_model(
            input_shape=(None, None, None, 1),
            downsampling_factor=self.downsampling_factor,
            n_downsampling_channels=self.n_downsampling_channels,
            n_output_channels=3 + self.n_classes,
            n_groups=self.n_groups,
            dilation_rates=self.dilation_rates,
            channels_per_group=self.channels_per_group,
            n_steps=self.n_steps,
            n_classes=self.n_classes,
            spacing=self.spacing,
            dropout=self.dropout)

        return model

    def _get_transforms(self):
        from dlutils.dataset.augmentations import random_axis_flip, random_intensity_scaling, random_gaussian_offset
        from lstree.segmentation.nuclei import split_samples

        transforms = [
            random_axis_flip(axis=0, flip_prob=0.5),
            random_axis_flip(axis=1, flip_prob=0.5),
            random_axis_flip(axis=2, flip_prob=0.5),
            random_intensity_scaling(self.intensity_scaling_bounds, ['image']),
            random_gaussian_offset(self.intensity_offset_sigma, ['image']),
            split_samples
        ]

        return transforms

    def gpu_run(self):
        import tensorflow as tf
        from dlutils.preprocessing.normalization import standardize
        from dlutils.dataset.dataset import create_dataset
        from dlutils.dataset.tfrecords import ImageToSegmentationRecordParser
        from dlutils.losses.jaccard_loss import HingedJaccardLoss
        # ~from dlutils.losses.embedding.embedding_loss import MarginInstanceEmbeddingLoss
        from lstree.segmentation.training_utils import RegMarginInstanceEmbeddingLoss as MarginInstanceEmbeddingLoss
        from lstree.segmentation.training_utils import mix_datasets_with_reps
        from lstree.segmentation.training_utils import common_callbacks
        from lstree.segmentation.training_utils import plot_instance_dataset
        from lstree.segmentation.nuclei import get_nuclei_inference_fun, get_seeded_nuclei_inference_fun

        record_parser = ImageToSegmentationRecordParser(tf.float32,
                                                        tf.int16,
                                                        fixed_ndim=4)

        model_dir = self.output().path.replace('_inference', '')

        logger = logging.getLogger('luigi-interface')
        logger.info('Starting training a nuclei segmentation model: {}'.format(
            model_dir))

        trainset_paths = [inp['train'].path for inp in self.input()['records']]
        validset_paths = [inp['valid'].path for inp in self.input()['records']]

        trainset = create_dataset(trainset_paths,
                                  batch_size=self.train_batch_size,
                                  parser_fn=record_parser.parse,
                                  transforms=self._get_transforms(),
                                  shuffle_buffer=500,
                                  shuffle=True,
                                  drop_remainder=False,
                                  cache_after_parse=False,
                                  patch_size=self.patch_size)

        validset = create_dataset(validset_paths,
                                  batch_size=self.valid_batch_size,
                                  parser_fn=record_parser.parse,
                                  transforms=self._get_transforms(),
                                  drop_remainder=False,
                                  cache_after_parse=False,
                                  patch_size=self.patch_size)

        weak_trainset_paths = [
            inp['train'].path for inp in self.input()['weak_records']
        ]

        weak_trainset = create_dataset(weak_trainset_paths,
                                       batch_size=self.train_batch_size,
                                       parser_fn=record_parser.parse,
                                       transforms=self._get_transforms(),
                                       shuffle_buffer=500,
                                       shuffle=True,
                                       drop_remainder=False,
                                       cache_after_parse=False,
                                       patch_size=self.patch_size)

        # mix complete and weak annot
        trainset = mix_datasets_with_reps(trainset,
                                          weak_trainset,
                                          batch_size=self.train_batch_size)

        model = self._build_model()
        model.save(model_dir)

        if self.plot_dataset:
            logger = logging.getLogger('luigi-interface')
            logger.info('plotting nuclei training examples to pdf')
            plot_instance_dataset(
                os.path.join(model_dir, 'training_samples.pdf'), trainset, 100)

        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate),
                      loss={
                          'embeddings':
                          MarginInstanceEmbeddingLoss(
                              intra_margin=self.intra_margin,
                              inter_margin=self.inter_margin,
                              parallel_iterations=4),
                          'semantic_class':
                          HingedJaccardLoss(hinge_thresh=self.jaccard_hinge,
                                            eps=self.jaccard_eps),
                      },
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
                            steps_per_epoch=self.train_batches_per_epoch,
                            callbacks=callbacks)

        # save inference model
        serve = get_nuclei_inference_fun(model, self.spacing,
                                         self.intra_margin)
        serve_seeded = get_seeded_nuclei_inference_fun(model, self.spacing,
                                                       self.inter_margin)
        with self.output().temporary_path() as temp_output_path:
            tf.saved_model.save(model,
                                export_dir=temp_output_path,
                                signatures={
                                    'serve': serve,
                                    'serve_seeded': serve_seeded
                                })


class NucleiSegmentationTask(ExperimentParams, GPUTask):
    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')
    ch_subdir = luigi.Parameter(
        description='Input subdirectory containing original tif files')
    out_subdir = luigi.Parameter(description='Output subdirectory')
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
        model_task = NucleiSegmentationTrainingTask()
        self.model_spacing = model_task.spacing

        return {
            'model':
            model_task,
            'images':
            DeconvolutionTask(movie_dir=self.movie_dir,
                              ch_subdir=self.ch_subdir),
            'experiment_config':
            ExternalInputFile(
                os.path.join(self.movie_dir, self.movie_config_name)),
        }

    def output(self):
        paths = [
            inp.path.replace(os.sep + inp.path.split(os.sep)[-2] + os.sep,
                             os.sep + self.out_subdir + os.sep)
            for inp in self.input()['images']
        ]
        return [luigi.LocalTarget(p) for p in paths]

    def _predict_single_frame(self, img_target, output, model, lock):
        import tensorflow as tf
        from lstree.segmentation.nuclei import segment_nuclei, segment_nuclei_seeded

        logger = logging.getLogger('luigi-interface')
        logger.info('nuclei segmentation of: {}'.format(img_target.path))

        img = imread(img_target.path)

        # open tree props matching current timepoint
        img_name = os.path.basename(img_target.path)
        mamut_t = int(re.search('T[0-9]{4}', img_name)[0][1:])
        seed_path = os.path.join(self.tree_props_path,
                                 'T{:04d}.csv'.format(mamut_t))

        if not os.path.isfile(seed_path):
            logger.info('no seed available for {}'.format(img_target.path))
            pred = np.zeros_like(img, dtype=np.uint16)

        else:
            tree_props = pd.read_csv(seed_path)
            seeds = tree_props[['z', 'y', 'x']].values.astype(np.int32)
            pred = segment_nuclei_seeded(img, seeds, model, self.model_spacing,
                                         self.image_spacing, lock)

        output.makedirs()
        with output.temporary_path() as temp_output_path:

            imsave(temp_output_path,
                   pred.astype(np.uint16),
                   plugin='tifffile',
                   check_contrast=False,
                   compress=9)

    def gpu_run(self):
        import tensorflow as tf

        logger = logging.getLogger('luigi-interface')

        model = tf.saved_model.load(self.input()['model'].path)
        lock = threading.Lock()

        with open(self.input()['experiment_config'].path, 'r') as f:
            experiment_config = json.load(f)
            self.image_spacing = experiment_config['spacing']

        # use multithreading to load/write pre/post process images
        # thread lock for gpu step
        # --> ~maximize gpu usage (without batching, i.e. risk OOM)
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            for img_target, output in zip(self.input()['images'],
                                          self.output()):
                if output.exists():
                    logger.info(
                        'nuclei segmentation output already exist: {}'.format(
                            output.path))
                    continue

                futures.append(
                    executor.submit(self._predict_single_frame, img_target,
                                    output, model, lock))
            monitor_futures(futures)

    def run(self):

        # optional tree dependency (else output segmentation without seeds)
        tree_task = TreePropsTask(movie_dir=self.movie_dir)
        if tree_task.input()['xml_tree'].exists():
            yield tree_task
            self.tree_props_path = tree_task.output().path

        else:
            self.tree_props_path = ''

        return super().run()


class MultiNucleiSegmentationTask(luigi.WrapperTask):

    movie_dirs = luigi.ListParameter(
        description='List of movie directories or glob patterns')

    def _expand_movie_dirs(self):
        movie_dirs = []
        for movie_dir in self.movie_dirs:
            movie_dirs.extend(glob(movie_dir))

        return sorted(movie_dirs)

    def requires(self):

        for movie_dir in self._expand_movie_dirs():
            yield NucleiSegmentationTask(movie_dir=movie_dir)
