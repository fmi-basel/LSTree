import luigi
import os
import json
import re
import logging
import numpy as np
import pandas as pd
from glob import glob
import threading

from concurrent.futures import ThreadPoolExecutor
from skimage.io import imread, imsave
from improc.resample import match_spacing

from improc.roi import crop_object
from lstree.config import RDCNetParams, TrainingParams, AugmentationTrainingParams, InstanceTrainingParams, InstanceHeadParams, ExperimentParams
from lstree.luigi_utils import ExternalInputFile, BuildTrainingRecordBaseTask, stable_hash, GPUTask, _format_tuple
from lstree.deconv.deconv_tasks import DeconvolutionTask
from lstree.lineage.tree_tasks import TreePropsTask
from lstree.segmentation.nuclei_tasks import BuildNucleiTrainingRecordTask, NucleiSegmentationTask


class BuildLumenTrainingRecordTask(BuildNucleiTrainingRecordTask):
    # same as nuclei training record but different default config in luigi.cfg

    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        self.record_name = 'lumen_' + os.path.basename(self.movie_dir)


class BuildWeakCellTrainingRecordTask(ExperimentParams,
                                      BuildTrainingRecordBaseTask):
    '''Builds a training record with nuclei segmentation as partial annotation
    to train a cell segmentation model
    
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
        self.record_name = 'weak_cell_' + os.path.basename(self.movie_dir)

    def requires(self):
        return {
            'experiment_config':
            ExternalInputFile(
                os.path.join(self.movie_dir, self.movie_config_name)),
            'nuclei_seg':
            NucleiSegmentationTask(movie_dir=self.movie_dir),
            'tree_props':
            TreePropsTask(movie_dir=self.movie_dir),
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
        from lstree.segmentation.cell import merge_double_nuclei

        logger = logging.getLogger('luigi-interface')

        for raw_target, nuclei_seg_target in inputs:

            # open tree props matching current timepoint
            img_name = os.path.basename(raw_target.path)
            mamut_t = int(re.search('T[0-9]{4}', img_name)[0][1:])
            seed_path = os.path.join(self.input()['tree_props'].path,
                                     'T{:04d}.csv'.format(mamut_t))

            if os.path.isfile(seed_path):
                raw_path = raw_target.path
                nuclei_seg_path = nuclei_seg_target.path

                annot_mamut_t = int(
                    re.search('T[0-9]{4}',
                              os.path.basename(nuclei_seg_path))[0][1:])
                if mamut_t != annot_mamut_t:
                    raise ValueError(
                        "image and nuclei_seg names don't match:\n\t{}\n\t{}\n"
                        .format(raw_path, nuclei_seg_path))

                seeds = pd.read_csv(seed_path)
                annot = imread(nuclei_seg_path)
                annot = merge_double_nuclei(annot, seeds)

                logger.info(
                    'adding raw|annot pair to training record: {} - {} to {}'.
                    format(raw_path, nuclei_seg_path, self.record_name))

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
                self.input()['nuclei_seg']))

        # read image spacing from config file before building the record as usual
        with open(self.input()['experiment_config'].path, 'r') as file:
            experiment_config = json.load(file)
        self.img_spacing = tuple(experiment_config['spacing'])

        super().run()


class CellSegmentationTrainingTask(RDCNetParams, TrainingParams,
                                   AugmentationTrainingParams,
                                   InstanceHeadParams, InstanceTrainingParams,
                                   GPUTask):

    training_base_dir = luigi.Parameter(
        description='Directory to read the training records and save the model'
    )
    movie_dirs_lumen = luigi.ListParameter(
        description=
        'List of movie directories or glob patterns to include in training set for lumen annotation'
    )
    movie_dirs_cell = luigi.ListParameter(
        description=
        'List of movie directories or glob patterns to include in training set for cell annotation (i.e. xml tree exists)'
    )
    spacing = luigi.TupleParameter(
        description=
        'voxel size of the training record. (resample image if different)')
    train_batches_per_epoch = luigi.IntParameter(
        description=
        'Number of training batches per epoch (infinitely repeating dataset)')

    def _expand_movie_dirs(self):
        movie_dirs_lumen = []
        for pattern in self.movie_dirs_lumen:
            expanded_dirs = sorted(glob(pattern))
            movie_dirs_lumen.extend(expanded_dirs)

        self.movie_dirs_lumen = movie_dirs_lumen

        movie_dirs_cell = []
        for pattern in self.movie_dirs_cell:
            expanded_dirs = sorted(glob(pattern))
            movie_dirs_cell.extend(expanded_dirs)

        self.movie_dirs_cell = movie_dirs_cell

    def requires(self):
        self._expand_movie_dirs()
        return {
            'lumen_records': [
                BuildLumenTrainingRecordTask(movie_dir=d,
                                             seed=stable_hash(d) % 2**31)
                for d in self.movie_dirs_lumen
            ],
            'cell_records': [
                BuildWeakCellTrainingRecordTask(movie_dir=d,
                                                seed=stable_hash(d) % 2**31 +
                                                1)
                for d in self.movie_dirs_cell
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

    def _get_transforms(self, split_samples):
        from dlutils.dataset.augmentations import random_axis_flip, random_intensity_scaling, random_gaussian_offset

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
        from dlutils.training.callbacks import create_callbacks
        from dlutils.dataset.tfrecords import ImageToSegmentationRecordParser
        from dlutils.losses.jaccard_loss import HingedJaccardLoss
        # ~from dlutils.losses.embedding.embedding_loss import MarginInstanceEmbeddingLoss
        from lstree.segmentation.training_utils import RegMarginInstanceEmbeddingLoss as MarginInstanceEmbeddingLoss
        from lstree.segmentation.training_utils import mix_datasets_with_reps
        from lstree.segmentation.training_utils import common_callbacks
        from lstree.segmentation.training_utils import plot_instance_dataset
        from lstree.segmentation.cell import split_samples_lumen, split_samples_cell, get_cell_inference_fun, get_seeded_cell_inference_fun

        record_parser = ImageToSegmentationRecordParser(tf.float32,
                                                        tf.int16,
                                                        fixed_ndim=4)

        model_dir = self.output().path.replace('_inference', '')

        lumen_trainset_paths = [
            inp['train'].path for inp in self.input()['lumen_records']
        ]
        lumen_validset_paths = [
            inp['valid'].path for inp in self.input()['lumen_records']
        ]

        lumen_trainset = create_dataset(
            lumen_trainset_paths,
            batch_size=self.train_batch_size,
            parser_fn=record_parser.parse,
            transforms=self._get_transforms(split_samples_lumen),
            shuffle_buffer=500,
            shuffle=True,
            drop_remainder=False,
            cache_after_parse=False,
            patch_size=self.patch_size)

        lumen_validset = create_dataset(
            lumen_validset_paths,
            batch_size=self.valid_batch_size,
            parser_fn=record_parser.parse,
            transforms=self._get_transforms(split_samples_lumen),
            drop_remainder=False,
            cache_after_parse=False,
            patch_size=self.patch_size)

        cell_trainset_paths = [
            inp['train'].path for inp in self.input()['cell_records']
        ]
        cell_validset_paths = [
            inp['valid'].path for inp in self.input()['cell_records']
        ]

        cell_trainset = create_dataset(
            cell_trainset_paths,
            batch_size=self.train_batch_size,
            parser_fn=record_parser.parse,
            transforms=self._get_transforms(split_samples_cell),
            shuffle_buffer=500,
            shuffle=True,
            drop_remainder=False,
            cache_after_parse=False,
            patch_size=self.patch_size)

        cell_validset = create_dataset(
            cell_validset_paths,
            batch_size=self.valid_batch_size,
            parser_fn=record_parser.parse,
            transforms=self._get_transforms(split_samples_cell),
            drop_remainder=False,
            cache_after_parse=False,
            patch_size=self.patch_size)

        # mix complete lumen and weak cell annot
        trainset = mix_datasets_with_reps(lumen_trainset,
                                          cell_trainset,
                                          batch_size=self.train_batch_size)

        validset = mix_datasets_with_reps(lumen_validset,
                                          cell_validset,
                                          batch_size=self.train_batch_size)

        model = self._build_model()
        model.save(model_dir)

        if self.plot_dataset:
            logger = logging.getLogger('luigi-interface')
            logger.info('plotting cell training examples to pdf')
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
                            validation_steps=self.train_batches_per_epoch // 4,
                            callbacks=callbacks)

        model.save_weights(
            os.path.join(model_dir, 'checkpoints', 'weights_latest'))

        # reinstentiate uncompiled model and load latest weights
        model = self._build_model()
        model.load_weights(
            os.path.join(model_dir, 'checkpoints', 'weights_latest'))

        # save inference model
        serve = get_cell_inference_fun(model, self.spacing, self.intra_margin)
        serve_seeded = get_seeded_cell_inference_fun(model, self.spacing,
                                                     self.inter_margin)
        with self.output().temporary_path() as temp_output_path:
            tf.saved_model.save(model,
                                export_dir=temp_output_path,
                                signatures={
                                    'serve': serve,
                                    'serve_seeded': serve_seeded
                                })


class CellSegmentationTask(ExperimentParams, GPUTask):
    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')
    ch_subdir = luigi.Parameter(
        description='Input subdirectory containing original tif files')
    out_subdir_lumen = luigi.Parameter(description='Output subdirectory')
    out_subdir_cell = luigi.Parameter(description='Output subdirectory')
    n_threads = luigi.IntParameter(
        4, description='max number of threads for pre/post processing')
    avg_active_threads = luigi.IntParameter(
        2, description='estimated average number of active threads')

    @property
    def resources(self):
        # gpu bound, estimated effective load
        resources = super().resources
        resources.update({'pool_workers': self.avg_active_threads})
        return resources

    def requires(self):
        model_task = CellSegmentationTrainingTask()
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
        cell_paths = [
            inp.path.replace(os.sep + inp.path.split(os.sep)[-2] + os.sep,
                             os.sep + self.out_subdir_cell + os.sep)
            for inp in self.input()['images']
        ]

        lument_paths = [
            inp.path.replace(os.sep + inp.path.split(os.sep)[-2] + os.sep,
                             os.sep + self.out_subdir_lumen + os.sep)
            for inp in self.input()['images']
        ]

        return {
            'organoid': [luigi.LocalTarget(p) for p in lument_paths],
            'cell': [luigi.LocalTarget(p) for p in cell_paths]
        }

    def _predict_single_frame(self, img_target, lumen_output, cell_output,
                              model, lock):
        import tensorflow as tf
        from lstree.segmentation.cell import segment_cells_seeded, segment_cells

        logger = logging.getLogger('luigi-interface')
        logger.info('cell/lumen segmentation of: {}'.format(img_target.path))

        img = imread(img_target.path)

        # open tree props matching current timepoint
        img_name = os.path.basename(img_target.path)
        mamut_t = int(re.search('T[0-9]{4}', img_name)[0][1:])
        seed_path = os.path.join(self.tree_props_path,
                                 'T{:04d}.csv'.format(mamut_t))

        if not os.path.isfile(seed_path):
            logger.info('no seed available for {}'.format(img_target.path))
            lumen_seg, _ = segment_cells(img, model, self.model_spacing,
                                         self.image_spacing, lock)
            cell_seg = np.zeros_like(img, dtype=np.uint16)

        else:
            tree_props = pd.read_csv(seed_path)
            lumen_seg, cell_seg = segment_cells_seeded(img, tree_props, model,
                                                       self.model_spacing,
                                                       self.image_spacing,
                                                       lock)

        lumen_output.makedirs()
        with lumen_output.temporary_path() as temp_output_path:

            imsave(temp_output_path,
                   lumen_seg,
                   plugin='tifffile',
                   check_contrast=False,
                   compress=9)

        cell_output.makedirs()
        with cell_output.temporary_path() as temp_output_path:

            imsave(temp_output_path,
                   cell_seg,
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
        with ThreadPoolExecutor(max_workers=self.n_threads) as threads:
            futures = []
            for img_target, lumen_output, cell_output in zip(
                    self.input()['images'],
                    self.output()['organoid'],
                    self.output()['cell']):
                if lumen_output.exists():
                    logger.info(
                        'lumen segmentation output already exist: {}'.format(
                            lumen_output.path))
                    continue

                futures.append(
                    threads.submit(self._predict_single_frame, img_target,
                                   lumen_output, cell_output, model, lock))

            [f.result() for f in futures]  # necessary to print error

    def run(self):

        # optional tree dependency (else output segmentation without seeds)
        tree_task = TreePropsTask(movie_dir=self.movie_dir)
        if tree_task.input()['xml_tree'].exists():
            yield tree_task
            self.tree_props_path = tree_task.output().path

        else:
            self.tree_props_path = ''

        return super().run()


class MultiCellSegmentationTask(luigi.WrapperTask):

    movie_dirs = luigi.ListParameter(
        description='List of movie directories or glob patterns')

    def _expand_movie_dirs(self):
        movie_dirs = []
        for movie_dir in self.movie_dirs:
            movie_dirs.extend(glob(movie_dir))

        return sorted(movie_dirs)

    def requires(self):

        for movie_dir in self._expand_movie_dirs():
            yield CellSegmentationTask(movie_dir=movie_dir)
