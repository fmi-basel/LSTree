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
from lstree.segmentation.nuclei_tasks import NucleiSegmentationTask, NucleiWeakAnnotTask
from lstree.lineage.parse_ltree import construct_tree
from lstree.lineage.utils import label_parent_id, compute_timepoint_ids
from lstree.lineage.plot import tree_to_dataframe


class BuildTrackingTrainingRecordTask(ExperimentParams,
                                      BuildTrainingRecordBaseTask):
    '''Builds a training record to train a segmentation model
    
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
    xml_tree = luigi.Parameter(description='filename of xml tree')

    record_name = None  # delete param, infered from movie_dir on init

    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        self.record_name = os.path.basename(self.movie_dir)

    def requires(self):
        return {
            'experiment_config':
            ExternalInputFile(
                os.path.join(self.movie_dir, self.movie_config_name)),
            'nuclei_seg':
            NucleiSegmentationTask(movie_dir=self.movie_dir),
            'weak_nuclei_seg':
            NucleiWeakAnnotTask(movie_dir=self.movie_dir),
            'xml_tree':
            ExternalInputFile(
                path=os.path.join(self.movie_dir, self.xml_tree)),
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

    @staticmethod
    def _use_weak_annot_if_missing_label(pred, weak_annot):

        pred_unique_l = np.unique(pred)
        pred_unique_l = pred_unique_l[pred_unique_l > 0]

        annot_unique_l = np.unique(weak_annot)
        annot_unique_l = annot_unique_l[annot_unique_l > 0]

        if set(pred_unique_l) != set(annot_unique_l):
            return weak_annot
        else:
            return pred

    def get_parent_label_lut(self, mamut_t):
        '''
        label_id(T) -> mamut_id(T) -> mamut_parent_id(T) == mamut_id(T-1) -> label_id(T-1)
        '''

        t = mamut_t
        tm = t - 1

        mamutTm_to_labelTM = self.dfn.set_index('mamut_t').loc[[tm]][[
            'mamut_id', 'label_id'
        ]].astype(int).set_index('mamut_id', drop=True).squeeze(axis=1)
        mamutT_to_mamutTm = self.dfn.set_index('mamut_t').loc[[t]][[
            'mamut_id', 'mamut_parent_id'
        ]].astype(int).set_index('mamut_id', drop=True).squeeze(axis=1)
        labelT_to_mamutT = self.dfn.set_index('mamut_t').loc[[t]][[
            'label_id', 'mamut_id'
        ]].astype(int).set_index('label_id', drop=True).squeeze(axis=1)

        mamutT_to_labelTM = mamutT_to_mamutTm.map(mamutTm_to_labelTM)
        labelT_to_labelTM = labelT_to_mamutT.map(mamutT_to_labelTM)

        # convert lut to numpy array for index with ndarray
        return labelT_to_labelTM.reindex(
            range(labelT_to_labelTM.index.max() + 1)).fillna(0).astype(
                np.int16).values

    def _data_gen(self, inputs):
        from dlutils.preprocessing.normalization import standardize

        logger = logging.getLogger('luigi-interface')

        for raw_target, nuclei_target, weak_nuclei_target in inputs:

            raw_path = raw_target.path
            nuclei_path = nuclei_target.path
            weak_nuclei_path = weak_nuclei_target.path

            if (os.path.basename(raw_path) != os.path.basename(nuclei_path)
                ) or (os.path.basename(raw_path) !=
                      os.path.basename(weak_nuclei_path)):
                raise ValueError(
                    "image and (weak) segmentation don't match: {} - {} - {}".
                    format(raw_path, nuclei_path, weak_path))

            # parse timepoint and get path for t-1 (_tm suffix)
            fname = os.path.basename(raw_path)
            mamut_t = int(re.search('T[0-9]{4}', fname)[0][1:])

            raw_path_tm = raw_path.replace('T{:04d}'.format(mamut_t),
                                           'T{:04d}'.format(mamut_t - 1))
            nuclei_path_tm = nuclei_path.replace('T{:04d}'.format(mamut_t),
                                                 'T{:04d}'.format(mamut_t - 1))
            weak_nuclei_path_tm = weak_nuclei_path.replace(
                'T{:04d}'.format(mamut_t), 'T{:04d}'.format(mamut_t - 1))

            if (mamut_t not in self.dfn.mamut_t.unique()) or (
                    mamut_t - 1 not in self.dfn.mamut_t.unique()):
                # skip tracking info not available
                continue

            try:
                raw = imread(raw_path)
                raw_tm = imread(raw_path_tm)
                nuclei = imread(nuclei_path).astype(np.int16)
                nuclei_tm = imread(nuclei_path_tm).astype(np.int16)
                weak_nuclei = imread(weak_nuclei_path)
                weak_nuclei_tm = imread(weak_nuclei_path_tm)
            except Exception as e:
                logger.info('T-1 not found, skipping {}'.format(
                    os.path.basename(raw_path)))
                continue

            if weak_nuclei.max() <= 0:
                # no tracking info for this timepoint
                continue

            logger.info(
                'adding raw|nuclei_seg pair to training record: {} - {} to {}'.
                format(raw_path, nuclei_path, self.record_name))

            # check if nuclei prediction is complete, else use weak annots
            if mamut_t >= self.earliest_track_ends:
                # ~print('using weak annot')
                nuclei = weak_nuclei
                nuclei_tm = weak_nuclei_tm
            else:
                nuclei = self._use_weak_annot_if_missing_label(
                    nuclei, weak_nuclei)
                nuclei_tm = self._use_weak_annot_if_missing_label(
                    nuclei_tm, weak_nuclei_tm)

            # map to parent labels
            lut = self.get_parent_label_lut(mamut_t)
            nuclei[nuclei > 0] = lut[nuclei[nuclei > 0]]

            raw = [raw_tm, raw]
            annot = [nuclei_tm, nuclei]

            # resample to match voxel size of training record if necessary
            if self.spacing != self.img_spacing:
                raw = [
                    match_spacing(arr,
                                  self.img_spacing,
                                  self.spacing,
                                  image_type='greyscale') for arr in raw
                ]
                annot = [
                    match_spacing(arr,
                                  self.img_spacing,
                                  self.spacing,
                                  image_type='label_nearest') for arr in annot
                ]

            raw = np.stack(raw, axis=-1)
            annot = np.stack(annot, axis=-1)

            raw = standardize(raw, separate_channels=True).astype(np.float32)

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

            yield raw, annot

    def get_first_terminus(self, tree):
        '''Returns the timepoint of first terminus (i.e. nuclei stopped being track or goes to the lumen)'''

        terminues_timpoints = [
            attr['mamut_t'] for n, attr in tree.nodes
            if len(list(tree.successors(n))) <= 0
        ]
        return min(terminues_timpoints)

    def run(self):

        self.record_input = list(
            zip(
                self.input()['images'],
                self.input()['nuclei_seg'],
                self.input()['weak_nuclei_seg'],
            ))

        # read image spacing from config file before building the record as usual
        with open(self.input()['experiment_config'].path, 'r') as file:
            experiment_config = json.load(file)
        self.img_spacing = tuple(experiment_config['spacing'])

        # load the tree and find ending timepoint of first track
        tree = construct_tree(self.input()['xml_tree'].path)
        label_parent_id(tree)
        compute_timepoint_ids(tree)
        self.dfn, _ = tree_to_dataframe(tree)
        self.dfn['label_id'] = self.dfn['timepoint_id'] + 1

        self.earliest_track_ends = self.get_first_terminus(tree)

        super().run()


class TrackingTrainingTask(RDCNetParams, TrainingParams,
                           AugmentationTrainingParams, InstanceHeadParams,
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

    @property
    def resources(self):
        # gpu bound, estimated effective load
        resources = super().resources
        resources.update({'memory': 32000})
        return resources

    def _expand_movie_dirs(self):
        movie_dirs = []
        for pattern in self.movie_dirs:
            expanded_dirs = sorted(glob(pattern))
            movie_dirs.extend(expanded_dirs)

        self.movie_dirs = movie_dirs

    def requires(self):
        self._expand_movie_dirs()
        return [
            BuildTrackingTrainingRecordTask(movie_dir=d,
                                            seed=stable_hash(d) % 2**31)
            for d in self.movie_dirs
        ]

    @property
    def model_name(self):
        '''
        '''
        # do not build the model here to avoid tensorflow import/init in the scheduler process
        model_name = 'RDCNet-F{}-DC{}-OC{}-G{}-DR{}-GC{}-S{}-D{}'.format(
            _format_tuple(self.downsampling_factor),
            self.n_downsampling_channels,
            2 * (3 + self.n_classes), self.n_groups,
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
        from lstree.lineage.tracking import build_instancetracking_model

        model = build_instancetracking_model(
            input_shape=(None, None, None, 2),
            downsampling_factor=self.downsampling_factor,
            n_downsampling_channels=self.n_downsampling_channels,
            n_output_channels=2 * (3 + self.n_classes),
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
        from lstree.lineage.tracking import split_samples

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
        from lstree.segmentation.training_utils import common_callbacks
        from lstree.lineage.tracking import plot_instance_tracking_dataset
        from lstree.lineage.tracking import get_tracking_inference_fun

        record_parser = ImageToSegmentationRecordParser(tf.float32,
                                                        tf.int16,
                                                        fixed_ndim=4)

        model_dir = self.output().path.replace('_inference', '')

        logger = logging.getLogger('luigi-interface')
        logger.info(
            'Starting training a nuclei tracking model: {}'.format(model_dir))

        trainset_paths = [inp['train'].path for inp in self.input()]
        validset_paths = [inp['valid'].path for inp in self.input()]

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

        trainset = trainset.unbatch().repeat().batch(self.train_batch_size)
        validset = validset.unbatch().repeat().batch(self.valid_batch_size)

        model = self._build_model()
        model.save(model_dir)

        if self.plot_dataset:
            logger = logging.getLogger('luigi-interface')
            logger.info('plotting tracking training examples to pdf')
            plot_instance_tracking_dataset(
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
                            validation_steps=self.train_batches_per_epoch // 8,
                            callbacks=callbacks)

        # save inference model
        serve = get_tracking_inference_fun(model, self.spacing,
                                           self.intra_margin)
        with self.output().temporary_path() as temp_output_path:
            tf.saved_model.save(model,
                                export_dir=temp_output_path,
                                signatures={
                                    'serve': serve,
                                })


class NucleiTrackingTask(ExperimentParams, GPUTask):
    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')
    ch_subdir = luigi.Parameter(
        description='Input subdirectory containing original tif files')
    out_subdir_nuclei = luigi.Parameter(
        description=
        'Output subdirectory for nuclei segmentation predicted by the tracking model'
    )
    out_subdir_link = luigi.Parameter(
        description=
        'Output subdirectory for linking segmentation predicted by the tracking model'
    )
    out_subdir_score = luigi.Parameter(
        description=
        'Output subdirectory for linking scores (i.e. embeddings distance)')
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
        model_task = TrackingTrainingTask()
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
        def _replace_subdir(path, subdir):
            return path.replace(os.sep + path.split(os.sep)[-2] + os.sep,
                                os.sep + subdir + os.sep)

        return {
            'nuclei': [
                luigi.LocalTarget(
                    _replace_subdir(i.path, self.out_subdir_nuclei))
                for i in self.input()['images']
            ],
            'link': [
                luigi.LocalTarget(_replace_subdir(i.path,
                                                  self.out_subdir_link))
                for i in self.input()['images']
            ],
            'score': [
                luigi.LocalTarget(
                    _replace_subdir(i.path, self.out_subdir_score).replace(
                        '.tif', '.csv')) for i in self.input()['images']
            ]
        }

    def _predict_single_frame(self, sample_idx, nuclei_target, link_target,
                              score_target, model, lock):
        from lstree.lineage.tracking import track_nuclei

        img_t_path = self.input()['images'][sample_idx].path

        logger = logging.getLogger('luigi-interface')
        logger.info('nuclei tracking of: {}'.format(img_t_path))

        img_t = imread(img_t_path)

        if sample_idx + 1 >= len(self.input()['images']):
            labels = np.zeros_like(img_t, dtype=np.uint16)
            labels_link = np.zeros_like(img_t, dtype=np.uint16)
            dists = pd.DataFrame({
                'label_id': [],
                'linking_distance': [],
                'mamut_t': []
            })

        else:
            img_tp_path = self.input()['images'][sample_idx + 1].path

            mamut_t = int(re.search('T[0-9]{4}', img_t_path)[0][1:])
            mamut_tp = int(re.search('T[0-9]{4}', img_tp_path)[0][1:])
            assert mamut_t + 1 == mamut_tp, 'Tracking inputs are not consecutive frames: {}, {}'.format(
                img_t_path, img_tp_path)

            img_tp = imread(img_tp_path)

            labels, labels_link, dists = track_nuclei(img_t, img_tp, model,
                                                      self.model_spacing,
                                                      self.image_spacing, lock)
            unique_l = np.unique(labels)
            unique_l = unique_l[unique_l != 0]
            dists = pd.DataFrame({
                'label_id': unique_l,
                'linking_distance': dists,
                'mamut_t': [mamut_t] * len(dists)
            })
            dists.fillna(999., inplace=True)

        nuclei_target.makedirs()
        link_target.makedirs()
        score_target.makedirs()

        with nuclei_target.temporary_path() as tmp_path:
            imsave(tmp_path,
                   labels,
                   plugin='tifffile',
                   check_contrast=False,
                   compress=9)

        with link_target.temporary_path() as tmp_path:
            imsave(tmp_path,
                   labels_link,
                   plugin='tifffile',
                   check_contrast=False,
                   compress=9)

        with score_target.temporary_path() as tmp_path:
            dists.to_csv(tmp_path, index=False)

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
            for sample_idx, (nuclei_target, link_target,
                             score_target) in enumerate(
                                 zip(self.output()['nuclei'],
                                     self.output()['link'],
                                     self.output()['score'])):
                if nuclei_target.exists() and link_target.exists(
                ) and score_target.exists():
                    logger.info(
                        'all tracking outputs already exist for: {}'.format(
                            nuclei_target.path))
                    continue

                futures.append(
                    executor.submit(self._predict_single_frame, sample_idx,
                                    nuclei_target, link_target, score_target,
                                    model, lock))

            monitor_futures(futures)
