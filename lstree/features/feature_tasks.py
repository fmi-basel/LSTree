import luigi
import os
import json
import logging
import re
import pandas as pd
import numpy as np
from glob import glob
import threading

from concurrent.futures import ThreadPoolExecutor
from skimage.io import imread
from improc.io import parse_collection, DCAccessor
DCAccessor.register()

from lstree.luigi_utils import ExternalInputFile, monitor_futures
from lstree.config import ExperimentParams
from lstree.deconv.deconv_tasks import DeconvolutionTask
from lstree.segmentation.nuclei_tasks import NucleiSegmentationTask
from lstree.segmentation.cell_tasks import CellSegmentationTask

from improc.regionprops import BaseFeatureExtractor, QuantilesFeatureExtractor, IntensityFeatureExtractor, DistanceTransformFeatureExtractor, SKRegionPropFeatureExtractor, DerivedFeatureCalculator, GlobalFeatureExtractor
from improc.resample import match_spacing
from scipy.ndimage.morphology import distance_transform_edt
from skimage.future import graph
from skimage.measure import block_reduce


class NeighborFeatureExtractor(BaseFeatureExtractor):
    '''Extract a list of neighboring label ids for each label.
    
    Cells further than 1 z-plane away are not considered neighbor.
    '''
    def __init__(self, spacing, *args, **kwargs):

        # override channel target
        try:
            del kwargs['channel_targets']
        except KeyError:
            pass
        super().__init__(channel_targets=None, *args, **kwargs)

        self.spacing = spacing

    def _extract_features(self, labels, _):

        spacing = np.broadcast_to(np.asarray(self.spacing), labels.ndim)

        # downsample lables in xy with max function
        # fills potential 1-2 px gap between labels (cells should still be considered neighbors)
        # speeds up processing
        block_size = tuple(np.ceil(spacing.max() / spacing).astype(int))
        labels = block_reduce(labels, block_size=block_size, func=np.max)
        g = graph.RAG(labels, connectivity=labels.ndim)

        unique_l = np.unique(labels)
        unique_l = unique_l[unique_l != 0]

        neighbors = []
        for l in unique_l:
            neighbors.append(list(filter(None, g.neighbors(l))))

        props = {'label': unique_l, 'neighbors': neighbors}
        return props


def get_organoid_feature_extractor(spacing):

    feature_extractor = GlobalFeatureExtractor(extractors=[
        NeighborFeatureExtractor(label_targets=['cell'], spacing=spacing),
        QuantilesFeatureExtractor(
            label_targets=['nuclei'],
            channel_targets=['nuclei_intensity'],
        ),
        IntensityFeatureExtractor(
            features=['mean', 'std'],
            label_targets=['nuclei'],
            channel_targets=['nuclei_intensity'],
        ),
        IntensityFeatureExtractor(
            features=['mean'],
            label_targets=['nuclei'],
            channel_targets=['dist_to_lumen', 'dist_to_basal'],
        ),
        SKRegionPropFeatureExtractor(features=['volume'],
                                     label_targets=['nuclei', 'cell'],
                                     channel_targets=None,
                                     physical_coords=True,
                                     spacing=spacing),
        SKRegionPropFeatureExtractor(features=['volume'],
                                     label_targets=['epithelium', 'lumen'],
                                     channel_targets=None,
                                     physical_coords=True,
                                     spacing=spacing),
        DistanceTransformFeatureExtractor(
            features=['mean_radius', 'max_radius'],
            label_targets=['nuclei', 'cell'],
            channel_targets=None,
            physical_coords=True,
            spacing=spacing),
    ],
                                               calculators=[])

    def _extract_features(labels, channels):

        # custom props: mean nuclei distance to lumen and outside
        organoid_seg = labels.pop('organoid')
        labels['epithelium'] = (organoid_seg == 2).astype(np.uint8)
        labels['lumen'] = (organoid_seg == 1).astype(np.uint8)

        non_lumen_mask = organoid_seg != 1
        if non_lumen_mask.min() > 0:
            # no lumen segmented, would result in distance to image corner
            channels['dist_to_lumen'] = np.zeros_like(non_lumen_mask,
                                                      dtype=np.float32)
        else:
            channels['dist_to_lumen'] = distance_transform_edt(
                non_lumen_mask, sampling=spacing).astype(np.float32)

        # not required here, something went terribly wrong if no background
        channels['dist_to_basal'] = distance_transform_edt(
            organoid_seg != 0, sampling=spacing).astype(np.float32)

        return feature_extractor(labels, channels)

    return _extract_features


class ExtractFeaturesTask(ExperimentParams, luigi.Task):
    '''
    
    '''

    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')
    out_subdir = luigi.Parameter(description='name of output subdirectory')
    nuclei_subdir = luigi.Parameter(
        description='Input subdirectory containing original nuclei tif files')

    n_threads = luigi.IntParameter(
        4, description='max number of threads for pre/post processing')

    @property
    def resources(self):
        return {
            'pool_workers': self.n_threads,
        }

    def requires(self):
        return {
            'experiment_config':
            ExternalInputFile(
                os.path.join(self.movie_dir, self.movie_config_name)),
            'nuclei_img':
            DeconvolutionTask(movie_dir=self.movie_dir,
                              ch_subdir=self.nuclei_subdir),
            'nuclei_seg':
            NucleiSegmentationTask(movie_dir=self.movie_dir),
            'cell_seg':
            CellSegmentationTask(movie_dir=self.movie_dir),
        }

    def output(self):
        paths = []

        # one csv file per timepoint
        for nuclei_target in self.input()['nuclei_img']:
            dirname, fname = os.path.split(nuclei_target.path)
            dirname, subdir = os.path.split(dirname)
            subdir = self.out_subdir

            mamut_t = int(re.search('T[0-9]{4}', fname)[0][1:])
            fname = 'T{:04d}.csv'.format(mamut_t)

            paths.append(os.path.join(dirname, subdir, fname))

        return [luigi.LocalTarget(p) for p in paths]

    def _extract_frame_feature(self, nuclei_img_target, nuclei_seg_target,
                               cell_seg_target, organoid_seg_target,
                               feature_output):

        logger = logging.getLogger('luigi-interface')
        logger.info('extracting features: {}'.format(feature_output.path))

        if not (int(re.search('T[0-9]{4}', nuclei_img_target.path)[0][1:]) ==
                int(re.search('T[0-9]{4}', nuclei_seg_target.path)[0][1:]) ==
                int(re.search('T[0-9]{4}', cell_seg_target.path)[0][1:]) ==
                int(re.search('T[0-9]{4}', organoid_seg_target.path)[0][1:]) ==
                int(re.search('T[0-9]{4}', feature_output.path)[0][1:])):

            logger.error("while extracting features, timepoints don't match: \
                          \n\t{}\n\t{}\n\t{}\n\t{}\n\t{}".format(
                nuclei_img_target.path, nuclei_seg_target.path,
                cell_seg_target.path, organoid_seg_target.path,
                feature_output.path))

        labels = {
            'nuclei': imread(nuclei_seg_target.path),
            'cell': imread(cell_seg_target.path),
            'organoid': imread(organoid_seg_target.path)
        }

        channels = {'nuclei_intensity': imread(nuclei_img_target.path)}

        props = self.feature_extractor(labels, channels)

        feature_output.makedirs()
        with feature_output.temporary_path() as temp_output_path:
            props.to_csv(temp_output_path, index=False)

    def run(self):

        with open(self.input()['experiment_config'].path, 'r') as f:
            experiment_config = json.load(f)
        self.img_spacing = tuple(experiment_config['spacing'])
        self.feature_extractor = get_organoid_feature_extractor(
            self.img_spacing)

        logger = logging.getLogger('luigi-interface')

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            for nuclei_img_target, nuclei_seg_target, cell_seg_target, organoid_seg_target, feature_output in zip(
                    self.input()['nuclei_img'],
                    self.input()['nuclei_seg'],
                    self.input()['cell_seg']['cell'],
                    self.input()['cell_seg']['organoid'], self.output()):

                if feature_output.exists():
                    logger.info('extracted features already exists: {}'.format(
                        feature_output.path))
                    continue

                futures.append(
                    executor.submit(self._extract_frame_feature,
                                    nuclei_img_target, nuclei_seg_target,
                                    cell_seg_target, organoid_seg_target,
                                    feature_output))
            monitor_futures(futures)


def get_generic_feature_extractor(spacing):

    feature_extractor = GlobalFeatureExtractor(extractors=[
        QuantilesFeatureExtractor(
            label_targets='all',
            channel_targets='all',
        ),
        IntensityFeatureExtractor(
            features=['mean', 'std'],
            label_targets='all',
            channel_targets='all',
        ),
        SKRegionPropFeatureExtractor(features=['volume'],
                                     label_targets='all',
                                     channel_targets=None,
                                     physical_coords=True,
                                     spacing=spacing),
        DistanceTransformFeatureExtractor(
            features=['mean_radius', 'max_radius'],
            label_targets='all',
            channel_targets=None,
            physical_coords=True,
            spacing=spacing),
    ],
                                               calculators=[])

    return feature_extractor


class GenericExtractFeaturesTask(ExperimentParams, luigi.Task):
    '''
    Extract generic features for a configurable combinations labels and channels images.
    Unlike ExtractFeaturesTask, all inputs are external and not dynamically generated
    '''

    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')
    out_subdir = luigi.Parameter(description='name of output subdirectory')
    label_subdirs = luigi.ListParameter(
        description='list labels subdirectories containing')
    raw_channel_subdirs = luigi.ListParameter(
        description=
        'list raw channels subdirectories, features are measured on deconvolved images'
    )
    pattern = luigi.Parameter(
        '{subdir}/{fname}_T{time:04d}.{ext}',
        description='parsing pattern, must include "subdir" and "time" fields')

    n_threads = luigi.IntParameter(
        4, description='max number of threads for pre/post processing')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # luigi create tuples even for ListParameter ???
        self.label_subdirs = list(self.label_subdirs)
        self.raw_channel_subdirs = list(self.raw_channel_subdirs)

    @property
    def resources(self):
        return {
            'pool_workers': self.n_threads,
        }

    def requires(self):
        deconv_tasks = [
            DeconvolutionTask(movie_dir=self.movie_dir, ch_subdir=ch_subdir)
            for ch_subdir in self.raw_channel_subdirs
        ]
        self.deconv_ch_subdirs = [
            t.ch_subdir + t.out_suffix for t in deconv_tasks
        ]

        return {
            'experiment_config':
            ExternalInputFile(
                os.path.join(self.movie_dir, self.movie_config_name)),
            'deconv_chs':
            deconv_tasks
        }

    def output(self):
        df = parse_collection(os.path.join(self.movie_dir, self.pattern),
                              ['subdir', 'time'])
        df_out = df.dc[self.raw_channel_subdirs[0]].reset_index()
        df_out.subdir = self.out_subdir
        df_out.pattern = '{basedir}/{subdir}/T{time:04d}.{ext}'
        df_out.ext = 'csv'

        self.timepoints = df_out.time.tolist()
        return [luigi.LocalTarget(p) for p in df_out.dc.path]

    def _extract_frame_feature(self, out_target, subdf, feature_extractor):

        logger = logging.getLogger('luigi-interface')
        logger.info('extracting features: {}'.format(out_target.path))

        imgs = {
            key: val
            for key, val in zip(self.raw_channel_subdirs, subdf.dc[
                self.deconv_ch_subdirs].dc.read())
        }
        labels = {
            key: val
            for key, val in zip(self.label_subdirs, subdf.dc[
                self.label_subdirs].dc.read())
        }

        props = feature_extractor(labels, imgs)

        out_target.makedirs()
        with out_target.temporary_path() as temp_output_path:
            props.to_csv(temp_output_path, index=False)

    def run(self):
        #TODO could be cleaner with collection parsing task as input
        logger = logging.getLogger('luigi-interface')

        with open(self.input()['experiment_config'].path, 'r') as f:
            experiment_config = json.load(f)
        self.img_spacing = tuple(experiment_config['spacing'])
        feature_extractor = get_generic_feature_extractor(self.img_spacing)

        df = parse_collection(os.path.join(self.movie_dir, self.pattern),
                              ['subdir', 'time'])

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            for out_target, mamut_t in zip(self.output(), self.timepoints):

                if out_target.exists():
                    logger.info('extracted features already exists: {}'.format(
                        out_target.path))
                    continue

                subdf = df.dc[:, mamut_t]
                futures.append(
                    executor.submit(self._extract_frame_feature, out_target,
                                    subdf, feature_extractor))

            monitor_futures(futures)
