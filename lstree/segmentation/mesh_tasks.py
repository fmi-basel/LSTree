import luigi
import os
import numpy as np
import re
import logging
import json
from glob import glob
import threading
import itertools
import pyvista as pv

from concurrent.futures import ThreadPoolExecutor
from skimage.io import imread
from skimage.exposure import rescale_intensity
from improc.mesh import labels_to_mesh, add_mesh_points_attribute, export_vtk_polydata
from improc.io import parse_collection, DCAccessor
from inter_view.utils import make_composite
import inter_view.color

from lstree.config import ExperimentParams
from lstree.luigi_utils import ExternalInputFile, monitor_futures
from lstree.deconv.deconv_tasks import DeconvolutionTask
from lstree.segmentation.nuclei_tasks import NucleiSegmentationTask
from lstree.segmentation.cell_tasks import CellSegmentationTask
from lstree.features.agg_feature_tasks import MultiAggregateFeaturesTask
from lstree.features.agg_feature_tasks import MultiAggregateOrganoidFeaturesTask

DCAccessor.register()


def rgb_stack_to_grid(rgb, spacing, bounds=None):

    if bounds is None:
        bounds = np.array([[0, s] for s in rgb.shape[:-1]])

    loc = [slice(start, stop) for start, stop in bounds]
    origin = [start for start, stop in bounds]

    rgb = rgb[loc]
    rgb = np.moveaxis(rgb, [0, 1, 2, 3], [2, 1, 0, 3])

    grid = pv.UniformGrid()
    grid.dimensions = rgb.shape[:-1]
    grid.origin = (np.array(origin) * spacing)[::-1]
    grid.spacing = spacing[::-1]

    grid.point_data["values"] = rgb.reshape(-1, 3, order="F")

    return grid


class SegmentationMeshTask(ExperimentParams, luigi.Task):

    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')
    in_subdir = luigi.Parameter(description='segmentation input subdirectory')
    out_subdir = luigi.Parameter(description='meshes output subdirectory')
    pattern = luigi.Parameter(
        '{subdir}/{fname}_T{time:04d}.{ext}',
        description='parsing pattern, must include "subdir" and "time" fields')

    n_threads = luigi.IntParameter(
        3, description='max number of threads for pre/post processing')

    @property
    def resources(self):
        resources = super().resources
        resources.update({'pool_workers': self.n_threads})
        return resources

    def requires(self):
        df = parse_collection(os.path.join(self.movie_dir, self.pattern),
                              ['subdir', 'time'])
        label_paths = df.dc[self.in_subdir].dc.path

        return {
            'experiment_config':
            ExternalInputFile(
                os.path.join(self.movie_dir, self.movie_config_name)),
            'seg': [ExternalInputFile(p) for p in label_paths]
        }

    def output(self):
        out_paths = [
            inp.path.replace(os.sep + inp.path.split(os.sep)[-2] + os.sep,
                             os.sep + self.out_subdir + os.sep) \
                    .replace('.tif', '.vtp')
            for inp in self.input()['seg']
        ]

        return [luigi.LocalTarget(p) for p in out_paths]

    def _label_to_mesh(self, label_target, mesh_target):

        logger = logging.getLogger('luigi-interface')
        logger.info('extracting meshes: {}'.format(label_target.path))

        labels = imread(label_target.path)

        mesh_polydata = labels_to_mesh(labels,
                                       self.spacing,
                                       smoothing_iterations=100,
                                       pass_band_param=0.01,
                                       target_reduction=0.7,
                                       show_progress=False)

        mesh_target.makedirs()
        with mesh_target.temporary_path() as temp_output_path:

            export_vtk_polydata(temp_output_path, mesh_polydata)

    def run(self):
        logger = logging.getLogger('luigi-interface')

        with open(self.input()['experiment_config'].path, 'r') as f:
            experiment_config = json.load(f)
            self.spacing = experiment_config['spacing']

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            for label_target, mesh_target in zip(self.input()['seg'],
                                                 self.output()):
                if mesh_target.exists():
                    logger.info('mesh output already exist: {}'.format(
                        mesh_target.path))
                    continue

                assert os.path.splitext(os.path.basename(
                    label_target.path))[0] == os.path.splitext(
                        os.path.basename(mesh_target.path))[0]

                futures.append(
                    executor.submit(self._label_to_mesh, label_target,
                                    mesh_target))

            monitor_futures(futures)


class VolumeGridTask(ExperimentParams, luigi.Task):
    '''Export RGB image volume as vti files'''

    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')
    raw_channel_subdirs = luigi.ListParameter(
        description=
        'list raw channels subdirectories, deconvolved images are used to build the grid'
    )
    colormaps = luigi.ListParameter(
        description='list of colormaps for each channel')
    blending_mode = luigi.Parameter(
        description='Blending mode when making color composite max|mean')
    ref_mesh_subdir = luigi.Parameter(
        description='reference mesh to determine clipping bounds')
    pattern = luigi.Parameter(
        '{subdir}/{fname}_T{time:04d}.{ext}',
        description=
        'parsing pattern, must include "subdir", "time", "fname" fields')

    out_subdir = luigi.Parameter(description='grids output subdirectory')
    crop_margin = luigi.IntParameter(
        5,
        description='margins outside of cell mesh used to crop the image grid')
    n_threads = luigi.IntParameter(
        3, description='max number of threads for pre/post processing')

    @property
    def resources(self):
        resources = super().resources
        resources.update({'pool_workers': self.n_threads})
        return resources

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
            deconv_tasks,
        }

    def output(self):
        df = parse_collection(os.path.join(self.movie_dir, self.pattern),
                              ['subdir', 'time'])
        df_out = df.dc[self.raw_channel_subdirs[0]].reset_index()
        df_out.subdir = self.out_subdir
        df_out.fname = 'grid'
        df_out.ext = 'vti'

        self.timepoints = df_out.time.tolist()
        return [luigi.LocalTarget(p) for p in df_out.dc.path]

    def _get_organoid_bounds(self, ref_mesh):

        bounds = np.array(ref_mesh.bounds).reshape(3, 2)[::-1]
        bounds = np.array(
            [[max(start - self.crop_margin, 0), stop + self.crop_margin]
             for start, stop in bounds])

        bounds /= np.asarray(self.spacing)[..., None]
        return bounds.round().astype(int)

    def _img_to_grid(self, out_target, subdf):

        logger = logging.getLogger('luigi-interface')
        logger.info('exporting img grid: {}'.format(out_target.path))

        # ~print('\n\n', nuclei_target.path, '\n', cell_target.path, '\n', cell_mesh_target.path, '\n', grid_target.path,'\n\n')

        imgs = subdf.dc[self.deconv_ch_subdirs].dc.read()
        ref_mesh = pv.read(subdf.dc[self.ref_mesh_subdir].dc.path[0])

        imgs = [
            rescale_intensity(img,
                              in_range=tuple(np.quantile(img, [0.2, 0.9999])),
                              out_range=np.uint8) for img in imgs
        ]
        rgb_img = make_composite(imgs, self.colormaps, self.blending_mode)

        bounds = self._get_organoid_bounds(ref_mesh)
        grid = rgb_stack_to_grid(rgb_img, self.spacing, bounds)

        out_target.makedirs()
        # NOTE luigi temporary_path adds a suffix --> change extension --> incompatible with pyvista
        tmp_path = os.path.split(out_target.path)
        tmp_path = os.path.join(tmp_path[0], 'tmp_' + tmp_path[1])
        grid.save(tmp_path)
        os.rename(tmp_path, out_target.path)

    def run(self):
        logger = logging.getLogger('luigi-interface')

        with open(self.input()['experiment_config'].path, 'r') as f:
            experiment_config = json.load(f)
            self.spacing = experiment_config['spacing']

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
                    executor.submit(self._img_to_grid, out_target, subdf))

            monitor_futures(futures)


class ViewerTask(luigi.Task):

    movie_dirs = luigi.ListParameter(
        description='List of movie directories or glob patterns')
    nuclei_seg_subdir = luigi.Parameter(
        description='nuclei segmentation subdirectory')
    cell_seg_subdir = luigi.Parameter(
        description='cell segmentation subdirectory')
    _completed = False

    def _expand_movie_dirs(self):
        movie_dirs = []
        for movie_dir in self.movie_dirs:
            movie_dirs.extend(glob(movie_dir))

        return sorted(movie_dirs)

    def complete(self):
        return self._completed

    def run(self):

        #NOTE connects several tasks that are not fully linked togetherthrough their dependency
        # this is need to allow some flexibility with external inputs
        # order in which they are run matters

        yield MultiAggregateFeaturesTask()
        yield MultiAggregateOrganoidFeaturesTask()

        # nuclei and cell meshes
        mesh_tasks = []
        for movie_dir in self._expand_movie_dirs():
            mesh_tasks.append(
                SegmentationMeshTask(movie_dir=movie_dir,
                                     in_subdir=self.nuclei_seg_subdir,
                                     out_subdir='nuclei_mesh'))
            mesh_tasks.append(
                SegmentationMeshTask(movie_dir=movie_dir,
                                     in_subdir=self.cell_seg_subdir,
                                     out_subdir='cell_mesh'))
        yield mesh_tasks

        for movie_dir in self._expand_movie_dirs():
            yield VolumeGridTask(movie_dir=movie_dir)

        self._completed = True
