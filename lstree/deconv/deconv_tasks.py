import luigi
import os
import json
import logging
import itertools
import numpy as np
from glob import glob

from concurrent.futures import ThreadPoolExecutor
from skimage.io import imread, imsave

from lstree.luigi_utils import ExternalInputFile, GPUTask, monitor_futures
from lstree.config import ExperimentParams
from lstree.deconv.denoise_tasks import DenoiseTask


class DeconvolutionTask(ExperimentParams, GPUTask):
    psf_dir = luigi.Parameter(
        description='folder containing psf for each wavelength / magnification'
    )
    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')
    ch_subdir = luigi.Parameter(
        description='Input subdirectory containing original tif files')
    out_suffix = luigi.Parameter(
        description='Suffix to add to create the output subdirectory')
    n_threads = luigi.IntParameter(
        3,
        description='max number of threads (e.g. for compressed tif export)')

    n_iter = luigi.IntParameter(128, description='number of iterations')
    max_patch_size = luigi.TupleParameter(
        (512, 512, 512),
        description=
        'Tiled processing with max_patch_size if the image is larger')
    pad_size = luigi.TupleParameter(
        (16, 16, 16), description=' padd size for tiled processing')

    @property
    def resources(self):
        resources = super().resources
        resources.update({'pool_workers': self.n_threads})
        return resources

    def requires(self):
        return {
            'experiment_config':
            ExternalInputFile(
                os.path.join(self.movie_dir, self.movie_config_name)),
            'images':
            DenoiseTask(movie_dir=self.movie_dir, ch_subdir=self.ch_subdir)
        }

    def output(self):
        paths = [
            inp.path.replace(
                os.sep + inp.path.split(os.sep)[-2] + os.sep,
                os.sep + self.ch_subdir + self.out_suffix + os.sep)
            for inp in self.input()['images']
        ]
        return [luigi.LocalTarget(p) for p in paths]

    def _save_prediction(self, pred, output):
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
        from lstree.deconv.deconv import tiled_deconv

        logger = logging.getLogger('luigi-interface')

        # load psf
        try:
            with open(self.input()['experiment_config'].path, 'r') as f:
                experiment_config = json.load(f)

            mag = experiment_config['mag']
            wavelength = experiment_config['wavelengths'][self.ch_subdir]
            psf_path = os.path.join(self.psf_dir,
                                    '{}nm_{}X.tif'.format(wavelength, mag))
            psf = imread(psf_path)

        except Exception as e:
            logger.error('unable to load the PSF: {}'.format(psf_path))
            raise e

        # use multithreading to write images (bottleneck when compression enabled)
        # gpu almost fully loaded without multithreaded read --> keep it simple
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            for input, output in zip(self.input()['images'], self.output()):
                if output.exists():
                    logger.info('deconvolve output already exist: {}'.format(
                        output.path))
                    continue

                logger.info('deconvolving: {}'.format(input.path))
                raw = imread(input.path)

                pred = tiled_deconv(raw,
                                    psf,
                                    niter=self.n_iter,
                                    max_patch_size=self.max_patch_size,
                                    pad_size=self.pad_size)
                pred = pred.clip(0, 2**16 - 1).astype(np.uint16)

                output.makedirs()
                futures.append(
                    executor.submit(self._save_prediction, pred, output))

            monitor_futures(futures)


class MultiDeconvolutionTask(luigi.WrapperTask):

    ch_subdirs = luigi.ListParameter(
        description='List of channel sub-directories')
    movie_dirs = luigi.ListParameter(
        description='List of movie directories or glob patterns')

    def _expand_movie_dirs(self):
        ch_paths = []
        for movie_dir, ch_subdir in itertools.product(self.movie_dirs,
                                                      self.ch_subdirs):
            pattern = os.path.join(movie_dir, ch_subdir)
            ch_paths.extend(glob(pattern))

        ch_paths = sorted(ch_paths)
        return [os.path.split(p) for p in ch_paths]

    def requires(self):

        for movie_dir, ch_subdir in self._expand_movie_dirs():
            yield DeconvolutionTask(ch_subdir=ch_subdir, movie_dir=movie_dir)
