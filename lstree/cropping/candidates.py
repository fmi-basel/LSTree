import os
import luigi
import logging
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from skimage.io import imread, imsave

from improc.segmentation import segment_from_projections
from improc.label import find_objects_bb
from skimage.exposure import rescale_intensity

from lstree.luigi_utils import ExternalInputFile, get_task_hash


class CropCandidatesFinder:
    '''Find objects in image and export bounding boxes.
    
    Parameters:
    spacing (int|tuple): voxel size of image stack
    sigma (float): blurring sigma for segmentation
    max_clip_factor (float): max clipping intensity defined relative to the image's mean
    threshold_method (str or callable): accepts one of ['otsu', 'li', 'yen', 'triangle', 'minimum'] or a callable
    '''

    # Note: used class with __call__ because otherwise inner function cannot be pickled when using multiproc

    def __init__(self,
                 spacing=1,
                 sigma=5.,
                 max_clip_factor=None,
                 threshold_method='otsu'):
        self.spacing = spacing
        self.sigma = sigma
        self.max_clip_factor = max_clip_factor
        if self.max_clip_factor is None:
            self.max_clip_factor = 99999999
        self.threshold_method = threshold_method

    def __call__(self, img):

        img = np.clip(img, None, self.max_clip_factor * img.mean())
        mask = segment_from_projections(img,
                                        self.spacing,
                                        sigma=self.sigma,
                                        threshold_method=self.threshold_method)

        bbs = find_objects_bb(mask)

        candidates = [(bb[0].start, bb[0].stop, bb[1].start, bb[1].stop,
                       bb[2].start, bb[2].stop) for bb in bbs]

        df = pd.DataFrame(candidates,
                          columns=[
                              'z_start', 'z_stop', 'x_start', 'x_stop',
                              'y_start', 'y_stop'
                          ])

        xy = rescale_intensity(img.max(axis=0), out_range=np.uint8)
        zy = rescale_intensity(img.max(axis=1), out_range=np.uint8)
        zx = rescale_intensity(img.max(axis=2), out_range=np.uint8)

        return df, (xy, zy, zx)


class CropCandidateParams(luigi.Config):
    out_directory = luigi.Parameter(
        description='base output directory for processed outputs')
    spacing = luigi.TupleParameter(description='z,y,x voxel spacing')
    sigma = luigi.FloatParameter(
        description='smoothing sigma prior thresholding')
    max_clip_factor = luigi.FloatParameter(
        description='maximum intensity clipping at: factor x mean(image)')
    threshold_method = luigi.Parameter(
        description=
        "threshold method one of ['otsu', 'li', 'yen', 'triangle', 'minimum']")
    memory = luigi.FloatParameter(12.,
                                  description="estiamted memory usage in GB")


class FindCropCandidateTask(CropCandidateParams, luigi.Task):
    '''
    Find objects in an image stack from it's MIP along x,y,z axes.
    
    exports bounding boxes and xy, zx, zy MIPs for quick review.
    '''

    img_path = luigi.Parameter(description='Path to the image to process')

    @property
    def resources(self):
        resources = super().resources
        resources.update({'memory': self.memory})
        return resources

    def requires(self):
        return ExternalInputFile(path=self.img_path)

    @property
    def thresh_config_name(self):
        return 'CLIP-{:.2f}_TRESH-{}_SIGMA-{:.2f}'.format(
            self.max_clip_factor, self.threshold_method, self.sigma)

    def output(self):

        outdir = os.path.join(self.out_directory, 'crop_candidates',
                              self.thresh_config_name)
        fname = os.path.basename(self.img_path)
        fname, ext = os.path.splitext(fname)

        return {
            'box':
            luigi.LocalTarget(os.path.join(outdir, 'boxes', fname + '.csv')),
            'xy': luigi.LocalTarget(os.path.join(outdir, 'xy',
                                                 fname + '.jpg')),
            'zx': luigi.LocalTarget(os.path.join(outdir, 'zx',
                                                 fname + '.jpg')),
            'zy': luigi.LocalTarget(os.path.join(outdir, 'zy', fname + '.jpg'))
        }

    def run(self):

        crop_finder = CropCandidatesFinder(self.spacing, self.sigma,
                                           self.max_clip_factor,
                                           self.threshold_method)

        img = imread(self.input().path)
        df, (xy, zy, zx) = crop_finder(img)

        if len(df) <= 0:
            # no object detected --> add a single box at the center
            # that can be edited by hand in the verification step
            c = [s // 2 for s in img.shape]
            half_width = [min(10, s // 2) for s in img.shape]

            df = df.append(
                {
                    'z_start': c[0] - half_width[0],
                    'z_stop': c[0] + half_width[0],
                    'x_start': c[1] - half_width[1],
                    'x_stop': c[1] + half_width[1],
                    'y_start': c[2] - half_width[2],
                    'y_stop': c[2] + half_width[2]
                },
                ignore_index=True)

        self.output()['xy'].makedirs()
        imsave(self.output()['xy'].path, xy, check_contrast=False)

        self.output()['zy'].makedirs()
        imsave(self.output()['zy'].path, zy, check_contrast=False)

        self.output()['zx'].makedirs()
        imsave(self.output()['zx'].path, zx, check_contrast=False)

        # atomic write for the last file
        with self.output()['box'].temporary_path() as tmp_path:
            df.to_csv(tmp_path, index=False)


class BatchedFindCropCandidateTask(CropCandidateParams, luigi.Task):
    '''
    Finds crop candidates and write a h5 files with 3 dataframes:
        - dc: data collection of input image
        - MIPs: data collection of MIPs preview
        - boxes: dataframe with roi boxes for all detected objects
    '''

    datadir = luigi.Parameter()
    pattern = luigi.Parameter()
    ref_channel = luigi.Parameter()
    index = luigi.ListParameter(['channel', 'time'])
    timepoint_step = luigi.IntParameter(1)

    def output(self):

        fname = 'agg_candidates_{}.h5'.format(get_task_hash(self))
        return luigi.LocalTarget(os.path.join(self.out_directory, fname))

    def run(self):

        from improc.io import parse_collection, DCAccessor
        DCAccessor.register()

        df = parse_collection(os.path.join(self.datadir, self.pattern),
                              list(self.index))

        crop_candidates = []
        for path in df.dc[
                self.ref_channel, ::self.timepoint_step].dc.path.tolist():
            crop_candidates.append(
                FindCropCandidateTask(img_path=path,
                                      out_directory=self.out_directory,
                                      spacing=self.spacing,
                                      sigma=self.sigma,
                                      max_clip_factor=self.max_clip_factor,
                                      threshold_method=self.threshold_method,
                                      memory=self.memory))

        yield crop_candidates

        # build collection candidates
        candidates_subdir = 'CLIP-{:.2f}_TRESH-{}_SIGMA-{:.2f}'.format(
            self.max_clip_factor, self.threshold_method, self.sigma)
        candidates_dir = os.path.join(self.out_directory, 'crop_candidates',
                                      candidates_subdir)
        candidates = '{subdir}/' + os.path.splitext(
            os.path.basename(self.pattern))[0] + '.{ext}'
        df_candidates = parse_collection(
            os.path.join(candidates_dir, candidates), ['subdir', 'time'])

        df_boxes = pd.concat(
            {
                t: b
                for t, b in zip(
                    df_candidates.dc['boxes'].index.get_level_values('time'),
                    df_candidates.dc['boxes'].dc.read())
            },
            ignore_index=False,
            names=['time', 'bb_id'])

        df_mip = df_candidates.dc[['xy', 'zx', 'zy']]

        self.output().makedirs()
        with self.output().temporary_path() as path:
            df.to_hdf(path, key='dc')
            df_mip.to_hdf(path, key='MIPs')
            df_boxes.to_hdf(path, key='boxes')


def link_crop_candidates(dfb, spacing=1, margin=10, start_tp=None):
    '''Links pre-computed object bouding boxes starting from the largest object on the last frame or given timepoint
    
    Args:
        - dfb: dataframe of bounding boxes with time,bb_id as index
    '''

    spacing = np.broadcast_to(np.asarray(spacing), 3)
    margin = np.broadcast_to(np.asarray(margin), 3)

    def closest_point(c, centers):
        distances = cdist(c * spacing, centers * spacing)
        return np.argmin(distances)

    dfb['x_center'] = (dfb.x_stop + dfb.x_start) // 2
    dfb['y_center'] = (dfb.y_stop + dfb.y_start) // 2
    dfb['z_center'] = (dfb.z_stop + dfb.z_start) // 2

    last_center = None

    tps = dfb.index.levels[0]

    if 'linked' not in dfb.columns:
        dfb['linked'] = 0
    elif start_tp is not None:
        if dfb.groupby('time')['linked'].agg('max').min() < 1:
            raise RuntimeError(
                'Linked bounding boxes improperly initialized. Some timepoints have no selected boxes'
            )

        tps = [t for t in tps if t < start_tp]
        center_idx = dfb.loc[start_tp, 'linked'].argmax()
        last_center = [
            dfb.loc[(start_tp, center_idx),
                    ['z_center', 'x_center', 'y_center']].values
        ]

    for tp in tps[::-1]:

        # reset link for current timepoint
        dfb.loc[tp, 'linked'] = 0

        if last_center is None:
            last_center = [
                dfb.loc[(tp, 0), ['z_center', 'x_center', 'y_center']].values
            ]
            dfb.loc[(tp, 0), 'linked'] = 1
        else:
            candidate_c = dfb.loc[tp,
                                  ['z_center', 'x_center', 'y_center']].values
            closest_idx = closest_point(last_center, candidate_c)
            last_center = candidate_c[closest_idx:closest_idx + 1]
            dfb.loc[(tp, closest_idx), 'linked'] = 1

    # add movie bb
    max_bb_size = (dfb.loc[dfb.linked==True, ['z_stop', 'x_stop', 'y_stop']].values - \
                   dfb.loc[dfb.linked==True, ['z_start', 'x_start', 'y_start']].values).max(axis=0)

    max_bb_size += (2 * margin)

    dfb['z_start_movie'] = np.round(dfb['z_center'] -
                                    max_bb_size[0] // 2).astype(int)
    dfb['z_stop_movie'] = np.round(dfb['z_center'] +
                                   max_bb_size[0] // 2).astype(int)
    dfb['x_start_movie'] = np.round(dfb['x_center'] -
                                    max_bb_size[1] // 2).astype(int)
    dfb['x_stop_movie'] = np.round(dfb['x_center'] +
                                   max_bb_size[1] // 2).astype(int)
    dfb['y_start_movie'] = np.round(dfb['y_center'] -
                                    max_bb_size[2] // 2).astype(int)
    dfb['y_stop_movie'] = np.round(dfb['y_center'] +
                                   max_bb_size[2] // 2).astype(int)

    # update box size
    dfb['x_box_size'] = dfb.x_stop - dfb.x_start
    dfb['y_box_size'] = dfb.y_stop - dfb.y_start
    dfb['z_box_size'] = dfb.z_stop - dfb.z_start

    # compute physical coordinates
    dfb[['z_start_phy', 'x_start_phy',
         'y_start_phy']] = dfb[['z_start', 'x_start', 'y_start']] * spacing
    dfb[['z_stop_phy', 'x_stop_phy',
         'y_stop_phy']] = dfb[['z_stop', 'x_stop', 'y_stop']] * spacing

    dfb[[
        'z_start_movie_phy', 'x_start_movie_phy', 'y_start_movie_phy'
    ]] = dfb[['z_start_movie', 'x_start_movie', 'y_start_movie']] * spacing
    dfb[['z_stop_movie_phy', 'x_stop_movie_phy', 'y_stop_movie_phy'
         ]] = dfb[['z_stop_movie', 'x_stop_movie', 'y_stop_movie']] * spacing

    return dfb
