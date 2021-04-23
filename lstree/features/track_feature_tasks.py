import luigi
import os
import json
import logging
import pandas as pd
import numpy as np
import threading
from glob import glob

# ~from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from concurrent.futures import ProcessPoolExecutor as PoolExecutor
from skimage.io import imread, imsave
from scipy.ndimage.measurements import center_of_mass
from sklearn.metrics import confusion_matrix

from lstree.config import ExperimentParams
from lstree.lineage.tracking_tasks import NucleiTrackingTask
from lstree.luigi_utils import ExternalInputFile, monitor_futures
from lstree.lineage.utils import LSCoordinates, compute_dendrogram_coordinates
from lstree.lineage.ltree import LineageTree
from lstree.lineage.utils import repack_spatial_coordinates, label_split_cells
from lstree.lineage.plot import tree_to_dataframe, plot_tree
from lstree.lineage.parse_ltree import write_tree_mamut


def extract_tracking_features(segm, segm_parent, link_scores, spacing):

    voxel_volume = np.prod(spacing)

    # init dataframe with id and volume
    unique_l, count = np.unique(segm, return_counts=True)
    unique_l, count = unique_l[unique_l > 0], count[unique_l >
                                                    0]  # exclude background
    seeds = pd.DataFrame({
        'label_id': unique_l,
        'volume': count * voxel_volume
    })
    seeds['parent_label_id'] = 0
    seeds = seeds.join(link_scores.set_index('label_id'), on='label_id')

    if len(seeds) > 0:
        # add cm coordinate
        cms = center_of_mass(segm > 0, segm, unique_l)
        cms = np.round(cms).astype(int)
        seeds['z'] = 0
        seeds['y'] = 0
        seeds['x'] = 0
        seeds[['z', 'y', 'x']] = cms

        # add conversion to physical coords
        for idx, sr in seeds.iterrows():
            mamut_c = LSCoordinates.from_img((sr[['z', 'y', 'x']]),
                                             voxel_spacing=spacing).mamut
            seeds.loc[idx, 'mamut_x'] = mamut_c[0]
            seeds.loc[idx, 'mamut_y'] = mamut_c[1]
            seeds.loc[idx, 'mamut_z'] = mamut_c[2]

    if segm.max() > 0 and segm_parent.max() > 0:
        # find parent id
        intersection = confusion_matrix(
            segm.flat,
            segm_parent.flat,
            labels=range(np.maximum(segm.max(), segm_parent.max()) + 1))
        intersection = intersection[:segm.max() + 1, :segm_parent.max() + 1]

        segm_sum = np.bincount(segm.flat)
        segm_parent_sum = np.bincount(segm_parent.flat)
        union = segm_sum[..., None] + segm_parent_sum
        union -= intersection
        iou = intersection / union

        # find parent with maximum overlap
        max_overlap_ids = np.argmax(intersection, axis=1)
        seeds['parent_label_id'] = max_overlap_ids[seeds.label_id]

        max_overlap_count = intersection.max(axis=1)
        seeds['overlap_to_parent'] = max_overlap_count[seeds.label_id] / count
        seeds['link_iou'] = iou[tuple(seeds.label_id),
                                tuple(seeds.parent_label_id)]

    return seeds


def build_tree(features_paths, max_n_nuclei=None):
    '''Builds a tree from predicted seeds'''

    ids_count = 0
    nodes = {}

    for p in features_paths:

        seeds = pd.read_csv(p)

        if max_n_nuclei is not None and len(seeds) > max_n_nuclei:
            break

        # start index from last id + 1
        seeds['mamut_id'] = seeds.label_id + ids_count
        ids_count += len(seeds)
        seeds['node_id'] = seeds['mamut_id']
        seeds.set_index('mamut_id', inplace=True)
        nodes.update(seeds.to_dict(orient='index'))

    nodes = [(key, val) for key, val in nodes.items()]
    tree = LineageTree(nodes=nodes, edges=[], time_key='mamut_t')
    repack_spatial_coordinates(tree, 'mamut_x', 'mamut_y', 'mamut_z')

    # add edges
    reverse_index = {(attr['mamut_t'], attr['label_id']): nid
                     for nid, attr in tree.nodes}

    for nid, attr in tree.nodes:
        parent_id = reverse_index.get(
            (attr['mamut_t'] - 1, attr['parent_label_id']))
        if parent_id is not None:
            tree.add_edge(parent_id, nid)

    tree.correct_edge_direction()

    return tree


def remove_short_tracklets(tree, min_track_length):
    nodes_to_remove = []
    for n_id, _ in tree.nodes:
        if len(list(tree.predecessors(n_id))) < 1:
            children = tree.successor_track_nodes(n_id)
            if len(children) < min_track_length - 1:
                nodes_to_remove.extend(children)
                nodes_to_remove.append(n_id)

    tree.delete_nodes(nodes_to_remove)


def clean_tree(tree, min_track_length=25):
    '''Removes short tracks, including spurious short tracks after splits'''

    label_split_cells(tree)

    nodes_to_remove = []
    for n_id, _ in tree.filter('split', lambda x: x == True).nodes:

        for successor_n_id in tree.successors(n_id):

            children = tree.successor_track_nodes(successor_n_id)

            if len(children) < min_track_length:
                nodes_to_remove.append(successor_n_id)
                nodes_to_remove.extend(children)

    #TODO do not access underlying networkx graph directly, write method in LineageTree
    tree.g.remove_nodes_from(nodes_to_remove)
    remove_short_tracklets(tree, min_track_length)


class ExtractTrackingFeaturesTask(ExperimentParams, luigi.Task):
    '''extract features on tracked nuclei including center mass, volume and link to previous'''

    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')
    out_subdir = luigi.Parameter(description='name of output subdirectory')

    #NOTE uses multiprocessing internally as there is almost no gain with threads (i.e. scikit-learn confusion_matrix() doesn't release the GIL)
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
            'tracks':
            NucleiTrackingTask(movie_dir=self.movie_dir),
        }

    def output(self):
        paths = [
            inp.path.replace(os.sep + inp.path.split(os.sep)[-2] + os.sep,
                             os.sep + self.out_subdir + os.sep)
            for inp in self.input()['tracks']['score']
        ]
        return [luigi.LocalTarget(p) for p in paths]

    def _extract_frame_track_feature(self, sample_idx, output_target):

        logger = logging.getLogger('luigi-interface')
        logger.info('extracting tracking features: {}'.format(
            output_target.path))

        segm = imread(self.input()['tracks']['nuclei'][sample_idx].path)
        if sample_idx == 0:
            # no previous frame to link to
            segm_parent = np.zeros_like(segm)
        else:
            segm_parent = imread(self.input()['tracks']['link'][sample_idx -
                                                                1].path)
        link_scores = pd.read_csv(
            self.input()['tracks']['score'][sample_idx].path)

        track_props = extract_tracking_features(segm, segm_parent, link_scores,
                                                self.img_spacing)

        output_target.makedirs()
        with output_target.temporary_path() as tmp_path:
            track_props.to_csv(tmp_path, index=False)

    def run(self):

        with open(self.input()['experiment_config'].path, 'r') as f:
            experiment_config = json.load(f)
        self.img_spacing = tuple(experiment_config['spacing'])

        logger = logging.getLogger('luigi-interface')

        with PoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            for sample_idx, output_target in enumerate(self.output()):

                if output_target.exists():
                    logger.info('tracking features already exist: {}'.format(
                        output_target.path))
                    continue

                futures.append(
                    executor.submit(self._extract_frame_track_feature,
                                    sample_idx, output_target))
            monitor_futures(futures)


class BuildTreeTask(ExperimentParams, luigi.Task):
    '''Aggregates tree features into a predicted tree'''

    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')
    max_n_nuclei = luigi.IntParameter(
        description=
        'stop predicting tree after reaching a timepoint with this many nuclei'
    )
    min_track_length = luigi.IntParameter(
        description='minimum track length to consider as part of the tree')
    xml_bdv = luigi.Parameter(
        description='big data viewer xml dataset filename')

    def requires(self):
        return {
            'experiment_config':
            ExternalInputFile(
                os.path.join(self.movie_dir, self.movie_config_name)),
            'tracks':
            ExtractTrackingFeaturesTask(movie_dir=self.movie_dir)
        }

    def output(self):
        return {
            'h5_tree':
            luigi.LocalTarget(os.path.join(self.movie_dir, 'tree_pred.h5')),
            'xml_tree':
            luigi.LocalTarget(os.path.join(self.movie_dir, 'tree_pred.xml')),
            'html_tree':
            luigi.LocalTarget(os.path.join(self.movie_dir, 'tree_pred.html'))
        }

    def run(self):
        import holoviews as hv
        from holoviews import opts, dim
        import panel as pn
        from inter_view.color import clipped_plasma_r
        hv.extension('bokeh')

        opts.defaults(
            opts.Overlay('tree',
                         yaxis=None,
                         width=900,
                         height=1000,
                         show_title=True),
            opts.Points('tree', size=3, tools=['hover'],
                        cmap=clipped_plasma_r),
            opts.Segments('tree', line_width=1, color='black'),
        )

        with open(self.input()['experiment_config'].path, 'r') as f:
            experiment_config = json.load(f)
        self.img_spacing = tuple(experiment_config['spacing'])

        features_paths = sorted(
            [target.path for target in self.input()['tracks']])
        tree = build_tree(features_paths, self.max_n_nuclei)
        clean_tree(tree, self.min_track_length)
        compute_dendrogram_coordinates(tree)
        dfn, dfe = tree_to_dataframe(tree)

        # TODO atomic write in case this is changed to be the last output
        # hv.save adds '.html' extension to provided path and breaks luigi temporary path cleanup
        hv_tree_volume = plot_tree(dfn,
                                   dfe,
                                   coord_x='mamut_t',
                                   node_color='volume').relabel(label='Volume')
        hv_tree_dist = plot_tree(dfn,
                                 dfe,
                                 coord_x='mamut_t',
                                 node_color='linking_distance').relabel(
                                     label='Linking distance to children')
        # manually clip node color (not supported by plotting fun)
        hv_tree_dist.opts(
            opts.Points(color=dim('linking_distance').clip(max=20.).norm()))

        hv_tree_overlap = plot_tree(
            dfn, dfe, coord_x='mamut_t',
            node_color='overlap_to_parent').relabel(label='Overlap to parent')

        hv_tree = (hv_tree_volume + hv_tree_dist + hv_tree_overlap).cols(1)

        hv.save(hv_tree,
                self.output()['html_tree'].path,
                fmt='html',
                backend='bokeh',
                toolbar=True)

        with self.output()['h5_tree'].temporary_path() as temp_output_path:
            dfn.to_hdf(temp_output_path, key='nodes')
            dfe.to_hdf(temp_output_path, key='edges')

        with self.output()['xml_tree'].temporary_path() as temp_output_path:
            write_tree_mamut(tree,
                             output_path=temp_output_path,
                             bdv_xml_path=os.path.join(self.movie_dir,
                                                       self.xml_bdv),
                             sampling=self.img_spacing)


class MultiBuildTreeTask(luigi.WrapperTask):

    movie_dirs = luigi.ListParameter(
        description='List of movie directories or glob patterns')

    def _expand_movie_dirs(self):
        movie_dirs = []
        for movie_dir in self.movie_dirs:
            movie_dirs.extend(glob(movie_dir))

        return sorted(movie_dirs)

    def requires(self):

        for movie_dir in self._expand_movie_dirs():
            yield BuildTreeTask(movie_dir=movie_dir)
