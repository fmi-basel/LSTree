import luigi
import os
import json
import re
import pandas as pd
import numpy as np
from glob import glob

from lstree.luigi_utils import ExternalInputFile
from lstree.features.feature_tasks import ExtractFeaturesTask, GenericExtractFeaturesTask
from lstree.lineage.tree_tasks import TreePropsTask
from lstree.config import ExperimentParams
from lstree.lineage.parse_ltree import construct_tree
from lstree.lineage.utils import calculate_displacement


def flatten_features(props):
    '''Converts extracted features into a dataframe with 1 column per feature'''

    props = props.pivot_table(index=['object_id'],
                              columns=['region', 'channel', 'feature_name'],
                              values='feature_value',
                              aggfunc='first')
    props.columns = [
        '{}_{}_{}'.format(*idx).replace('_na', '')
        for idx in props.columns.to_flat_index()
    ]

    # convert columns that can be to numeric (i.e. everything except list of neighbor cells)
    props = props.apply(pd.to_numeric, errors='ignore')

    return props


def reshape_nuclei_props(props, mamut_t):
    '''Reorganizes extracted features and set global index'''

    props = flatten_features(props)
    props = props.reset_index().rename({'object_id': 'label_id'}, axis=1)
    props['timepoint_id'] = props['label_id'] - 1
    props['mamut_t'] = mamut_t
    props.set_index(['mamut_t', 'timepoint_id'], inplace=True)

    return props


def reshape_organoid_props(props, mamut_t, time_interval):
    props = flatten_features(props)

    props['mamut_t'] = mamut_t

    if not 'epithelium_volume' in props:
        props['epithelium_volume'] = 0
    if not 'lumen_volume' in props:
        props['lumen_volume'] = 0

    props['organoid_volume'] = props.epithelium_volume + props.lumen_volume
    props['time'] = props['mamut_t'] * time_interval
    props.set_index('mamut_t', inplace=True)

    return props


def convert_cell_neighbor_ids_to_mamut(dfn):
    '''Maps neighbhor timepoint ids to gloabl mamut ids'''

    lut = dfn[['mamut_id', 'mamut_t',
               'label_id']].set_index(['mamut_t', 'label_id'])
    dfn['cell_neighbors'] = [
        lut.loc(axis=0)[t,
                        nb].mamut_id.tolist() if isinstance(nb, list) else []
        for nb, t in zip(dfn.cell_neighbors, dfn.mamut_t)
    ]


def MixTypeParser(data):
    '''pandas parse with json. e.g. list of cell neighbors'''

    import json
    if data:
        data = json.loads(data)
    else:
        data = np.nan
    return data


class AggregateFeaturesTask(ExperimentParams, luigi.Task):
    '''
    Maps extracted features to tracked nuclei and export a single h5 file
    '''

    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')
    extractor_type = luigi.Parameter(
        'nuclei-membrane',
        description=
        'either "nuclei-membrane" for the standard pipeline or "generic" to extract features on external files'
    )

    _extractor_tasks = {
        'nuclei-membrane': ExtractFeaturesTask,
        'generic': GenericExtractFeaturesTask
    }

    def requires(self):
        # NOTE we need to reopen the tree to extract edges
        tree_task = TreePropsTask(movie_dir=self.movie_dir)
        return {
            'experiment_config':
            ExternalInputFile(
                os.path.join(self.movie_dir, self.movie_config_name)),
            'features':
            self._extractor_tasks[self.extractor_type](
                movie_dir=self.movie_dir),
            'tree_props':
            tree_task,
            'xml_tree':
            ExternalInputFile(
                path=os.path.join(self.movie_dir, tree_task.xml_tree))
        }

    def output(self):
        return luigi.LocalTarget(
            os.path.join(self.movie_dir, 'agg_features.h5'))

    def run(self):
        with open(self.input()['experiment_config'].path, 'r') as f:
            experiment_config = json.load(f)

        tree = construct_tree(self.input()['xml_tree'].path)
        calculate_displacement(tree)

        tree_props_paths = glob(
            os.path.join(self.input()['tree_props'].path, '*.csv'))
        features_paths = sorted(
            [target.path for target in self.input()['features']])

        # load tree features
        dfn = pd.concat([
            pd.read_csv(p,
                        index_col=['mamut_t', 'timepoint_id'],
                        converters={'feature_value': MixTypeParser})
            for p in tree_props_paths
        ]).sort_index()
        tracked_timepoints = dfn.index.levels[0].unique()

        all_props = []
        all_organoid_props = []

        for p in features_paths:

            features = pd.read_csv(p,
                                   converters={'feature_value': MixTypeParser})
            if features.empty:
                continue

            mamut_t = int(re.search('T[0-9]{4}', p)[0][1:])
            if mamut_t in tracked_timepoints:
                props = features[~features.region.isin(['epithelium', 'lumen']
                                                       )]
                props = reshape_nuclei_props(props, mamut_t)
                all_props.append(props)

            organoid_props = features[features.region.isin(
                ['epithelium', 'lumen'])]
            organoid_props = reshape_organoid_props(
                organoid_props, mamut_t, experiment_config['time_interval'])
            all_organoid_props.append(organoid_props)

        if len(all_props) > 0:
            all_props = pd.concat(all_props)
            all_organoid_props = pd.concat(all_organoid_props)

            # combine tree and cell props into nodes dataframe
            dfn = dfn.join(all_props).reset_index().set_index('mamut_id',
                                                              drop=False)

            convert_cell_neighbor_ids_to_mamut(dfn)

        else:
            all_organoid_props = pd.DataFrame()
            dfn = dfn.reset_index().set_index('mamut_id', drop=False)

        # get edges (including merge)
        dfe = pd.DataFrame([{
            'node_start': n[0],
            'node_stop': n[1],
            **n[2]
        } for n in tree.subtree(dfn.index).edges])

        with self.output().temporary_path() as temp_output_path:
            dfn.to_hdf(temp_output_path, key='nodes')
            dfe.to_hdf(temp_output_path, key='edges')
            all_organoid_props.to_hdf(temp_output_path, key='organoid')


class AggregateOrganoidFeaturesTask(ExperimentParams, luigi.Task):
    '''
    Aggregates organoid level features and export a single h5 file.
    Does not require the tracking tree
    '''

    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')

    def requires(self):

        return {
            'experiment_config':
            ExternalInputFile(
                os.path.join(self.movie_dir, self.movie_config_name)),
            'features':
            ExtractFeaturesTask(movie_dir=self.movie_dir)
        }

    def output(self):
        return luigi.LocalTarget(
            os.path.join(self.movie_dir, 'agg_features.h5'))

    def run(self):
        with open(self.input()['experiment_config'].path, 'r') as f:
            experiment_config = json.load(f)

        features_paths = sorted(
            [target.path for target in self.input()['features']])
        all_organoid_props = []

        for p in features_paths:

            features = pd.read_csv(p,
                                   converters={'feature_value': MixTypeParser})
            if features.empty:
                continue

            mamut_t = int(re.search('T[0-9]{4}', p)[0][1:])
            organoid_props = features[features.region.isin(
                ['epithelium', 'lumen'])]
            organoid_props = reshape_organoid_props(
                organoid_props, mamut_t, experiment_config['time_interval'])
            all_organoid_props.append(organoid_props)

        all_organoid_props = pd.concat(all_organoid_props)

        with self.output().temporary_path() as temp_output_path:
            all_organoid_props.to_hdf(temp_output_path, key='organoid')


class MultiAggregateFeaturesTask(luigi.WrapperTask):

    movie_dirs = luigi.ListParameter(
        description='List of movie directories or glob patterns')

    def _expand_movie_dirs(self):
        movie_dirs = []
        for movie_dir in self.movie_dirs:
            movie_dirs.extend(glob(movie_dir))

        return sorted(movie_dirs)

    def requires(self):
        for movie_dir in self._expand_movie_dirs():
            yield AggregateFeaturesTask(movie_dir=movie_dir)


class MultiAggregateOrganoidFeaturesTask(luigi.WrapperTask):

    movie_dirs = luigi.ListParameter(
        description='List of movie directories or glob patterns')

    def _expand_movie_dirs(self):
        movie_dirs = []
        for movie_dir in self.movie_dirs:
            movie_dirs.extend(glob(movie_dir))

        return sorted(movie_dirs)

    def requires(self):
        for movie_dir in self._expand_movie_dirs():
            yield AggregateOrganoidFeaturesTask(movie_dir=movie_dir)
