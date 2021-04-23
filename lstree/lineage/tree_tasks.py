import luigi
import os
import json
import pandas as pd
import numpy as np

from lstree.luigi_utils import ExternalInputFile
from lstree.config import ExperimentParams

from lstree.lineage.parse_ltree import construct_tree
from lstree.lineage.utils import (
    set_time_attribute, compute_timepoint_ids, label_split_cells,
    label_merge_cells, calculate_displacement, compute_dendrogram_coordinates,
    label_generation, label_cg, compute_split_orientation, label_parent_id,
    calculate_time_since_split, calculate_image_coords, label_cell_id,
    label_merged_tracks)


class TreePropsTask(ExperimentParams, luigi.Task):
    '''Read xml tree and export features per timepoint (seeds, generation, etc.)
    
    '''

    movie_dir = luigi.Parameter(
        description='Base processing directory for a timelapse movie')
    out_subdir = luigi.Parameter(description='name of output subdirectory')
    xml_tree = luigi.Parameter(description='filename of xml tree')

    def requires(self):
        return {
            'experiment_config':
            ExternalInputFile(
                os.path.join(self.movie_dir, self.movie_config_name)),
            'xml_tree':
            ExternalInputFile(path=os.path.join(self.movie_dir, self.xml_tree))
        }

    def output(self):
        return luigi.LocalTarget(
            path=os.path.join(self.movie_dir, self.out_subdir))

    def run(self):

        with open(self.input()['experiment_config'].path, 'r') as f:
            experiment_config = json.load(f)

        with self.output().temporary_path() as self.temp_output_dir:
            os.makedirs(self.temp_output_dir, exist_ok=True)

            # parse mamut tree
            tree = construct_tree(self.input()['xml_tree'].path)

            # calculate all implemented features
            set_time_attribute(tree, experiment_config['time_interval'])
            label_merge_cells(tree)
            label_merged_tracks(tree)
            compute_timepoint_ids(tree)
            compute_dendrogram_coordinates(tree)
            label_split_cells(tree)
            label_generation(tree)
            label_cell_id(tree)

            calculate_displacement(tree)
            for es, ee, e_attr in tree.edges:
                # copy displacement attribute to end node
                tree.nodes[ee]['displacement'] = e_attr['displacement']

            max_generation = max(
                tree.get_all_nodes_attribute('generation').values())
            for gen in range(1, max_generation + 1):
                label_cg(tree, gen)

            tree.set_all_nodes_attribute('split_orient', 0)
            compute_split_orientation(tree)

            calculate_time_since_split(tree)
            label_parent_id(tree)
            calculate_image_coords(tree, experiment_config['spacing'])

            # convert tree to a dataframe
            df = pd.DataFrame([{'node_id': n[0], **n[1]} for n in tree.nodes])
            df.drop(['spatial_coord', 'dendogram_coord'], axis=1, inplace=True)
            df['merge'] = df['merge'].astype(np.uint8)
            df['mamut_t'] = df['mamut_t'].astype(int)

            # add timepoint cell id columns
            cell_to_nuclei_lut = np.zeros(df.cell_id.max() + 1,
                                          dtype=np.int) - 1
            cell_to_nuclei_lut[df.cell_id] = df.timepoint_id
            df['timepoint_cell_id'] = cell_to_nuclei_lut[df.cell_id.values]

            for mamut_t, subdf in df.groupby('mamut_t'):
                out_path = os.path.join(self.temp_output_dir,
                                        'T{:04d}.csv'.format(mamut_t))

                if not os.path.exists(out_path):
                    subdf.to_csv(out_path, index=False)
