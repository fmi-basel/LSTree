import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd


class LSCoordinates():
    '''class implementing conversion between lightsheet related coordinates systems
    
    mamut_c: (x,y,z) isotropic, float, unit=um
    img_c: (z,y,x) anisotropic, int, unit=px'''

    # NOTE internal implementation with mamut/physical coordinates to not loose floating point info

    def __init__(self, mamut_c, voxel_spacing):
        '''
        args:
            mamut_c: (x,y,z) um coordinates
            voxel_spacing: (zs, ys, xs) voxel size
        '''

        self.mamut = mamut_c
        self.spacing = voxel_spacing

    def __str__(self):
        return str(self.mamut)

    def __repr__(self):
        return repr(self.mamut)

    @property
    def img(self):

        coords = (self.mamut[2] / self.spacing[0],
                  self.mamut[1] / self.spacing[1],
                  self.mamut[0] / self.spacing[2])

        return tuple(int(round(c)) for c in coords)

    @classmethod
    def from_img(cls, img_c, voxel_spacing):

        coords = (img_c[2] * voxel_spacing[2], img_c[1] * voxel_spacing[1],
                  img_c[0] * voxel_spacing[0])

        return cls(coords, voxel_spacing)


def set_time_attribute(tree, time_interval):
    '''Set a new "time" attribute = "mamut_t" x time_interval'''

    for n, attr in tree.nodes:
        attr['time'] = attr['mamut_t'] * time_interval


def label_split_cells(tree):
    '''Labels splitting cells as 1 before and as 2 after split'''

    tree.set_all_nodes_attribute('split', 0)

    for n_id, n_attr in tree.nodes:
        successors = list(tree.successors(n_id))
        if len(successors) > 1:
            n_attr['split'] = 1
            for nid in successors:
                tree.nodes[nid]['split'] = 2


def compute_split_orientation(tree):
    '''compute abs(cos(alpha)) where alpha is the angle between segment 
    linking children and segment linking their midpoint and the 
    organoid's center of mass'''
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    label_split_cells(tree)
    for parent_id, _ in tree.filter('split', lambda x: x == 1).nodes:
        # get center point of both children
        c1_id, c2_id = list(tree.successors(parent_id))[0:2]
        c1 = np.asarray(tree.nodes[c1_id]['spatial_coord'])
        c2 = np.asarray(tree.nodes[c2_id]['spatial_coord'])
        cmid = (c1 + c2) / 2

        # get center of mass at current timepoint (base on seeds only)
        t = tree.nodes[c1_id]['mamut_t']
        nt_pos = tree.filter(
            'mamut_t',
            lambda x: x == t).get_all_nodes_attribute('spatial_coord')
        cm = np.asarray(list(nt_pos.values())).mean(axis=0)

        # split angle definition only makes sense if there is more than 2 cells
        if len(nt_pos) > 2:
            dp = np.abs(np.dot(unit_vector(c1 - c2), unit_vector(cmid - cm)))

            tree.nodes[parent_id]['split_orient'] = np.arccos(dp) * 180 / np.pi


def label_merge_cells(tree):
    for n_id, n_attr in tree.nodes:
        n_attr['merge'] = len(list(tree.predecessors(n_id))) > 1


def repack_spatial_coordinates(tree,
                               x_key='mamut_x',
                               y_key='mamut_y',
                               z_key='mamut_z'):
    for n_id, n_attr in tree.nodes:
        n_attr['spatial_coord'] = tuple(n_attr[key]
                                        for key in [x_key, y_key, z_key])


def calculate_displacement(tree):
    if len(tree.get_all_nodes_attribute('spatial_coord')) != len(tree.nodes):
        repack_spatial_coordinates(tree)

    for es, ee, e_attr in tree.edges:
        source = np.asarray(tree.nodes[es]['spatial_coord'])
        target = np.asarray(tree.nodes[ee]['spatial_coord'])
        e_attr['displacement'] = np.linalg.norm(source - target)


def label_generation(tree):
    '''Adds a 'generation' attribute to each node and a unique id for each branch within a generation'''

    label_split_cells(tree)

    tree.set_all_nodes_attribute('generation', -1)
    tree.set_all_nodes_attribute('branch_id', -1)

    current_gen_id = np.zeros(32, dtype=int)  # assumes at most 32 generations

    def increment_gen_ids():
        current_gen_id[current_gen] += 1
        # set subsequent gen ids to their minimum possible value
        for i in range(1, 32 - current_gen):
            current_gen_id[current_gen +
                           i] = 2 * current_gen_id[current_gen + i - 1]

    def label_from_root(root):
        nonlocal current_gen, previous_t
        tree.nodes[root]['generation'] = current_gen
        tree.nodes[root]['branch_id'] = current_gen_id[current_gen]

        for e in tree.depth_first_edges(root):
            current_t = tree.nodes[e[1]][tree.time_key]

            if current_t <= previous_t:
                # end of a track was reached (jump back in time)
                # or track ended immediatly after split (stay on same timepoint)
                increment_gen_ids()
                current_gen = tree.nodes[
                    e[0]]['generation'] + 1  # restart after gen of parent node
            elif tree.nodes[e[0]]['split'] == 1:
                # split was reached
                current_gen_id[current_gen] += 1
                current_gen += 1

            tree.nodes[e[1]]['generation'] = current_gen
            tree.nodes[e[1]]['branch_id'] = current_gen_id[current_gen]

            previous_t = current_t

    # iterate edges, depth first starting from each root of disjoint trees
    for root in tree.roots:
        current_gen = 0
        previous_t = -1

        label_from_root(root)
        increment_gen_ids()


def compute_symmetric_dendrogram_coordinates(tree):

    label_generation(tree)

    px = tree.get_all_nodes_attribute(tree.time_key)

    gen = tree.get_all_nodes_attribute('generation')
    branch_id = tree.get_all_nodes_attribute('branch_id')
    max_gen = max(gen.values())
    py = {
        key: (0.5 + branch_id[key]) * (2**(max_gen - gen[key]))
        for key in gen.keys()
    }

    plot_pos = {key: (px[key], py[key]) for key in px.keys()}
    tree.set_all_nodes_attribute('dendogram_coord', plot_pos)
    tree.set_all_nodes_attribute('dendogram_symmetric', py)


def compute_uniform_dendrogram_coordinates(tree):
    '''Uniformly distribute terminuses'''

    label_split_cells(tree)
    label_merge_cells(tree)

    tree.set_all_nodes_attribute('dendogram_uniform', -1.)

    # find and set y coord of terminuses
    terminuses = []
    terminus_counts = 0

    for root in tree.roots:
        for e in tree.depth_first_edges(root):
            if len(list(tree.successors(e[1]))) == 0:
                terminuses.append(e[1])
                tree.nodes[e[1]]['dendogram_uniform'] = terminus_counts
                terminus_counts += 1

            # if e[1] is a merge add parents except first
            if tree.nodes[e[1]]['merge']:
                for a in list(tree.predecessors(e[1]))[1:]:
                    terminuses.append(a)
                    tree.nodes[a]['dendogram_uniform'] = terminus_counts
                    terminus_counts += 1

    for terminus in terminuses:
        for n in tree.predecessor_track_nodes(terminus):
            if tree.nodes[n]['dendogram_uniform'] > 0:
                # already visited and set coord
                # skip this node but continue to find possible merged branches
                continue

            successors_coord = np.array([
                tree.nodes[s_id]['dendogram_uniform']
                for s_id in tree.successors(n)
            ])

            if np.all(successors_coord >= 0):
                tree.nodes[n]['dendogram_uniform'] = successors_coord.mean()
            else:
                # stop here and restart from next teminus
                # we must visit side branches before going deeper
                break


def compute_dendrogram_coordinates(tree):

    compute_symmetric_dendrogram_coordinates(tree)
    compute_uniform_dendrogram_coordinates(tree)


def traceback_cell(tree,
                   node_id,
                   label=None,
                   attr_name='backtrace',
                   reset=False):
    '''Adds/modifies a node attribute labeling ancestry trace​.
    
    Args:
        tree: lineage tree.
        node_id: id of the node to backtrace.
        label: label to assign to ancestry nodes. Defaults to node_id if not specified. 
        attr_name: name of the attributes.
        reset: If True all nodes' label are reset to zero, else only the current ancestry trace is overwritten.
    '''

    if reset or len(tree.get_all_nodes_attribute(attr_name)) == 0:
        tree.set_all_nodes_attribute(attr_name, 0)

    if label is None:
        label = node_id

    def label_ancestors(node_id):
        tree.nodes[node_id][attr_name] = label

        for p_id in tree.predecessors(node_id):
            label_ancestors(p_id)

    label_ancestors(node_id)


def label_cg(tree, generation):
    '''propagates the labels of the n-cell generation until the end.'''

    label_generation(tree)

    attr_name = 'gen{}_label'.format(generation)
    tree.set_all_nodes_attribute(attr_name, 0)

    gentree = tree.filter('generation', lambda x: x == generation)
    first_timepoint = min(
        [v for v in gentree.get_all_nodes_attribute(tree.time_key).values()])

    label = 1
    for root in gentree.roots:
        tree.nodes[root][attr_name] = label
        for n in tree.successor_track_nodes(root):
            tree.nodes[n][attr_name] = label
        label += 1


def compute_timepoint_ids(tree):
    '''Adds sequential ids per timepoint'''

    max_t = int(max(tree.get_all_nodes_attribute('mamut_t').values()))
    counts = np.zeros((max_t + 1, ), dtype=int)

    for n, n_attr in tree.nodes:
        n_attr['timepoint_id'] = counts[int(n_attr['mamut_t'])]
        counts[int(n_attr['mamut_t'])] += 1


def label_parent_id(tree):
    for n, n_attr in tree.nodes:
        parents = list(tree.predecessors(n))
        if len(parents) > 0:
            n_attr['mamut_parent_id'] = parents[0]
        else:
            n_attr['mamut_parent_id'] = 0


def calculate_time_since_split(tree):
    def _add_time_since_split(subdf):
        subdf['time_since_split'] = subdf.mamut_t - subdf.mamut_t.min()
        return subdf

    df = pd.DataFrame([{'node_id': n[0], **n[1]} for n in tree.nodes])
    df = df.groupby(['generation', 'branch_id']).apply(_add_time_since_split)
    for _, row in df.iterrows():
        tree.set_node_attribute(row.mamut_id, 'time_since_split',
                                row.time_since_split)


def calculate_image_coords(tree, spacing):
    for n, attr in tree.nodes:
        coord = LSCoordinates(
            (attr['mamut_x'], attr['mamut_y'], attr['mamut_z']),
            voxel_spacing=spacing)
        img_c = coord.img
        attr['z'] = img_c[0]
        attr['y'] = img_c[1]
        attr['x'] = img_c[2]


def label_merged_branch(tree):
    '''Gives the same branch id to nodes in merging branches, other are set to -1'''

    label_generation(tree)
    label_merge_cells(tree)

    # build merge branch lut: (generation,branch_id) --> merge_branch_id
    branch_lut = {}

    for nid, attr in tree.nodes:
        if attr['merge']:
            merged_branches = []

            for pid in tree.predecessors(nid):
                merged_branches.append(tree.nodes[pid]['branch_id'])

            for merged_branch in merged_branches:
                branch_lut[(attr['generation'],
                            merged_branch)] = min(merged_branches)

    # set merged branch id
    for nid, attr in tree.nodes:
        attr['merged_branch_id'] = branch_lut.get(
            (attr['generation'], attr['branch_id']), -1)


def label_merged_tracks(tree):
    '''sets a "merged_track" attributes=1 to tracks following a merge'''

    label_merge_cells(tree)
    tree.set_all_nodes_attribute('merged_track', 0)

    for nid, attr in tree.nodes:
        if attr['merge'] > 0:
            for successor_nid in tree.successor_track_nodes(nid):
                tree.nodes[successor_nid]['merged_track'] = 1


def label_cell_id(tree):
    '''Adds cell_id attribute equal to the node (nuclei) id except
    when they are multiple nuclei in a single cell (merged tracks)'''

    label_merged_branch(tree)

    # build cell lut: (timepoint, generation, branch_id) --> cell_id
    cell_lut = {}

    for nid, attr in tree.nodes:

        # if merged branch cell_lut element will be reassigned --> last label is kept (i.e. no guarentee of being min label etc.)
        branch_id = attr['branch_id'] if attr[
            'merged_branch_id'] < 0 else attr['merged_branch_id']
        cell_lut[(attr['mamut_t'], attr['generation'], branch_id)] = nid

    # apply cell lut
    for nid, attr in tree.nodes:
        branch_id = attr['branch_id'] if attr[
            'merged_branch_id'] < 0 else attr['merged_branch_id']
        attr['cell_id'] = cell_lut[(int(attr[tree.time_key]),
                                    attr['generation'], branch_id)]


def add_coordinate_attribute(tree, coords_key, x_key, y_key):
    '''repacks node attributes into 2d coordinates for plotting'''

    px = tree.get_all_nodes_attribute(x_key)
    py = tree.get_all_nodes_attribute(y_key)

    plot_pos = {key: (px[key], py[key]) for key in px.keys()}
    tree.set_all_nodes_attribute(coords_key, plot_pos)


def map_ref_tree(tree, ref_tree):
    '''maps 'spatial_coord' from predicted tree to groundtruth'''

    ref_tree.set_all_nodes_attribute('matched', 0)
    last_timepoint = int(
        max([
            v for v in ref_tree.get_all_nodes_attribute(
                ref_tree.time_key).values()
        ]))

    for t in range(1, last_timepoint + 1):
        ref_subtree = ref_tree.filter(ref_tree.time_key, lambda x: t == x)
        subtree = tree.filter(tree.time_key, lambda x: t == x)

        ref_points = [att['spatial_coord'] for _, att in ref_subtree.nodes]
        points = [att['spatial_coord'] for _, att in subtree.nodes]

        # get nearest in groundtruth for each candidates
        if points and ref_points:
            pw_dist = cdist(ref_points, points, 'euclidean')
            nearest_d = pw_dist.min(axis=0)
            nearest_ids = np.argmin(pw_dist, axis=0)
            lut = np.array([id for id, _ in ref_subtree.nodes])

            for dist, idx in zip(nearest_d, lut[nearest_ids]):
                if dist < 30:
                    ref_tree.nodes[idx]['matched'] = 1
