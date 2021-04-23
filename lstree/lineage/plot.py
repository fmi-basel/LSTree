import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import dim
from bokeh.models import HoverTool


def tree_to_dataframe(tree):
    '''Returns DataFrames of nodes and edges.'''

    dfn = pd.DataFrame([{'mamut_id': n[0], **n[1]} for n in tree.nodes])
    dfn = dfn.select_dtypes([np.number, 'string'])
    dfn.set_index('mamut_id', drop=False, inplace=True)

    dfe = pd.DataFrame([{
        'node_start': n[0],
        'node_stop': n[1],
        **n[2]
    } for n in tree.subtree(dfn.index).edges])

    return dfn, dfe


def plot_tree(tree,
              dfe=None,
              coord_x='time',
              coord_y='dendogram_uniform',
              tooltips=['node_id'],
              node_size=5,
              node_color='dodgerblue',
              edge_width=1,
              edge_color='black',
              min_node_size=2,
              max_node_size=5,
              min_edge_width=1,
              max_edge_width=7,
              backend='bokeh'):
    '''
    Plots a LineageTree. alternatively dataframes of nodes (mamut_id as index) and edges can also be passed.
    '''

    if dfe is None:
        dfn, dfe = tree_to_dataframe(tree)
    else:
        dfn = tree

    dfe['x0'] = dfn.loc[dfe.node_start, coord_x].values
    dfe['x1'] = dfn.loc[dfe.node_stop, coord_x].values
    dfe['y0'] = dfn.loc[dfe.node_start, coord_y].values
    dfe['y1'] = dfn.loc[dfe.node_stop, coord_y].values

    nodes_vdims = tooltips
    if node_size in dfn.columns:
        nodes_vdims = nodes_vdims + [node_size]
        node_size = dim(node_size).norm() * (max_node_size -
                                             min_node_size) + min_node_size
    if node_color in dfn.columns:
        nodes_vdims = nodes_vdims + [node_color]

    hv_nodes = hv.Points(dfn,
                         kdims=[coord_x, coord_y],
                         vdims=nodes_vdims,
                         group='tree')

    edges_vdims = []
    if edge_width in dfe.columns:
        edges_vdims += [edge_width]
        edge_width = dim(edge_width).norm() * (max_edge_width -
                                               min_edge_width) + min_edge_width
    if edge_color in dfe.columns:
        edges_vdims += [edge_color]

    hv_edges = hv.Segments(dfe,
                           kdims=['x0', 'y0', 'x1', 'y1'],
                           vdims=edges_vdims,
                           group='tree')

    if backend == 'bokeh':
        hv_nodes.opts(color=node_color, size=node_size)
        hv_edges.opts(color=edge_color, line_width=edge_width)
        if tooltips is not None and backend == 'bokeh':
            tooltips = [(t.replace('_', ' '), '@' + t) for t in tooltips]
            hv_nodes.opts(tools=[HoverTool(tooltips=tooltips)])

    elif backend == 'matplotlib':
        hv_nodes.opts(color=node_color, s=node_size)
        hv_edges.opts(color=edge_color, linewidth=edge_width)
    else:
        raise ValueError(
            "backend not recognized, options are ['bokeh', 'matplotlib'], got: {}"
            .format(backend))

    hv_tree = (hv_edges * hv_nodes).relabel(group='tree').opts(xlabel=coord_x)

    return hv_tree
