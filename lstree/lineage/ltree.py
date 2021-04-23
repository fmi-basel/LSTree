import copy
import pandas as pd
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_successors, dfs_edges


class LineageTree:
    def __init__(self, nodes, edges, time_key):
        ''''''
        self.g = nx.DiGraph()
        self.g.add_nodes_from(nodes)
        self.g.add_edges_from(edges)
        self.time_key = time_key

        self.correct_edge_direction()

    @classmethod
    def from_dataframe(cls, dfn, dfe, time_key):
        n = list(zip(dfn.mamut_id, dfn.to_dict('records')))
        e = dfe.set_index(['node_start', 'node_stop'])
        e = list(
            zip(e.index.get_level_values(0), e.index.get_level_values(1),
                e.to_dict('records')))

        return cls(n, e, time_key)

    def correct_edge_direction(self):
        '''Forces forward edges in time'''
        edges_to_reverse = []
        for e in self.g.edges():
            if self.g.nodes[e[0]][self.time_key] > self.g.nodes[e[1]][
                    self.time_key]:
                edges_to_reverse.append(e)

        for e in edges_to_reverse:
            self.g.remove_edge(*e)
            self.g.add_edge(*e[::-1])

    def copy(self):
        tree = LineageTree([], [], self.time_key)
        tree.g = self.g.copy()
        return tree

    @property
    def nodes(self):
        return self.g.nodes(data=True)

    @property
    def edges(self):
        return self.g.edges(data=True)

    def get_edge(self, start, end):
        return self.g.edges[(start, end)]

    def add_edge(self, start, end, attributes={}):
        self.g.add_edge(start, end, **attributes)

    def add_edges(self, edges):
        self.g.add_edges_from(edges)

    def clear_edges(self):
        self.g.remove_edges_from(list(self.g.edges()))

    def delete_nodes(self, ids):
        self.g.remove_nodes_from(ids)

    def successors(self, node_id):
        return self.g.successors(node_id)

    def predecessors(self, node_id):
        return self.g.predecessors(node_id)

    def depth_first_edges(self, root):
        return dfs_edges(self.g, source=root)

    def successor_track_nodes(self, root):
        # flatten list of list
        return [y for x in dfs_successors(self.g, root).values() for y in x]

    def predecessor_track_nodes(self, root):
        # note: dfs_predecessors returns int instead of list --> not equivalent to dfs_successors
        # flatten list of list
        return [
            y
            for x in dfs_successors(self.g.reverse(copy=False), root).values()
            for y in x
        ]

    def set_node_attribute(self, node_id, key, val):
        self.g.nodes[node_id][key] = val

    def get_node_attribute(self, node_id, key):
        return self.g.nodes[node_id][key]

    def set_all_nodes_attribute(self, key, vals):
        nx.set_node_attributes(self.g, vals, name=key)

    def get_all_nodes_attribute(self, key):
        return nx.get_node_attributes(self.g, key)

    def subtree(self, sub_node_list):
        '''returns a view on filtered subtree'''

        st = LineageTree([], [], self.time_key)
        st.g = self.g.subgraph(sub_node_list)
        return st

    def filter(self, key, filter_fn):
        '''Returns a view on subtree with node attribute by filter_fn on key'''

        # TODO: find if key corresponds to nodes or edges (--> edge_subgraph)
        return self.subtree([
            n for n, attr in self.g.nodes(data=True)
            if filter_fn(attr.get(key))
        ])

    def get_all_edges_attribute(self, key):
        return nx.get_edge_attributes(self.g, name=key)

    @property
    def roots(self):
        '''roots  id of disjoint trees'''
        return [n for n, d in self.g.in_degree() if d == 0]
