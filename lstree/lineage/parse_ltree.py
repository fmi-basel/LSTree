from glob import glob
import os
import pandas as pd
import numpy as np

from xml.etree import ElementTree as ET

from lstree.lineage.ltree import LineageTree
from lstree.lineage.utils import repack_spatial_coordinates

NODE_KEYS_MAPPING = {
    'ID': ('mamut_id', int),
    'POSITION_X': ('mamut_x', float),
    'POSITION_Y': ('mamut_y', float),
    'POSITION_Z': ('mamut_z', float),
    'POSITION_T': ('mamut_t', float),
}
EDGE_KEYS = ['SPOT_SOURCE_ID', 'SPOT_TARGET_ID']


def parse_nodes(xml_root):
    '''
    '''
    key = 'SpotsInFrame'
    for frame in xml_root.iter(key):
        for spot in frame.findall('Spot'):
            # offset to match tif filename index starting at one
            spot.attrib['POSITION_T'] = float(spot.attrib['POSITION_T']) + 1
            yield (int(spot.attrib['ID']), {
                val[0]: val[1](spot.attrib[key])
                for key, val in NODE_KEYS_MAPPING.items()
            })


def parse_edges(xml_root):
    '''
    '''
    key = 'Track'
    for track in xml_root.iter(key):
        for edge in track.findall('Edge'):
            yield (int(edge.attrib[EDGE_KEYS[0]]),
                   int(edge.attrib[EDGE_KEYS[1]]))


def parse_region_props(tree, segmentation_csv_dir, props_list=None, **kwargs):
    '''adds precomputed regionprops to the graph'''

    # Note:
    # itertuple dow not support . in list
    # iterrows does not preserve dtype

    for p in sorted(glob(os.path.join(segmentation_csv_dir, '*.csv'))):
        try:
            seg_df = pd.read_csv(p, **kwargs)
            seg_df.columns = [c.replace('.', '_') for c in seg_df.columns]
            for nucleus in seg_df.itertuples():

                # TODO remove props_list option and parse everything available
                if props_list is None:
                    _props_list = seg_df.columns
                else:
                    _props_list = props_list

                # ~for prop in props_list:
                for prop in _props_list:
                    try:
                        tree.set_node_attribute(
                            nucleus.mamut_id, prop,
                            getattr(nucleus, prop.replace('.', '_')))
                    except Exception as e:
                        # ~print(e)
                        pass
        except Exception as e:
            pass


def parse_nuclei_props(tree, segmentation_csv_dir, **kwargs):
    '''adds precomputed nuclei props to the graph'''

    for p in sorted(glob(os.path.join(segmentation_csv_dir, '*.csv'))):
        props = pd.read_csv(p, **kwargs)
        for row in props.itertuples():
            for name, val in row._asdict().items():
                if name in ['Index', 'node_id', 'mamut_t', 'mamut_id']:
                    continue

                try:
                    tree.set_node_attribute(row.mamut_id, name, val)
                except Exception as e:
                    # print(e)
                    pass


def parse_cell_props(tree, segmentation_csv_dir, **kwargs):
    '''adds precomputed cell props to the graph'''

    cell_to_nuclei_lut = {}
    for n, attr in tree.nodes:
        cell_id = attr.get('cell_id', None)
        if cell_id is not None:
            cell_to_nuclei_lut[cell_id] = cell_to_nuclei_lut.get(cell_id,
                                                                 []) + [n]

    for p in sorted(glob(os.path.join(segmentation_csv_dir, '*.csv'))):
        props = pd.read_csv(p, **kwargs)
        for row in props.itertuples():
            for name, val in row._asdict().items():
                try:
                    for node_id in cell_to_nuclei_lut[row.cell_id]:
                        tree.set_node_attribute(node_id, name, val)
                except Exception as e:
                    # print(e)
                    pass


def construct_tree(path):
    '''
    '''
    xmltree = ET.parse(path)

    tree = LineageTree(nodes=parse_nodes(xmltree.getroot()),
                       edges=parse_edges(xmltree.getroot()),
                       time_key='mamut_t')

    return tree


def write_tree(tree, path, node_attr=[]):
    '''Export tree as vtk file (paraview compatible)'''

    if len(tree.get_all_nodes_attribute('spatial_coord')) != len(tree.nodes):
        repack_spatial_coordinates(tree)

    node_list = np.asarray([n for n, _ in tree.nodes])

    def get_attribute(attr):
        nonlocal node_list
        return np.asarray([tree.nodes[n].get(attr, 0) for n in node_list],
                          dtype=float)

    node_lut = np.zeros(node_list.max() + 1, dtype=int)
    for idx, n in enumerate(node_list):
        node_lut[n] = idx

    polydata = tvtk.PolyData(
        points=get_attribute('spatial_coord'),
        #         lines=np.asarray([(node_lut[n],node_lut[n]) for  n in node_list])
        lines=np.asarray([(node_lut[es], node_lut[ee])
                          for es, ee, d in tree.edges]),
    )

    polydata.point_data.scalars = get_attribute('mamut_t')
    polydata.point_data.get_array(0).name = 'timepoint'

    for idx, attr in enumerate(node_attr):
        polydata.point_data.add_array(get_attribute(attr))
        polydata.point_data.get_array(idx + 1).name = attr

    write_data(polydata, path)


# xml export
def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


xml_template = r'''<TrackMate version="3.6.0">
                      <Model spatialunits="µm" timeunits="frame">
                        <FeatureDeclarations>
                          <SpotFeatures />
                          <EdgeFeatures />
                          <TrackFeatures />
                        </FeatureDeclarations>
                        <AllSpots nspots="" />
                        <AllTracks>
                            <Track name="1" TRACK_ID="1" />
                        </AllTracks>
                        <FilteredTracks>
                          <TrackID TRACK_ID="1" />
                        </FilteredTracks>
                      </Model>
                      <Settings>
                        <ImageData filename="" folder="" height="" nframes="" nslices="" pixelheight="" pixelwidth="" voxeldepth="" width="" />
                        <InitialSpotFilter feature="QUALITY" isabove="true" value="0.0" />
                        <SpotFilterCollection />
                        <TrackFilterCollection />
                        <AnalyzerCollection>
                          <SpotAnalyzers />
                          <EdgeAnalyzers>
                            <Analyzer key="Edge target" />
                          </EdgeAnalyzers>
                          <TrackAnalyzers />
                        </AnalyzerCollection>
                      </Settings>
                    </TrackMate>'''


def write_tree_mamut(tree,
                     output_path,
                     bdv_xml_path,
                     sampling=1,
                     coord_keys=['mamut_x', 'mamut_y', 'mamut_z']):
    '''Replaces nodes and edges in mamut/mastodon xml by the ones from the tree given in argument'''
    def add_edges(xml_track, tree):
        for e1, e2, _ in tree.edges:
            ET.SubElement(xml_track, 'Edge', {
                'SPOT_SOURCE_ID': str(e1),
                'SPOT_TARGET_ID': str(e2)
            })

    def add_nodes(xml_allspots, tree, n_frames, coord_keys):
        for t in range(1, n_frames + 1):
            frame = ET.SubElement(xml_allspots, 'SpotsInFrame',
                                  {'frame': str(t - 1)})
            subtree = tree.filter(tree.time_key, lambda x: t == x)

            for n_id, n_attr in subtree.nodes:
                ET.SubElement(
                    frame, 'Spot', {
                        'ID': str(n_id),
                        'name': str(n_id),
                        'POSITION_X': str(n_attr[coord_keys[0]]),
                        'POSITION_Y': str(n_attr[coord_keys[1]]),
                        'POSITION_Z': str(n_attr[coord_keys[2]]),
                        'FRAME': str(t - 1),
                        'POSITION_T': '{:.1f}'.format(t - 1),
                        'QUALITY': '-1.0',
                        'VISIBILITY': '1',
                        'RADIUS': '3'
                    })

    root = ET.fromstring(xml_template)

    allspots = next(root.iter('AllSpots'))
    allspots.attrib['nspots'] = str(len(tree.nodes))

    track = next(root.iter('Track'))

    # write imagedata info
    imagedata = next(root.iter('ImageData'))
    imagedata.attrib['filename'] = os.path.basename(bdv_xml_path)
    imagedata.attrib['folder'] = os.path.dirname(bdv_xml_path)
    imagedata.attrib['width'], imagedata.attrib['height'], imagedata.attrib[
        'nslices'] = next(ET.parse(bdv_xml_path).iter('size')).text.split(' ')
    imagedata.attrib['voxeldepth'], imagedata.attrib[
        'pixelheight'], imagedata.attrib['pixelwidth'] = [
            str(s) for s in np.broadcast_to(np.asarray(sampling), 3)
        ]

    n_frames = max([
        int(r.attrib['timepoint'])
        for r in ET.parse(bdv_xml_path).iter('ViewRegistration')
    ]) + 1
    imagedata.attrib['nframes'] = str(n_frames)

    add_nodes(allspots, tree, n_frames=n_frames, coord_keys=coord_keys)
    add_edges(track, tree)

    indent(root)
    #     ET.dump(root)

    xmltree = ET.ElementTree(root)
    xmltree.write(output_path, xml_declaration=True, encoding='UTF-8')
