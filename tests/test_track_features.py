import pytest
import numpy as np
import pandas as pd

from lstree.features.track_feature_tasks import extract_tracking_features


def test_extract_tracking_features():

    spacing = (1., 1., 1.)
    segm = np.zeros((10, 10, 10), np.uint16)
    segm_parent = np.zeros((10, 10, 10), np.uint16)

    # split nuclei perfectly match to parent
    segm[5, 0:2, 0:2] = 1
    segm[5, 2:4, 2:4] = 2

    segm_parent[5, 0:2, 0:2] = 1
    segm_parent[5, 2:4, 2:4] = 1

    # nuclei with varying overlap to parent

    # perfect overlap
    segm[0, 0:6, 0] = 3
    segm_parent[0, 0:6, 0] = 2

    # 2/3 overlap
    segm[0, 0:6, 1] = 4
    segm_parent[0, 0:4, 1] = 3

    # larger overalp with background
    segm[0, 0:6, 2] = 5
    segm_parent[0, 0:1, 2] = 4

    # only overlap background
    segm[0, 3:6, 3] = 6
    segm_parent[0, 0:2, 3] = 5

    link_scores = pd.DataFrame({
        'label_id': [1, 2],
        'linking_distance': [1., 5.]
    })

    props = extract_tracking_features(segm, segm_parent, link_scores, spacing)

    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None):
        print(props)

    bg_size = (segm_parent == 0).sum()
    np.testing.assert_equal(props.parent_label_id, np.array([1, 1, 2, 3, 0,
                                                             0]))
    np.testing.assert_almost_equal(props.overlap_to_parent,
                                   np.array([1., 1., 1., 2 / 3, 5 / 6, 1.]))
    np.testing.assert_almost_equal(
        props.link_iou,
        np.array([0.5, 0.5, 1., 2 / 3, 5 / (bg_size + 1), 3 / bg_size]))
