import pytest
import numpy as np

from lstree.features.feature_tasks import NeighborFeatureExtractor


def test_neighbor_feature_extractor():
    feature_extractor = NeighborFeatureExtractor(spacing=(4., 1., 1.))

    labels = np.zeros((25, 100, 100), dtype=np.uint8)

    # rectangles aligned in x with increasing gap
    labels[2:7, 10:20, 10:20] = 1
    labels[2:7, 10:20, 20:30] = 2
    labels[2:7, 10:20, 32:40] = 3
    labels[2:7, 10:20, 45:50] = 4  # gap too large with label 3

    # rectangles aligned in z with increasing gap
    labels[9:11, 40:50, 45:50] = 5
    labels[11:20, 40:50, 45:50] = 6
    labels[21:23, 40:50, 45:50] = 7

    # rectangles arranged in 3x3 grid
    # 8  | 11 | 14
    # 9  | 12 | 15
    # 10 | 13 | 16
    labels[5:10, 60:70, 55:65] = 8
    labels[5:10, 70:80, 55:65] = 9
    labels[5:10, 80:90, 55:65] = 10
    labels[5:10, 60:70, 65:75] = 11
    labels[5:10, 70:80, 65:75] = 12
    labels[5:10, 80:90, 65:75] = 13
    labels[5:10, 60:70, 75:85] = 14
    labels[5:10, 70:80, 75:85] = 15
    labels[5:10, 80:90, 75:85] = 16

    props = feature_extractor({'cell': labels}, None)

    # yapf: disable
    expected_neighbor = [[2], #1
                         [1,3],
                         [2],
                         [],
                         [6], #5
                         [5],
                         [],
                         [9, 11, 12],
                         [8, 10 ,12, 11, 13],
                         [9, 13, 12], #10
                         [8, 12 ,14, 9 ,15],
                         [9, 11, 13, 15, 8, 14, 10, 16],
                         [10, 12, 16, 9, 15],
                         [11, 15, 12],
                         [12, 14, 16, 11, 13], #15
                         [13, 15, 12],]
    # yapf: enable

    for pred, expected in zip(props.feature_value, expected_neighbor):
        assert set(pred) == set(expected)
