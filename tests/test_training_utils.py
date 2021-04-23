import pytest
import tensorflow as tf

from lstree.segmentation.training_utils import mix_datasets_with_reps


def test_mix_datasets_with_reps():

    # unbatched datasets
    d1 = tf.data.Dataset.from_tensor_slices(list('abc'))
    d2 = tf.data.Dataset.from_tensor_slices([str(i) for i in range(7)])
    expected_d = [c.encode() for c in 'a0b1c2a3b4c5a6b0c1a2']
    d = mix_datasets_with_reps(d1, d2)

    for val, target in zip(d.take(20).as_numpy_iterator(), expected_d):
        assert val == target

    # batched datasets
    d1 = tf.data.Dataset.from_tensor_slices([['a', 'b'], ['c', 'd']])
    d2 = tf.data.Dataset.from_tensor_slices([[str(i), str(i + 1)]
                                             for i in range(7)])
    expected_d = [[
        c[0].encode(), c[1].encode()
    ] for c in ['a0', 'b1', 'c1', 'd2', 'a2', 'b3', 'c3', 'd4', 'a4', 'b5']]
    d = mix_datasets_with_reps(d1, d2, batch_size=2)

    for val, target in zip(d.take(20).as_numpy_iterator(), expected_d):
        assert tuple(val) == tuple(target)
