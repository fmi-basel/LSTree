from lstree.lineage.utils import LSCoordinates

import pytest
import numpy as np

spacing = (2, 0.26, 0.26)

c_img = (1, 2, 3)
c_mamut = (0.78, 0.52, 2)


def test_padded_crop():

    assert LSCoordinates.from_img(
        c_img, voxel_spacing=spacing).mamut == pytest.approx(c_mamut)

    assert LSCoordinates(c_mamut,
                         voxel_spacing=spacing).mamut == pytest.approx(c_mamut)
    assert LSCoordinates(c_mamut,
                         voxel_spacing=spacing).img == pytest.approx(c_img)
