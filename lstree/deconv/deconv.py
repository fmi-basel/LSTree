import numpy as np

from dlutils.prediction.stitching import StitchingGenerator
from flowdec import restoration as fd_restoration
from flowdec import data as fd_data

algo = fd_restoration.RichardsonLucyDeconvolver(
    n_dims=3, pad_min=[0, 0,
                       0], pad_mode='none', start_mode='INPUT').initialize()


def deconvolve(img, psf, niter=128):

    acquisition = fd_data.Acquisition(data=img, kernel=psf)
    return algo.run(acquisition, niter=niter).data


# TODO avoid tensorflow reinit for each image/tile? (probably would have to rewrite  instead of using flowdec)
# TODO auto optimize patch size to minimize the number of tile
def tiled_deconv(raw,
                 psf,
                 niter=128,
                 max_patch_size=(512, 512, 512),
                 pad_size=(16, 16, 16)):

    patch_size = tuple(min(p, s) for p, s in zip(max_patch_size, raw.shape))

    # offset to allow negative values
    raw = raw.astype(np.float32) + 10000

    if patch_size == raw.shape:
        deconv = deconvolve(raw, psf)

    else:
        pad_width = [(b, b) for b in pad_size]
        raw = np.pad(raw, pad_width=pad_width, mode='symmetric')

        deconv = np.zeros_like(raw, dtype=np.float32)
        border_loc = tuple(slice(p1, -p2) for p1, p2 in pad_width)
        generator = StitchingGenerator(raw[..., None],
                                       batch_size=1,
                                       patch_size=patch_size,
                                       border=pad_size)

        for data in generator:
            tile = data['input']
            coord = data['coord']

            #ignore batch dim
            tile = tile.squeeze()
            coord = coord[0]

            tile_deconv = deconvolve(tile, psf)
            loc = tuple(
                slice(x + b, x + dx - b)
                for b, x, dx in zip(pad_size, coord, patch_size))
            deconv[loc] = tile_deconv[border_loc]

        deconv = deconv[border_loc]

    return (deconv - 10000)
