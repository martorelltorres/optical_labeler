"""Training windows: window centres, cropping with scale, and geometric augmentation.

Arrays are cropped jointly (image, validity, codes) so that they stay aligned.
"""
import numpy as np
from scipy.ndimage import zoom


def crop(arr, r, c, size, fill=0):
    """size x size window centred on (r, c), padded with `fill` beyond the raster."""
    h, w = arr.shape[:2]
    r0, c0 = r - size // 2, c - size // 2
    out = np.full((size, size) + arr.shape[2:], fill, arr.dtype)
    rs, cs = max(r0, 0), max(c0, 0)
    re, ce = min(r0 + size, h), min(c0 + size, w)
    if rs < re and cs < ce:
        out[rs - r0:re - r0, cs - c0:ce - c0] = arr[rs:re, cs:ce]
    return out


class CenterSampler:
    """Random window centres inside a mask, a fraction of them on positive pixels."""

    def __init__(self, mask, positive, pos_fraction=0.5):
        self.all = np.flatnonzero(mask)
        self.pos = np.flatnonzero(mask & positive)
        if self.all.size == 0:
            raise ValueError("empty sampling mask")
        self.pos_fraction = pos_fraction if self.pos.size else 0.0
        self.shape = mask.shape

    def __call__(self, rng):
        pool = self.pos if rng.random() < self.pos_fraction else self.all
        return np.unravel_index(pool[rng.integers(pool.size)], self.shape)


def scaled_crop(arrays, center, size, scale, fills):
    """Crop round(size / scale) around center from each array and resize to size.

    The first array is resampled linearly, the rest by nearest neighbour. Trailing axes
    beyond the first two (e.g. colour channels) are not resampled.
    """
    src = max(int(round(size / scale)), 1)
    out = []
    for i, (a, f) in enumerate(zip(arrays, fills)):
        w = crop(a, center[0], center[1], src, f)
        if src != size:
            is_bool = w.dtype == bool
            zf = (size / src, size / src) + (1,) * (w.ndim - 2)
            w = zoom(w.astype(np.float32) if is_bool else w, zf,
                     order=1 if i == 0 else 0, mode="nearest", grid_mode=True)[:size, :size]
            w = w > 0.5 if is_bool else w.astype(a.dtype)
        out.append(w)
    return out


def geometric(arrays, rng):
    k = int(rng.integers(4))
    flip = rng.random() < 0.5
    out = []
    for a in arrays:
        a = np.rot90(a, k)
        if flip:
            a = a[:, ::-1]
        out.append(np.ascontiguousarray(a))
    return out
