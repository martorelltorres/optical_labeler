"""Spatial partition of a site into folds made of grid blocks."""
import numpy as np
from scipy.ndimage import distance_transform_edt

NO_FOLD = 255


def block_folds(positive, negative, k=3, block_px=150, gap_px=0.0, unbalanced=None):
    """Assign square blocks of a regular grid to k folds, balancing both classes.

    unbalanced: optional boolean array of further pixels (e.g. an ignore band) that receive
    the fold of their block and obey the gap, but do not count in the balance. A block made
    only of such pixels takes the fold of its nearest assigned pixel.

    Blocks are assigned largest first, each to the fold where it least increases the worst
    deviation of the positive and negative shares from 1/k. Pixels closer than gap_px / 2 to a pixel of another fold
    belong to no fold.

    Returns (folds, info): folds is uint8 with fold index 0..k-1 or NO_FOLD; info holds
    the worst deviation of a class share from 1/k and the shares per fold.
    """
    labeled = positive | negative
    h, w = labeled.shape
    br, bc = np.indices((h, w)) // block_px
    block_id = br * ((w + block_px - 1) // block_px) + bc
    n = block_id.max() + 1
    npos = np.bincount(block_id[positive], minlength=n) / positive.sum()
    nneg = np.bincount(block_id[negative], minlength=n) / negative.sum()
    shares = np.zeros((k, 2))
    assign = np.full(n, NO_FOLD, np.int64)
    for b in np.argsort(-(npos + nneg)):
        if npos[b] + nneg[b] == 0:
            continue
        cand = shares + np.array([npos[b], nneg[b]])
        f = int(np.argmin(np.abs(cand - 1.0 / k).max(1) - np.abs(shares - 1.0 / k).max(1)
                          + 1e-6 * cand.sum(1)))
        assign[b] = f
        shares[f] += (npos[b], nneg[b])
    folds = np.full((h, w), NO_FOLD, np.uint8)
    folds[labeled] = assign[block_id[labeled]].astype(np.uint8)
    if unbalanced is not None:
        extra = unbalanced & ~labeled
        folds[extra] = np.minimum(assign[block_id[extra]], NO_FOLD).astype(np.uint8)
        orphan = extra & (folds == NO_FOLD)
        if orphan.any():
            _, (ir, ic) = distance_transform_edt(folds == NO_FOLD, return_indices=True)
            folds[orphan] = folds[ir, ic][orphan]
    if gap_px > 0:
        near_other = np.zeros((h, w), bool)
        for f in range(k):
            dist = distance_transform_edt(folds != f)
            near_other |= (folds != f) & (folds != NO_FOLD) & (dist < gap_px / 2)
        folds[near_other] = NO_FOLD
    return folds, {"score": float(np.abs(shares - 1.0 / k).max()), "shares": shares.tolist()}
