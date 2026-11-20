"""Per-pixel class codes of the optical mosaics and the ignore band around Posidonia.

    0   background: annotated, but only known not to be any class annotated at that site
    1   Posidonia
    2   rock
    3   sand
    4   dead Posidonia (matte)
    5   other bottom (mud, gravel, algae, anything not in the classes above)
    254 ignore (band around Posidonia boundaries, objects to leave out)
    255 no label (outside the annotated area or without data)
Codes below IGNORE are labeled.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

BACKGROUND, POSIDONIA, ROCK, IGNORE, NODATA = 0, 1, 2, 254, 255
SAND, DEAD_POSIDONIA, OTHER = 3, 4, 5
CLASS_NAMES = {POSIDONIA: "posidonia", DEAD_POSIDONIA: "dead_posidonia", ROCK: "rock", SAND: "sand",
               OTHER: "other"}
NAME_CODES = {v: k for k, v in CLASS_NAMES.items()}


def labeled(codes):
    """Pixels with a class (any code below IGNORE)."""
    return codes < IGNORE


def boundary_band(posidonia, has_label, width_px):
    """Pixels closer than width_px to a Posidonia / not-Posidonia transition."""
    if width_px <= 0:
        return np.zeros_like(posidonia)
    pos = posidonia & has_label
    neg = ~posidonia & has_label
    near_pos = distance_transform_edt(~pos) <= width_px
    near_neg = distance_transform_edt(~neg) <= width_px
    return has_label & ((pos & near_neg) | (neg & near_pos))


def apply_band(labels, width_px):
    """Copy of labels with the band around Posidonia boundaries set to IGNORE.

    The band is computed from the Posidonia / not-Posidonia transitions among labeled
    pixels; pixels already IGNORE stay IGNORE.
    """
    out = labels.copy()
    out[boundary_band(labels == POSIDONIA, labeled(labels), width_px)] = IGNORE
    return out
