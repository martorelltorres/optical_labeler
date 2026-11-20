"""Reviewed label polygons (GeoJSON features) to class masks on a raster grid.

Each polygon carries a `class` property: `posidonia`, `dead_posidonia` (dead Posidonia matte),
`rock`, `sand`, `other` (any other bottom) or `ignore` (objects to leave out, e.g. man-made
targets). Where polygons of different classes overlap the masks are made disjoint with the
priority ignore > rock > dead_posidonia > sand > other > posidonia, so patches of other bottom can
be drawn on top of a Posidonia polygon without cutting it.

An optional annotated area (polygons without a class) marks what was reviewed: every mask is
clipped to it, and its own mask tells a consumer which pixels count as background.
"""
import numpy as np
from rasterio.features import rasterize

CLASSES = ("posidonia", "dead_posidonia", "rock", "sand", "other", "ignore")
PRIORITY = ("posidonia", "other", "sand", "dead_posidonia", "rock", "ignore")   # later wins


def _mask(geoms, shape, transform):
    geoms = [g for g in geoms if g]
    if not geoms:
        return np.zeros(shape, bool)
    return rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform,
                     dtype="uint8").astype(bool)


def class_masks(features, shape, transform, area_features=None):
    """Disjoint boolean masks {class: (H, W)} and the area mask (None without area_features).

    Raises ValueError for a polygon whose class is missing or not in CLASSES, so that a typo
    made while reviewing is not silently dropped.
    """
    bad = sorted({str(f["properties"].get("class")) for f in features
                  if f["properties"].get("class") not in CLASSES})
    if bad:
        raise ValueError(f"unknown polygon classes {bad}; expected one of {CLASSES}")
    masks = {c: _mask([f["geometry"] for f in features if f["properties"]["class"] == c],
                      shape, transform) for c in CLASSES}
    for i, low in enumerate(PRIORITY):
        for high in PRIORITY[i + 1:]:
            masks[low] &= ~masks[high]
    area = None
    if area_features is not None:
        area = _mask([f["geometry"] for f in area_features], shape, transform)
        for c in CLASSES:
            masks[c] &= area
    return masks, area
