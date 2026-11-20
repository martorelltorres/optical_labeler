"""Reviewed label polygons (GeoJSON features) to class masks on a raster grid.

Each polygon carries a `class` property: `posidonia`, `rock` or `ignore` (objects that must count
neither as Posidonia nor as background, e.g. man-made targets). Where polygons of different
classes overlap, ignore wins over rock and rock over Posidonia, so the masks are disjoint.

An optional annotated area (polygons without a class) marks what was reviewed: every mask is
clipped to it, and its own mask tells a consumer which pixels count as background.
"""
import numpy as np
from rasterio.features import rasterize

CLASSES = ("posidonia", "rock", "ignore")


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
    masks["rock"] &= ~masks["ignore"]
    masks["posidonia"] &= ~(masks["rock"] | masks["ignore"])
    area = None
    if area_features is not None:
        area = _mask([f["geometry"] for f in area_features], shape, transform)
        for c in CLASSES:
            masks[c] &= area
    return masks, area
