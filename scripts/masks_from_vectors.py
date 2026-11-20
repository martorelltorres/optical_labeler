#!/usr/bin/env python3
"""Rasterize reviewed Posidonia / rock / ignore polygons (GeoJSON from auto_label.py, edited in a
GIS) into masks in the format of the manual ones, on the grid of the optical mosaic reduced by
--factor. Writes <out>_<class>.png for posidonia, dead_posidonia, rock, sand, other and ignore
(255 = class), and <out>_area.png with --area. Overlaps resolve with the priority of
optical_labeler.vectors (ignore > rock > dead_posidonia > sand > other > posidonia).
The polygons must be in the CRS of the mosaic.
"""
import argparse
import json
import os
import sys

import numpy as np
import rasterio
from affine import Affine
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from optical_labeler.vectors import CLASSES, class_masks  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--mosaic", required=True)
    ap.add_argument("--out", required=True, help="output path prefix")
    ap.add_argument("--area", help="GeoJSON of the reviewed area; masks are clipped to it")
    ap.add_argument("--factor", type=int, default=4)
    args = ap.parse_args()
    with rasterio.open(args.mosaic) as src:
        h, w = src.height // args.factor, src.width // args.factor
        t = src.transform * Affine.scale(src.width / w, src.height / h)
    feats = json.load(open(args.vectors))["features"]
    area_feats = json.load(open(args.area))["features"] if args.area else None
    masks, area = class_masks(feats, (h, w), t, area_feats)
    if area is not None:
        masks["area"] = area
    for name in CLASSES + (("area",) if area is not None else ()):
        Image.fromarray(masks[name].astype(np.uint8) * 255).save(f"{args.out}_{name}.png")
        print(name, int(masks[name].sum()), "px")


if __name__ == "__main__":
    main()
