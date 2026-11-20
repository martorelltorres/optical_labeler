#!/usr/bin/env python3
"""Rasterize reviewed Posidonia / rock polygons (GeoJSON from auto_label.py, edited in a
GIS) into masks in the format of the manual ones, on the grid of the optical mosaic reduced by
--factor. Writes <out>_posidonia.png and <out>_rock.png (255 = class); a pixel in both classes
is Posidonia. The polygons must be in the CRS of the mosaic.
"""
import argparse
import json

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.features import rasterize


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--mosaic", required=True)
    ap.add_argument("--out", required=True, help="output path prefix")
    ap.add_argument("--factor", type=int, default=4)
    args = ap.parse_args()
    with rasterio.open(args.mosaic) as src:
        h, w = src.height // args.factor, src.width // args.factor
        t = src.transform * Affine.scale(src.width / w, src.height / h)
    feats = json.load(open(args.vectors))["features"]
    masks = {}
    for name in ("posidonia", "rock"):
        geoms = [f["geometry"] for f in feats if f["properties"].get("class") == name and f["geometry"]]
        masks[name] = rasterize(((g, 1) for g in geoms), out_shape=(h, w), transform=t,
                                dtype="uint8") if geoms else np.zeros((h, w), np.uint8)
    masks["rock"][masks["posidonia"] > 0] = 0
    for name, a in masks.items():
        Image.fromarray(a * 255).save(f"{args.out}_{name}.png")
        print(name, int(a.sum()), "px")


if __name__ == "__main__":
    main()
