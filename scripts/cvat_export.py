#!/usr/bin/env python3
"""Export an optical mosaic for annotation in CVAT: <out>/<name>.jpg (the mosaic reduced by
--factor, invalid pixels black), <name>.json (its grid, for cvat_import.py) and, with --labels,
<name>_annotations.xml (the GeoJSON polygons rasterized as one CVAT brush mask per class, holes
included, in CVAT for images 1.1, to upload as pre-annotations).
"""
import argparse
import json
import os
import sys

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.enums import ColorInterp, Resampling

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from optical_labeler.cvat import masks_xml, write_sidecar  # noqa: E402
from optical_labeler.vectors import class_masks  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mosaic", required=True)
    ap.add_argument("--out", required=True, help="output directory (e.g. the CVAT share)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--labels", help="GeoJSON polygons with a class property, in the mosaic CRS")
    ap.add_argument("--area", help="GeoJSON of the area to annotate (recorded for cvat_import.py)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with rasterio.open(args.mosaic) as src:
        h, w = src.height // args.factor, src.width // args.factor
        rgb = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.average)
        alpha = [i for i, c in enumerate(src.colorinterp, 1) if c == ColorInterp.alpha]
        valid = (src.read(alpha[0], out_shape=(h, w), resampling=Resampling.average) > 127) if alpha \
            else rgb.sum(0) > 0
        t = src.transform * Affine.scale(src.width / w, src.height / h)
        crs = src.crs
    img = np.moveaxis(rgb, 0, -1).astype(np.uint8)
    img[~valid] = 0
    Image.fromarray(img).save(os.path.join(args.out, f"{args.name}.jpg"), quality=92)
    rel = lambda p: os.path.relpath(os.path.abspath(p), ROOT)
    write_sidecar(os.path.join(args.out, f"{args.name}.json"), rel(args.mosaic), args.factor,
                  w, h, t, crs, rel(args.area) if args.area else None)
    if args.labels:
        feats = json.load(open(args.labels))["features"]
        masks, _ = class_masks(feats, (h, w), t)
        xml = masks_xml(masks, f"{args.name}.jpg", w, h)
        open(os.path.join(args.out, f"{args.name}_annotations.xml"), "w").write(xml)
    print(f"{args.name}: {w} x {h} px at {abs(t.a) * 100:.2f} cm/px")


if __name__ == "__main__":
    main()
