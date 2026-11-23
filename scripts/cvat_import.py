#!/usr/bin/env python3
"""Convert annotations exported from CVAT ("CVAT for images 1.1", the annotations.xml inside the
export zip) into class masks: <out>_<class>.png for posidonia, dead_posidonia, rock, sand, other
and ignore (255 = class) on the exported image, i.e. a downscaled copy of the mosaic, usable as
<class>_mask in configs/sites.yaml; and <out>_area.png when the export recorded an area.
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np
from PIL import Image
from rasterio.features import rasterize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from optical_labeler.cvat import masks_from_xml, read_sidecar  # noqa: E402
from optical_labeler.vectors import CLASSES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", required=True, help="CVAT export zip or its annotations.xml")
    ap.add_argument("--sidecar", required=True, help="<name>.json written by cvat_export.py")
    ap.add_argument("--out", required=True, help="output path prefix")
    args = ap.parse_args()
    s = read_sidecar(args.sidecar)
    if args.annotations.endswith(".zip"):
        with zipfile.ZipFile(args.annotations) as z:
            xml = z.read("annotations.xml").decode()
    else:
        xml = open(args.annotations).read()
    name = os.path.splitext(os.path.basename(args.sidecar))[0] + ".jpg"
    masks = masks_from_xml(xml, name, s["width"], s["height"])
    if s.get("area"):
        feats = json.load(open(os.path.join(ROOT, s["area"])))["features"]
        area = rasterize(((f["geometry"], 1) for f in feats), out_shape=(s["height"], s["width"]),
                         transform=s["transform"], dtype="uint8").astype(bool)
        for c in CLASSES:
            masks[c] &= area
        masks["area"] = area
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for c, m in masks.items():
        Image.fromarray(m.astype(np.uint8) * 255).save(f"{args.out}_{c}.png")
        print(c, int(m.sum()), "px")


if __name__ == "__main__":
    main()
