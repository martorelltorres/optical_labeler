#!/usr/bin/env python3
"""Propose Posidonia / rock labels on an optical mosaic with one or more trained segmenters.

Writes, for review in a GIS and use as label masks:
  <out>_<class>.png                     one mask per class the models predict (posidonia,
                                        dead_posidonia, rock, sand, other; 255 = class) in the
                                        format of the manual ones, covering the whole mosaic;
                                        usable as <class>_mask in a sites configuration
  <out>_prob.tif                        class probabilities (uint8 0..250, one band per class
                                        in the models' order, named in the band descriptions;
                                        mean over the models)
  <out>_doubt.tif                       band 1: 1 - highest class probability; band 2: standard
                                        deviation of P(Posidonia) between models (float32)
  <out>_labels.geojson                  polygons of every class (mosaic CRS) with their mean
                                        probability and doubt, to edit and convert back with
                                        scripts/masks_from_vectors.py

The mosaic is reduced to the working resolution of the models (the resolution they were
trained at), whatever its own resolution.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.features import rasterize, shapes, sieve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from optical_labeler.infer import predict_mosaic  # noqa: E402
from optical_labeler.codes import BACKGROUND, CLASS_NAMES, POSIDONIA  # noqa: E402
from optical_labeler.optical import (LEGACY_CODES, build_optical_unet, load_optical,  # noqa: E402
                                     optical_predictor)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="model.pt files or glob patterns")
    ap.add_argument("--mosaic", required=True, help="optical mosaic (GeoTIFF, RGB or RGBA)")
    ap.add_argument("--out", required=True, help="output path prefix")
    ap.add_argument("--min-area-m2", type=float, default=0.05,
                    help="regions smaller than this are merged into their surroundings")
    ap.add_argument("--train-res-m", type=float,
                    help="working resolution of models saved without it (m per pixel)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    paths = sorted(p for pat in args.models for p in glob.glob(pat))
    if not paths:
        raise SystemExit("no models found")
    device = torch.device(args.device)
    ckpts = [torch.load(p, map_location=device) for p in paths]
    classes = {tuple(c.get("classes", LEGACY_CODES)) for c in ckpts}
    if len(classes) != 1:
        raise SystemExit("models predict different classes and cannot be averaged")
    codes = classes.pop()
    names = {c: CLASS_NAMES[c] for c in codes if c != BACKGROUND}
    res = {c.get("res_m") or args.train_res_m for c in ckpts}
    if None in res or len({round(r, 4) for r in res}) != 1:
        raise SystemExit("models trained at different or unknown resolutions (see --train-res-m)")
    with rasterio.open(args.mosaic) as src:
        factor = max(int(round(res.pop() / abs(src.transform.a))), 1)
    m = load_optical(args.mosaic, factor=factor)
    probs = []
    for c in ckpts:
        model = build_optical_unet(**dict(c["config"]["model"], encoder_weights=None),
                                   classes=len(codes)).to(device)
        model.load_state_dict(c["state_dict"])
        probs.append(predict_mosaic(optical_predictor(model, device), m.image, m.valid,
                                    **c["config"]["infer"]))
    probs = np.stack(probs)
    mean = probs.mean(0)
    ok = np.isfinite(mean[0])
    ipos = codes.index(POSIDONIA)
    doubt = np.stack([1.0 - np.nan_to_num(mean).max(0), probs[:, ipos].std(0)]).astype(np.float32)
    doubt[:, ~ok] = np.nan
    cls = np.asarray(codes, np.uint8)[np.nan_to_num(mean).argmax(0)]
    min_px = max(int(round(args.min_area_m2 / m.res ** 2)), 1)
    cls = sieve(np.where(ok, cls, 255).astype(np.uint8), min_px, mask=ok).astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for code, name in names.items():
        Image.fromarray(np.where(ok & (cls == code), 255, 0).astype(np.uint8)).save(f"{args.out}_{name}.png")
    base = dict(driver="GTiff", height=m.valid.shape[0], width=m.valid.shape[1], crs=m.crs,
                transform=m.transform, compress="deflate")
    q = np.where(np.isfinite(mean), np.round(np.nan_to_num(mean) * 250), 255).astype(np.uint8)
    with rasterio.open(f"{args.out}_prob.tif", "w", **base, count=len(codes), dtype="uint8",
                       nodata=255) as dst:
        dst.write(q)
        for b, c in enumerate(codes, 1):
            dst.set_band_description(b, CLASS_NAMES.get(c, "background"))
    with rasterio.open(f"{args.out}_doubt.tif", "w", **base, count=2, dtype="float32", nodata=np.nan) as dst:
        dst.write(doubt)

    feats = []
    for code, name in names.items():
        sel = ok & (cls == code)
        geoms = [g for g, _ in shapes(sel.astype(np.uint8), mask=sel, transform=m.transform)]
        if not geoms:
            continue
        ids = rasterize(((g, k) for k, g in enumerate(geoms, 1)), out_shape=cls.shape,
                        transform=m.transform, dtype="int32")
        n = np.bincount(ids.ravel(), minlength=len(geoms) + 1).clip(min=1)
        pk = np.bincount(ids.ravel(), np.nan_to_num(mean[codes.index(code)]).ravel(),
                         minlength=len(geoms) + 1) / n
        dk = np.bincount(ids.ravel(), np.nan_to_num(doubt[0]).ravel(), minlength=len(geoms) + 1) / n
        for k, g in enumerate(geoms, 1):
            feats.append({"type": "Feature", "geometry": g,
                          "properties": {"class": name, "area_m2": round(float(n[k] * m.res ** 2), 3),
                                         "p_mean": round(float(pk[k]), 3),
                                         "doubt_mean": round(float(dk[k]), 3)}})
    epsg = m.crs.to_epsg()
    json.dump({"type": "FeatureCollection",
               "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}},
               "features": feats}, open(f"{args.out}_labels.geojson", "w"))
    area = lambda c: (ok & (cls == c)).sum() * m.res ** 2
    cover = ", ".join(f"{n} {area(c):.0f} m2" for c, n in names.items())
    print(f"{len(paths)} models; {ok.sum() * m.res ** 2:.0f} m2 with data; {cover}; {len(feats)} polygons; "
          f"median doubt {np.nanmedian(doubt[0]):.3f}")


if __name__ == "__main__":
    main()
