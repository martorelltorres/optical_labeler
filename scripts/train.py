#!/usr/bin/env python3
"""Train and evaluate the optical segmenter (optical_labeler.optical).

Protocols:
  cv        spatial k-fold cross-validation on --site (folds from configs/sites.yaml)
  transfer  train on every annotated pixel of --site, test on each site of --test

Each run directory (<out>/<name>/<protocol>/<split>/seed<k>) gets model.pt, metrics.json
(IoU counts per class and test site) and prob_<site>.tif, the class probabilities over the
whole test mosaic (uint8 0..250 = p * 250, 255 = no data; bands in OPTICAL_CODES order).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import rasterio
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from optical_labeler.infer import predict_mosaic  # noqa: E402
from optical_labeler.codes import CLASS_NAMES  # noqa: E402
from optical_labeler.optical import (MASK_ORDER, OPTICAL_CODES, OpticalTiles,  # noqa: E402
                                     build_optical_unet, class_iou, load_optical, optical_folds,
                                     optical_loss, optical_predictor, region_codes)
from optical_labeler.split import NO_FOLD  # noqa: E402


def lr_factor(it, warmup, total):
    if it < warmup:
        return (it + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (it - warmup) / max(total - warmup, 1)))


def write_class_probs(path, probs, m):
    q = np.where(np.isfinite(probs), np.round(np.nan_to_num(probs) * 250), 255).astype(np.uint8)
    profile = dict(driver="GTiff", height=q.shape[1], width=q.shape[2], count=q.shape[0],
                   dtype="uint8", nodata=255, crs=m.crs, transform=m.transform, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(q)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True, choices=("cv", "transfer"))
    ap.add_argument("--site", default="andratx_135222", help="training site")
    ap.add_argument("--test", nargs="*", default=[], help="test sites of the transfer protocol")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "unet.yaml"))
    ap.add_argument("--sites", default=os.path.join(ROOT, "configs", "sites.yaml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.iters:
        cfg["iters"] = args.iters
    scfg = yaml.safe_load(open(args.sites))
    rel = lambda p: p and os.path.join(ROOT, p)
    def load(n):
        site = scfg["sites"][n]
        masks = {k: rel(site.get(f"{k}_mask")) for k in MASK_ORDER}
        return load_optical(rel(site["mosaic"]), masks, scfg["factor"], rel(site.get("area_mask")),
                            site.get("annotated"))
    band, fc = scfg.get("band_m", 0.0), scfg["folds"]
    train_site = load(args.site)
    folds = optical_folds(train_site.codes, train_site.res, fc["k"], fc["block_m"], fc["gap_m"])
    if args.protocol == "cv":
        jobs = [(f"fold{f}", [g for g in range(fc["k"]) if g != f], [(args.site, train_site, [f])])
                for f in range(fc["k"])]
    else:
        jobs = [("all", None, [(n, load(n), None) for n in args.test])]
    device = torch.device(args.device)

    for split_id, train_folds, tests in jobs:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        codes = region_codes(train_site, folds, train_folds, band)
        ds = OpticalTiles([(train_site, codes)], cfg["tile"], cfg["iters"] * cfg["batch"],
                          cfg["pos_fraction"], cfg.get("augment"), seed=args.seed)
        dl = DataLoader(ds, batch_size=cfg["batch"], num_workers=args.workers, drop_last=True)
        model = build_optical_unet(**cfg["model"]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda it: lr_factor(it, cfg["warmup_iters"], cfg["iters"]))
        curve, window, t0 = [], [], time.time()
        model.train()
        for it, (img, code, allowed) in enumerate(dl):
            loss = optical_loss(model(img.to(device)), code.to(device), allowed)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            window.append(loss.item())
            if (it + 1) % 50 == 0 or it + 1 == cfg["iters"]:
                curve.append([it + 1, float(np.mean(window))])
                window = []
                print(f"{split_id} it {it + 1}/{cfg['iters']} loss {curve[-1][1]:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        d = os.path.join(args.out, cfg["name"], args.protocol, split_id, f"seed{args.seed}")
        os.makedirs(d, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg, "factor": scfg["factor"],
                    "res_m": train_site.res, "classes": list(OPTICAL_CODES)}, os.path.join(d, "model.pt"))
        record = {"model": cfg["name"], "protocol": args.protocol, "split": split_id,
                  "seed": args.seed, "train_site": args.site, "train_folds": train_folds,
                  "loss_curve": curve, "train_seconds": time.time() - t0, "test": {}}
        predict = optical_predictor(model, device)
        for name, m, test_folds in tests:
            probs = predict_mosaic(predict, m.image, m.valid, **cfg["infer"])
            pred = np.asarray(OPTICAL_CODES, np.uint8)[np.nan_to_num(probs).argmax(0)]
            tcodes = region_codes(m, folds if test_folds else None, test_folds, band)
            record["test"][name] = class_iou(pred, tcodes, m.valid & (tcodes != 255), m.annotated)
            write_class_probs(os.path.join(d, f"prob_{name}.tif"), probs, m)
        json.dump(record, open(os.path.join(d, "metrics.json"), "w"), indent=1)
        print(json.dumps({n: {CLASS_NAMES[int(c)]: round(v["inter"] / max(v["union"], 1), 3)
                              for c, v in r.items() if v["complete"]}
                          for n, r in record["test"].items()}), flush=True)


if __name__ == "__main__":
    main()
