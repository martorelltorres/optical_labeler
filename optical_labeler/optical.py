"""Optical (RGB) mosaics with manual class masks, and a U-Net that proposes those classes.

The segmenter labels camera mosaics automatically; its masks have the format of the manual
ones (PNG, 255 = class), so they can replace them wherever those are used.

Label codes as in optical_labeler.codes. The network predicts the classes of OPTICAL_CODES in
that order. Sites are annotated partially: each declares the classes it annotates, and a
BACKGROUND pixel there is only known to be one of the other classes (partial-label loss).
Models trained with the earlier three outputs (LEGACY_CODES) are still read.
"""
from dataclasses import dataclass

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from affine import Affine
from PIL import Image
from rasterio.enums import ColorInterp, Resampling
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

from optical_labeler.codes import (BACKGROUND, DEAD_POSIDONIA, IGNORE, NAME_CODES, NODATA,
                                   OTHER, POSIDONIA, ROCK, SAND, apply_band)
from optical_labeler.split import NO_FOLD, block_folds
from optical_labeler.tiling import CenterSampler, geometric, scaled_crop

Image.MAX_IMAGE_PIXELS = None
OPTICAL_CODES = (POSIDONIA, DEAD_POSIDONIA, ROCK, SAND, OTHER)
LEGACY_CODES = (BACKGROUND, POSIDONIA, ROCK)
MASK_ORDER = ("other", "sand", "rock", "dead_posidonia", "posidonia", "ignore")   # later wins
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
COLOR_DEFAULTS = {"scale": (0.8, 1.25), "gain": (0.8, 1.2), "contrast": (0.8, 1.2),
                  "brightness": (-0.08, 0.08), "noise_p": 0.3, "noise_sigma": (0.005, 0.03),
                  "blur_p": 0.2, "blur_sigma": (0.3, 1.2)}


@dataclass
class OpticalMosaic:
    image: np.ndarray       # float32 (H, W, 3) in [0, 1]
    valid: np.ndarray       # bool (H, W)
    codes: np.ndarray       # uint8 (H, W); NODATA without annotation
    transform: Affine
    crs: object
    source_shape: tuple     # (height, width) of the mosaic file
    annotated: tuple = ()   # class codes annotated; BACKGROUND pixels are one of the others

    @property
    def res(self):
        return abs(self.transform.a)


def read_class_mask(path, shape):
    """Boolean mask drawn on any downscaled copy of a mosaic, resampled to shape (H, W)."""
    m = Image.open(path).convert("L")
    return np.asarray(m.resize((shape[1], shape[0]), Image.BOX)) > 127


def load_optical(mosaic, masks=None, factor=4, area_mask=None, annotated=None):
    """Read an RGB(A) mosaic reduced by `factor` (area average) and its class masks.

    masks: {name: path} with names among optical_labeler.codes.CLASS_NAMES values and "ignore".
    Valid pixels come from the alpha band (or non-black RGB without one). Without class masks
    every code is NODATA; with them the valid pixels inside area_mask (all valid pixels without
    one) are annotated: the class of their mask, IGNORE under the ignore mask, BACKGROUND
    elsewhere. Overlaps resolve in MASK_ORDER (Posidonia over rock, ignore over everything).
    annotated: class names whose extent is complete at this site, so that BACKGROUND means "none
    of them"; defaults to the classes given in masks.
    """
    masks = {k: v for k, v in (masks or {}).items() if v is not None}
    unknown = set(masks) - set(MASK_ORDER)
    if unknown:
        raise ValueError(f"unknown mask names {sorted(unknown)}")
    with rasterio.open(mosaic) as src:
        h, w = src.height // factor, src.width // factor
        rgb = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.average)
        alpha = [i for i, c in enumerate(src.colorinterp, 1) if c == ColorInterp.alpha]
        if alpha:
            valid = src.read(alpha[0], out_shape=(h, w), resampling=Resampling.average) > 127
        else:
            valid = rgb.sum(0) > 0
        transform = src.transform * Affine.scale(src.width / w, src.height / h)
        crs, source_shape = src.crs, (src.height, src.width)
    img = np.moveaxis(rgb, 0, -1).astype(np.float32) / 255.0
    img[~valid] = 0.0
    codes = np.full((h, w), NODATA, np.uint8)
    classes = [n for n in masks if n != "ignore"]
    if classes:
        lab = valid & read_class_mask(area_mask, (h, w)) if area_mask is not None else valid
        codes[lab] = BACKGROUND
        for name in MASK_ORDER:
            if name in masks:
                code = IGNORE if name == "ignore" else NAME_CODES[name]
                codes[lab & read_class_mask(masks[name], (h, w))] = code
    names = classes if annotated is None else list(annotated)
    bad = set(names) - set(NAME_CODES)
    if bad:
        raise ValueError(f"unknown annotated classes {sorted(bad)}")
    return OpticalMosaic(img, valid, codes, transform, crs, source_shape,
                         tuple(NAME_CODES[n] for n in names))


def background_classes(annotated, class_codes=OPTICAL_CODES):
    """Boolean vector over class_codes: the classes a BACKGROUND pixel can be at a site."""
    return np.array([c not in annotated for c in class_codes], bool)


def optical_folds(codes, res, k=3, block_m=5.0, gap_m=0.5):
    """Spatial folds of the annotated pixels (optical_labeler.split.block_folds); NO_FOLD elsewhere."""
    pos = codes == POSIDONIA
    neg = (codes < IGNORE) & ~pos
    folds, _ = block_folds(pos, neg, k, max(int(round(block_m / res)), 1), gap_m / res)
    return folds


def region_codes(m, folds=None, train_folds=None, band_m=0.0):
    """Codes with the boundary band applied, NODATA outside train_folds (all if None)."""
    codes = apply_band(m.codes, band_m / m.res) if band_m else m.codes.copy()
    if train_folds is not None:
        codes[~np.isin(folds, list(train_folds)) | (folds == NO_FOLD)] = NODATA
    return codes


class RGBNormalized(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def build_optical_unet(encoder="resnet34", encoder_weights="imagenet", classes=len(OPTICAL_CODES)):
    net = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3,
                   classes=classes)
    return RGBNormalized(net)


def optical_loss(logits, code, allowed, dice_weight=1.0):
    """Partial-label loss over the classes of OPTICAL_CODES, plus soft Dice of P(Posidonia).

    code (B, 1, H, W): pixels of a class add -log p(class); BACKGROUND pixels add
    -log(sum of p over the classes allowed for their tile), allowed being a boolean (B, C) from
    background_classes, and carry no loss when nothing is allowed. The Dice of Posidonia counts
    class pixels and the BACKGROUND pixels of tiles where Posidonia is annotated.
    """
    code = code[:, 0].long()
    lut = torch.full((256,), -1, dtype=torch.long, device=code.device)
    for i, c in enumerate(OPTICAL_CODES):
        lut[c] = i
    idx = lut[code]
    specific = idx >= 0
    logp = F.log_softmax(logits, 1)
    ce = -(logp.gather(1, idx.clamp(min=0)[:, None])[:, 0] * specific).sum()
    allowed = allowed.to(logits.device).bool()
    masked = logp.masked_fill(~allowed[:, :, None, None], float("-inf"))
    has_any = allowed.any(1)[:, None, None]
    unspecified = (code == BACKGROUND) & has_any
    lse = torch.logsumexp(masked, 1).masked_fill(~unspecified, 0.0)
    partial = -lse.sum()
    n = (specific | unspecified).sum().clamp(min=1)
    ipo = OPTICAL_CODES.index(POSIDONIA)
    po_known = (~allowed[:, ipo])[:, None, None]
    m = (specific | (unspecified & po_known)).float()
    p = logp[:, ipo].exp() * m
    t = (code == POSIDONIA).float() * m
    dice = 1.0 - (2 * (p * t).sum() + 1.0) / (p.sum() + t.sum() + 1.0)
    return (ce + partial) / n + dice_weight * dice


def color_jitter(img, valid, rng, p=COLOR_DEFAULTS):
    """Per-channel gain, contrast, brightness, noise and blur; invalid pixels stay 0."""
    x = img * rng.uniform(*p["gain"], size=3).astype(np.float32)
    if valid.any():
        mean = x[valid].mean(0)
        x = (x - mean) * rng.uniform(*p["contrast"]) + mean + rng.uniform(*p["brightness"])
    if rng.random() < p["noise_p"]:
        x = x + rng.normal(0.0, rng.uniform(*p["noise_sigma"]), x.shape)
    if rng.random() < p["blur_p"]:
        s = rng.uniform(*p["blur_sigma"])
        x = gaussian_filter(x, (s, s, 0))
    x = np.clip(x, 0.0, 1.0)
    x[~valid] = 0.0
    return x.astype(np.float32)


class OpticalTiles(Dataset):
    """Random augmented tiles of (mosaic, codes) pairs; centres on Posidonia with pos_fraction.

    Items are (image float32 (3, tile, tile), codes uint8 (1, tile, tile), allowed bool (C,)):
    allowed lists the classes a BACKGROUND pixel of that mosaic can be (background_classes).
    """

    def __init__(self, sources, tile, length, pos_fraction=0.5, aug=None, seed=0):
        self.items, w = [], []
        for m, codes in sources:
            lab = codes < IGNORE
            self.items.append((m, codes, CenterSampler(lab, codes == POSIDONIA, pos_fraction),
                               torch.from_numpy(background_classes(m.annotated))))
            w.append(float(lab.sum()))
        self.p = np.asarray(w) / sum(w)
        self.tile, self.length, self.seed = tile, length, seed
        self.aug = dict(COLOR_DEFAULTS, **(aug or {})) if aug is not False else None

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, torch.initial_seed() % 2**31))
        m, codes, sampler, allowed = self.items[rng.choice(len(self.items), p=self.p)]
        scale = rng.uniform(*self.aug["scale"]) if self.aug else 1.0
        img, val, code = scaled_crop([m.image, m.valid, codes], sampler(rng), self.tile,
                                             scale, [0.0, False, NODATA])
        if self.aug:
            img, val, code = geometric([img, val, code], rng)
            img = color_jitter(img, val, rng, self.aug)
        img = torch.from_numpy(np.ascontiguousarray(np.moveaxis(img, -1, 0), np.float32))
        return img, torch.from_numpy(np.ascontiguousarray(code, np.uint8))[None], allowed


def optical_predictor(model, device):
    """predict_batch for optical_labeler.infer.predict_mosaic: (B, t, t, 3) -> class probs (B, C, t, t)."""
    @torch.no_grad()
    def predict(x):
        model.eval()
        t = torch.from_numpy(np.moveaxis(x, -1, 1).copy()).to(device)
        return torch.softmax(model(t), 1).cpu().numpy()
    return predict


def class_iou(pred_codes, codes, mask, annotated, class_codes=OPTICAL_CODES):
    """IoU counts per class over the labeled pixels of mask.

    For a class annotated at the site every labeled pixel counts (BACKGROUND as "not this class");
    for the others only pixels with a specific class count, and "complete" is False.
    """
    lab = mask & (codes < IGNORE)
    out = {}
    for c in class_codes:
        if c == BACKGROUND:
            continue
        m = lab if c in annotated else lab & (codes != BACKGROUND)
        p, t = (pred_codes == c) & m, (codes == c) & m
        out[str(c)] = {"inter": int((p & t).sum()), "union": int((p | t).sum()), "true": int(t.sum()),
                       "complete": c in annotated}
    return out
