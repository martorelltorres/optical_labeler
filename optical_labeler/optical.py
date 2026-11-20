"""Optical (RGB) mosaics with manual class masks, and a U-Net that proposes those classes.

The segmenter labels camera mosaics automatically; its masks have the format of the manual
ones (PNG, 255 = class), so they can replace them wherever those are used.

Label codes as in optical_labeler.codes: BACKGROUND (annotated, neither Posidonia nor rock),
POSIDONIA, ROCK, IGNORE (band around Posidonia boundaries), NODATA (not annotated).
The network predicts OPTICAL_CODES in this order.
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

from optical_labeler.codes import BACKGROUND, IGNORE, NODATA, POSIDONIA, ROCK, apply_band
from optical_labeler.split import NO_FOLD, block_folds
from optical_labeler.tiling import CenterSampler, geometric, scaled_crop

Image.MAX_IMAGE_PIXELS = None
OPTICAL_CODES = (BACKGROUND, POSIDONIA, ROCK)
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

    @property
    def res(self):
        return abs(self.transform.a)


def read_class_mask(path, shape):
    """Boolean mask drawn on any downscaled copy of a mosaic, resampled to shape (H, W)."""
    m = Image.open(path).convert("L")
    return np.asarray(m.resize((shape[1], shape[0]), Image.BOX)) > 127


def load_optical(mosaic, posidonia_mask=None, rock_mask=None, factor=4, ignore_mask=None,
                 area_mask=None):
    """Read an RGB(A) mosaic reduced by `factor` (area average) and its class masks.

    Valid pixels come from the alpha band (or non-black RGB without one). Without masks every
    code is NODATA; with them the valid pixels are BACKGROUND, ROCK or POSIDONIA (Posidonia over
    rock where both masks are set). area_mask limits the annotated pixels to the reviewed area
    (NODATA outside it); ignore_mask marks pixels that are IGNORE whatever the other masks say.
    """
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
    if posidonia_mask is not None:
        lab = valid & read_class_mask(area_mask, (h, w)) if area_mask is not None else valid
        codes[lab] = BACKGROUND
        if rock_mask is not None:
            codes[lab & read_class_mask(rock_mask, (h, w))] = ROCK
        codes[lab & read_class_mask(posidonia_mask, (h, w))] = POSIDONIA
        if ignore_mask is not None:
            codes[lab & read_class_mask(ignore_mask, (h, w))] = IGNORE
    return OpticalMosaic(img, valid, codes, transform, crs, source_shape)


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


def build_optical_unet(encoder="resnet34", encoder_weights="imagenet"):
    net = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3,
                   classes=len(OPTICAL_CODES))
    return RGBNormalized(net)


def optical_loss(logits, code, dice_weight=1.0):
    """Cross-entropy over annotated pixels plus soft Dice of P(Posidonia) over them."""
    code = code[:, 0].long()
    lut = torch.full((256,), -1, dtype=torch.long, device=code.device)
    for i, c in enumerate(OPTICAL_CODES):
        lut[c] = i
    idx = lut[code]
    m = idx >= 0
    ce = F.cross_entropy(logits, idx.clamp(min=0), reduction="none")
    n = m.sum().clamp(min=1)
    p = torch.softmax(logits, 1)[:, OPTICAL_CODES.index(POSIDONIA)] * m
    t = ((code == POSIDONIA) & m).float()
    dice = 1.0 - (2 * (p * t).sum() + 1.0) / (p.sum() + t.sum() + 1.0)
    return (ce * m).sum() / n + dice_weight * dice


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

    Items are (image float32 (3, tile, tile), codes uint8 (1, tile, tile)).
    """

    def __init__(self, sources, tile, length, pos_fraction=0.5, aug=None, seed=0):
        self.items, w = [], []
        for m, codes in sources:
            lab = codes < IGNORE
            self.items.append((m, codes, CenterSampler(lab, codes == POSIDONIA, pos_fraction)))
            w.append(float(lab.sum()))
        self.p = np.asarray(w) / sum(w)
        self.tile, self.length, self.seed = tile, length, seed
        self.aug = dict(COLOR_DEFAULTS, **(aug or {})) if aug is not False else None

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, torch.initial_seed() % 2**31))
        m, codes, sampler = self.items[rng.choice(len(self.items), p=self.p)]
        scale = rng.uniform(*self.aug["scale"]) if self.aug else 1.0
        img, val, code = scaled_crop([m.image, m.valid, codes], sampler(rng), self.tile,
                                             scale, [0.0, False, NODATA])
        if self.aug:
            img, val, code = geometric([img, val, code], rng)
            img = color_jitter(img, val, rng, self.aug)
        img = torch.from_numpy(np.ascontiguousarray(np.moveaxis(img, -1, 0), np.float32))
        return img, torch.from_numpy(np.ascontiguousarray(code, np.uint8))[None]


def optical_predictor(model, device):
    """predict_batch for optical_labeler.infer.predict_mosaic: (B, t, t, 3) -> class probs (B, C, t, t)."""
    @torch.no_grad()
    def predict(x):
        model.eval()
        t = torch.from_numpy(np.moveaxis(x, -1, 1).copy()).to(device)
        return torch.softmax(model(t), 1).cpu().numpy()
    return predict


def class_iou(pred_codes, codes, mask):
    """IoU per class of OPTICAL_CODES over the annotated pixels of mask, plus counts."""
    m = mask & (codes < IGNORE)
    out = {}
    for c in OPTICAL_CODES:
        p, t = (pred_codes == c) & m, (codes == c) & m
        inter, union = int((p & t).sum()), int((p | t).sum())
        out[str(c)] = {"inter": inter, "union": union, "true": int(t.sum())}
    return out
