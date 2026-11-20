"""Sliding-window inference over a whole mosaic with weighted blending of overlaps."""
import numpy as np


def blend_window(size, floor=1e-3):
    w = np.sin(np.linspace(0, np.pi, size + 2)[1:-1]) ** 2
    return np.maximum(np.outer(w, w), floor).astype(np.float32)


def predict_mosaic(predict_batch, image, valid, tile=256, overlap=64, batch=8):
    """Probability map of the same height and width as image; NaN where there is no data.

    image: (H, W) or (H, W, C). predict_batch: callable taking a float32 array (B, tile, tile)
    or (B, tile, tile, C) and returning
    probabilities of the same shape, or (B, C, tile, tile) for C channels, in which case the
    result is (C, H, W).
    """
    if overlap >= tile:
        raise ValueError("overlap must be smaller than tile")
    h, w = image.shape[:2]
    step = tile - overlap
    nr, nc = max(1, -(-(h - overlap) // step)), max(1, -(-(w - overlap) // step))
    H, W = (nr - 1) * step + tile, (nc - 1) * step + tile
    pad = np.zeros((H, W) + image.shape[2:], np.float32)
    pad[:h, :w] = image
    acc = None
    wsum = np.zeros((H, W), np.float32)
    win = blend_window(tile)
    origins = [(r * step, c * step) for r in range(nr) for c in range(nc)]
    vpad = np.zeros((H, W), bool)
    vpad[:h, :w] = valid
    origins = [o for o in origins if vpad[o[0]:o[0] + tile, o[1]:o[1] + tile].any()]
    for i in range(0, len(origins), batch):
        chunk = origins[i:i + batch]
        x = np.stack([pad[r:r + tile, c:c + tile] for r, c in chunk])
        p = predict_batch(x)
        if acc is None:
            acc = np.zeros(p.shape[1:-2] + (H, W), np.float32)
        for (r, c), pi in zip(chunk, p):
            acc[..., r:r + tile, c:c + tile] += pi * win
            wsum[r:r + tile, c:c + tile] += win
    if acc is None:
        return np.full((h, w), np.nan, np.float32)
    out = np.full(acc.shape[:-2] + (h, w), np.nan, np.float32)
    ok = valid & (wsum[:h, :w] > 0)
    out[..., ok] = acc[..., :h, :w][..., ok] / wsum[:h, :w][ok]
    return out
