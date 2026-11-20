import os
import tempfile
import unittest

import numpy as np
import rasterio
import torch
from affine import Affine
from PIL import Image

from optical_labeler.codes import BACKGROUND, IGNORE, NODATA, POSIDONIA, ROCK
from optical_labeler.infer import predict_mosaic
from optical_labeler.optical import (OpticalTiles, build_optical_unet, load_optical, optical_folds,
                                     optical_loss, region_codes)
from optical_labeler.tiling import scaled_crop


def write_mosaic(d, h=80, w=120):
    rgba = np.zeros((4, h, w), np.uint8)
    rgba[:3] = 100
    rgba[3, :, 10:] = 255                      # first 10 columns without data
    path = os.path.join(d, "m.tif")
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=4, dtype="uint8",
                       crs="EPSG:25831", transform=Affine(0.01, 0, 500000, 0, -0.01, 4000000),
                       photometric="RGB", alpha="YES") as dst:
        dst.write(rgba)
    pos = np.zeros((h // 2, w // 2), np.uint8)
    pos[:, 30:] = 255                         # right half, drawn at half resolution
    rock = np.zeros((h // 2, w // 2), np.uint8)
    rock[:10, :20] = 255
    Image.fromarray(pos).save(os.path.join(d, "pos.png"))
    Image.fromarray(rock).save(os.path.join(d, "rock.png"))
    return path


class LoadTest(unittest.TestCase):
    def test_codes_and_grid(self):
        with tempfile.TemporaryDirectory() as d:
            m = load_optical(write_mosaic(d), os.path.join(d, "pos.png"),
                             os.path.join(d, "rock.png"), factor=2)
        self.assertEqual(m.image.shape, (40, 60, 3))
        self.assertAlmostEqual(m.res, 0.02)
        self.assertTrue((m.codes[:, :5] == NODATA).all())          # no data
        self.assertTrue((m.codes[:, 30:] == POSIDONIA).all())
        self.assertEqual(m.codes[5, 15], ROCK)
        self.assertEqual(m.codes[30, 15], BACKGROUND)
        self.assertTrue((m.image[~m.valid] == 0).all())

    def test_ignore_and_area_masks(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_mosaic(d)
            ign = np.zeros((40, 60), np.uint8)
            ign[30:, 40:] = 255                   # inside the Posidonia half
            area = np.zeros((40, 60), np.uint8)
            area[:, :50] = 255                    # last 10 columns not reviewed
            Image.fromarray(ign).save(os.path.join(d, "ign.png"))
            Image.fromarray(area).save(os.path.join(d, "area.png"))
            m = load_optical(path, os.path.join(d, "pos.png"), os.path.join(d, "rock.png"), factor=2,
                             ignore_mask=os.path.join(d, "ign.png"), area_mask=os.path.join(d, "area.png"))
        self.assertTrue((m.codes[:, 50:] == NODATA).all())          # outside the reviewed area
        self.assertTrue((m.codes[30:, 40:50] == IGNORE).all())
        self.assertTrue((m.codes[:30, 30:50] == POSIDONIA).all())
        self.assertEqual(m.codes[5, 15], ROCK)

    def test_folds_and_region(self):
        with tempfile.TemporaryDirectory() as d:
            m = load_optical(write_mosaic(d), os.path.join(d, "pos.png"), None, factor=2)
        folds = optical_folds(m.codes, m.res, k=2, block_m=0.2, gap_m=0.0)
        r = region_codes(m, folds, [0])
        self.assertTrue((r[folds == 1] == NODATA).all())
        self.assertTrue((r[folds == 0] == m.codes[folds == 0]).all())


class ModelTest(unittest.TestCase):
    def test_loss_ignores_nodata(self):
        logits = torch.randn(1, 3, 8, 8, requires_grad=True)
        code = torch.full((1, 1, 8, 8), NODATA, dtype=torch.uint8)
        code[..., :4] = POSIDONIA
        loss = optical_loss(logits, code)
        loss.backward()
        self.assertTrue((logits.grad[..., 4:] == 0).all())

    def test_tiles_and_prediction_shapes(self):
        with tempfile.TemporaryDirectory() as d:
            m = load_optical(write_mosaic(d), os.path.join(d, "pos.png"), None, factor=1)
        ds = OpticalTiles([(m, m.codes)], 32, 4, seed=0)
        img, code = ds[0]
        self.assertEqual(tuple(img.shape), (3, 32, 32))
        self.assertEqual(tuple(code.shape), (1, 32, 32))
        model = build_optical_unet(encoder_weights=None).eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(img[None]).shape), (1, 3, 32, 32))
        probs = predict_mosaic(lambda x: np.ones((x.shape[0], 3) + x.shape[1:3], np.float32) / 3,
                               m.image, m.valid, tile=32, overlap=8, batch=2)
        self.assertEqual(probs.shape, (3, 80, 120))
        self.assertTrue(np.isnan(probs[:, :, :10]).all())

    def test_rgb_crop_keeps_channels(self):
        a = np.random.rand(20, 20, 3).astype(np.float32)
        out = scaled_crop([a], (10, 10), 8, 0.8, [0.0])[0]
        self.assertEqual(out.shape, (8, 8, 3))


if __name__ == "__main__":
    unittest.main()
