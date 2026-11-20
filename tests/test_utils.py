import unittest

import numpy as np

from optical_labeler.codes import IGNORE, POSIDONIA, ROCK, apply_band, boundary_band
from optical_labeler.infer import predict_mosaic
from optical_labeler.split import NO_FOLD, block_folds
from optical_labeler.tiling import CenterSampler, crop, geometric, scaled_crop


class BoundaryBandTest(unittest.TestCase):
    def test_band_straddles_the_transition(self):
        pos = np.zeros((10, 20), bool)
        pos[:, 10:] = True
        band = boundary_band(pos, np.ones_like(pos), 2)
        self.assertTrue(band[:, 8:12].all())
        self.assertFalse(band[:, :7].any() or band[:, 13:].any())

    def test_zero_width_is_empty(self):
        pos = np.eye(5, dtype=bool)
        self.assertFalse(boundary_band(pos, np.ones_like(pos), 0).any())

    def test_band_uses_every_labeled_class(self):
        lab = np.full((10, 20), ROCK, np.uint8)
        lab[:, :10] = POSIDONIA
        out = apply_band(lab, 2)
        self.assertTrue((out[:, 8:12] == IGNORE).all())
        self.assertTrue((out[:, :7] == POSIDONIA).all() and (out[:, 13:] == ROCK).all())


class TilingTest(unittest.TestCase):
    def test_crop_pads_outside(self):
        a = np.ones((5, 5), np.float32)
        w = crop(a, 0, 0, 4, fill=-1)
        self.assertEqual(w.shape, (4, 4))
        self.assertTrue((w[:2] == -1).all() and (w[2:, 2:] == 1).all())

    def test_centres_inside_mask(self):
        rng = np.random.default_rng(0)
        mask = rng.random((40, 50)) < 0.3
        s = CenterSampler(mask, rng.random((40, 50)) < 0.5, 0.5)
        for _ in range(200):
            r, c = s(rng)
            self.assertTrue(mask[r, c])

    def test_geometry_keeps_image_and_target_aligned(self):
        rng = np.random.default_rng(1)
        img = (rng.random((32, 32)) > 0.5).astype(np.float32)
        for _ in range(10):
            scale = rng.uniform(0.8, 1.25)
            a, t = scaled_crop([img, img.copy()], (16, 16), 24, scale, [0.0, 0.0])
            a, t = geometric([a, t], rng)
            self.assertEqual(a.shape, (24, 24))
            # Linear resampling of the image must agree with the nearest-neighbour target
            # wherever the image is unambiguous.
            sure = (a < 0.1) | (a > 0.9)
            self.assertTrue((np.round(a[sure]) == t[sure]).all())


class BlockFoldsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.pos = np.kron(rng.random((12, 12)) < 0.5, np.ones((10, 10), bool))
        self.neg = ~self.pos

    def test_classes_are_balanced(self):
        folds, info = block_folds(self.pos, self.neg, k=3, block_px=10)
        self.assertLess(info["score"], 0.05)

    def test_blocks_are_not_split(self):
        folds, _ = block_folds(self.pos, self.neg, k=3, block_px=10)
        for r in range(0, 120, 10):
            for c in range(0, 120, 10):
                self.assertEqual(len(np.unique(folds[r:r + 10, c:c + 10])), 1)

    def test_gap_separates_folds(self):
        folds, _ = block_folds(self.pos, self.neg, k=3, block_px=10, gap_px=4)
        self.assertTrue((folds == NO_FOLD).any())
        for a, b in [(folds[1:], folds[:-1]), (folds[:, 1:], folds[:, :-1])]:
            both = (a != NO_FOLD) & (b != NO_FOLD)
            self.assertTrue((a[both] == b[both]).all())


class PredictMosaicTest(unittest.TestCase):
    def test_identity_model_reconstructs_input(self):
        rng = np.random.default_rng(0)
        img = rng.random((150, 230)).astype(np.float32)
        valid = np.ones_like(img, bool)
        valid[:10, :10] = False
        out = predict_mosaic(lambda x: x, img, valid, tile=64, overlap=16, batch=3)
        self.assertTrue(np.isnan(out[:10, :10]).all())
        np.testing.assert_allclose(out[valid], img[valid], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
