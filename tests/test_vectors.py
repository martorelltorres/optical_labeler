import unittest

import numpy as np
from affine import Affine

from optical_labeler.vectors import class_masks

T = Affine(1.0, 0, 0, 0, -1.0, 10.0)        # 10 x 10 grid of 1 m cells, top-left at (0, 10)


def square(x0, y0, x1, y1, cls=None):
    g = {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}
    return {"type": "Feature", "properties": {} if cls is None else {"class": cls}, "geometry": g}


class ClassMasksTest(unittest.TestCase):
    def test_overlaps_resolve_ignore_rock_posidonia(self):
        feats = [square(0, 0, 10, 10, "posidonia"), square(0, 0, 6, 10, "rock"),
                 square(0, 0, 3, 10, "ignore")]
        m, area = class_masks(feats, (10, 10), T)
        self.assertIsNone(area)
        self.assertTrue(m["ignore"][:, :3].all())
        self.assertTrue(m["rock"][:, 3:6].all() and not m["rock"][:, :3].any())
        self.assertTrue(m["posidonia"][:, 6:].all() and not m["posidonia"][:, :6].any())
        total = m["posidonia"].astype(int) + m["rock"] + m["ignore"]
        self.assertEqual(total.max(), 1)

    def test_masks_clipped_to_area(self):
        m, area = class_masks([square(0, 0, 10, 10, "posidonia")], (10, 10), T,
                              [square(0, 5, 10, 10)])
        self.assertTrue(area[:5].all() and not area[5:].any())
        self.assertTrue((m["posidonia"] == area).all())

    def test_unknown_class_is_refused(self):
        with self.assertRaises(ValueError):
            class_masks([square(0, 0, 5, 5, "posidona")], (10, 10), T)
        with self.assertRaises(ValueError):
            class_masks([square(0, 0, 5, 5)], (10, 10), T)


if __name__ == "__main__":
    unittest.main()
