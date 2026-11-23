import unittest

import numpy as np
from affine import Affine

from optical_labeler.cvat import decode_rle, encode_rle, masks_from_xml, masks_xml

T = Affine(0.5, 0, 100.0, 0, -0.5, 200.0)          # 0.5 m pixels, top-left at (100, 200)


def square(x0, y0, x1, y1, cls):
    return {"type": "Feature", "properties": {"class": cls},
            "geometry": {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}}


class CvatTest(unittest.TestCase):
    def test_masks_round_trip_keeps_holes(self):
        po = np.zeros((20, 30), bool); po[2:18, 3:25] = True; po[8:12, 10:15] = False     # a hole
        rk = np.zeros((20, 30), bool); rk[0, 0] = True                                  # touches the corner
        m = masks_from_xml(masks_xml({"posidonia": po, "rock": rk}, "a.jpg", 30, 20), "a.jpg", 30, 20)
        self.assertTrue((m["posidonia"] == po).all() and (m["rock"] == rk).all())

    def test_encode_rle_inverts_decode(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            a = rng.random((7, 9)) < 0.4
            self.assertTrue((decode_rle(encode_rle(a), 9, 7) == a).all())

    def test_polygon_shapes_drawn_in_cvat_are_read(self):
        xml = """<annotations><image id="0" name="a.jpg" width="10" height="10">
          <polygon label="sand" points="0,0;5,0;5,5;0,5"/></image></annotations>"""
        self.assertTrue(masks_from_xml(xml, "a.jpg", 10, 10)["sand"][1:4, 1:4].all())

    def test_rle_mask_and_priority_over_posidonia(self):
        xml = """<annotations><image id="0" name="a.jpg" width="6" height="4">
          <polygon label="posidonia" points="0,0;6,0;6,4;0,4" z_order="0"/>
          <mask label="dead_posidonia" rle="1,4,1" left="1" top="1" width="3" height="2" z_order="1"/>
        </image></annotations>"""
        m = masks_from_xml(xml, "a.jpg", 6, 4)
        expect = np.zeros((4, 6), bool); expect[1, 2:4] = True; expect[2, 1:3] = True
        self.assertTrue((m["dead_posidonia"] == expect).all())
        self.assertFalse((m["posidonia"] & m["dead_posidonia"]).any())

    def test_decode_rle(self):
        self.assertEqual(decode_rle("0,3,1", 2, 2).tolist(), [[True, True], [True, False]])

    def test_unknown_label_is_refused(self):
        xml = '<annotations><image id="0" name="a.jpg" width="4" height="4"><polygon label="seagrass" points="0,0;3,0;3,3"/></image></annotations>'
        with self.assertRaises(ValueError):
            masks_from_xml(xml, "a.jpg", 4, 4)


if __name__ == "__main__":
    unittest.main()
