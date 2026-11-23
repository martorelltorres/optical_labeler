"""Exchange with the CVAT annotation tool ("CVAT for images 1.1" XML).

An optical mosaic is exported as a downscaled JPEG plus a JSON sidecar holding its grid, so that
annotations drawn on the image (polygons or brush masks) convert back to class masks on that
downscaled copy of the mosaic, the format optical_labeler reads as manual masks.
"""
import json
import xml.etree.ElementTree as ET

import numpy as np
from affine import Affine
from PIL import Image, ImageDraw

from optical_labeler.vectors import CLASSES, PRIORITY

COLORS = {"posidonia": "#22c55e", "dead_posidonia": "#8b5a2b", "rock": "#f97316", "sand": "#facc15",
          "other": "#a855f7", "ignore": "#9ca3af"}


def write_sidecar(path, mosaic, factor, width, height, transform, crs, area=None):
    """mosaic and area: paths relative to the package root."""
    json.dump({"mosaic": mosaic, "factor": factor, "width": width, "height": height,
               "transform": list(transform)[:6], "crs": str(crs), "area": area}, open(path, "w"), indent=1)


def read_sidecar(path):
    s = json.load(open(path))
    s["transform"] = Affine(*s["transform"])
    return s


def encode_rle(mask):
    """CVAT brush mask RLE of a bool array: alternating run lengths, background first, row-major."""
    flat = mask.ravel().astype(np.int8)
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(flat)) + 1, [flat.size]])
    runs = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        runs = [0] + runs
    return ",".join(str(r) for r in runs)


def masks_xml(masks, image_name, width, height):
    """CVAT for images 1.1 XML with one brush mask per non-empty class of {class: bool (H, W)}."""
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    img = ET.SubElement(root, "image", id="0", name=image_name, width=str(width), height=str(height))
    for cls in CLASSES:
        m = masks.get(cls)
        if m is None or not m.any():
            continue
        rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        ET.SubElement(img, "mask", label=cls, occluded="0", source="auto", z_order="0",
                      rle=encode_rle(m[r0:r1, c0:c1]), left=str(c0), top=str(r0),
                      width=str(c1 - c0), height=str(r1 - r0))
    return ET.tostring(root, encoding="unicode")


def decode_rle(rle, width, height):
    """CVAT brush mask RLE (alternating runs, starting with background, row-major) to bool."""
    runs = [int(v) for v in rle.split(",") if v.strip()]
    flat = np.zeros(width * height, bool)
    pos, val = 0, False
    for n in runs:
        if val:
            flat[pos:pos + n] = True
        pos += n
        val = not val
    return flat.reshape(height, width)


def masks_from_xml(xml_text, image_name, width, height):
    """{class: bool (height, width)} from the polygons and masks of one image of a CVAT XML,
    made disjoint with the priority of optical_labeler.vectors. Raises ValueError for labels
    that are not classes."""
    root = ET.fromstring(xml_text)
    img = next((i for i in root.iter("image") if i.get("name") == image_name), None)
    if img is None:
        raise ValueError(f"image {image_name} not in the annotations")
    layers = {c: Image.new("1", (width, height), 0) for c in CLASSES}
    bad = set()
    for el in img:
        cls = el.get("label")
        if el.tag not in ("polygon", "mask", "box"):
            continue
        if cls not in CLASSES:
            bad.add(cls)
            continue
        if el.tag == "polygon":
            pts = [tuple(map(float, p.split(","))) for p in el.get("points").split(";")]
            ImageDraw.Draw(layers[cls]).polygon(pts, fill=1)
        elif el.tag == "box":
            ImageDraw.Draw(layers[cls]).rectangle([float(el.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")], fill=1)
        else:
            left, top = int(float(el.get("left"))), int(float(el.get("top")))
            w, h = int(float(el.get("width"))), int(float(el.get("height")))
            m = Image.fromarray(decode_rle(el.get("rle"), w, h).astype(np.uint8) * 255).convert("1")
            layers[cls].paste(1, (left, top), m)
    if bad:
        raise ValueError(f"labels that are not classes: {sorted(bad)}; expected {CLASSES}")
    masks = {c: np.array(layers[c], bool) for c in CLASSES}
    for i, low in enumerate(PRIORITY):
        for high in PRIORITY[i + 1:]:
            masks[low] &= ~masks[high]
    return masks
