# optical_labeler

Automatic proposal of **Posidonia / rock / other** labels on georeferenced optical mosaics (AUV
camera). A U-Net on RGB proposes the masks. A person reviews them in a GIS, and the result is a set
of masks in the same format as hand-drawn ones, usable wherever manual masks are used; for example,
to label co-registered side-scan sonar mosaics.

## Workflow

1. **Train** on optical mosaics that have manual masks (`configs/sites.yaml`).
2. **Propose** labels on a new optical mosaic: masks, probabilities, a doubt map and GeoJSON
   polygons.
3. **Review** the polygons in a GIS (QGIS), correcting their class or shape.
4. **Convert** the reviewed polygons back to masks.

## Data

Data are not versioned. `docker-compose.yml` mounts the `OPTICAL_DATA` directory (by default
`../sss_segmentation/bagfiles`) read-only at `/workspace/data`, and the paths in
`configs/sites.yaml` are relative to `/workspace`. No script writes to `data/`.

Input format:

- **Optical mosaic**: RGB or RGBA GeoTIFF. The alpha band marks the valid pixels (without alpha,
  the non-black pixels are valid).
- **Manual masks**: grayscale PNG, 255 = class, drawn on a copy of the mosaic downscaled by any
  factor (they are resampled to the working grid). One per class (`posidonia_mask`,
  `rock_mask`); `rock_mask: null` if rock is not annotated (it then counts as other).

The working resolution is the mosaic resolution reduced by `factor` (~2.6 cm/px with x4 mosaics).

## Setup

```bash
docker compose build                                                   # CPU image
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build   # CUDA image
```

With a GPU, `OPTICAL_GPU` selects the card by its `nvidia-smi` index:

```bash
OPTICAL_GPU=1 docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm labeler <command>
```

## Usage

```bash
# Spatial cross-validation on one site (blocks from configs/sites.yaml)
docker compose run --rm labeler python scripts/train.py --protocol cv --site andratx_135222

# Train on a whole site and test on others
docker compose run --rm labeler python scripts/train.py --protocol transfer \
    --site andratx_135222 --test pollenca_134616

# Label proposal on an optical mosaic (mean of the given models)
docker compose run --rm labeler python scripts/auto_label.py \
    --models 'runs/optical_r34/transfer/all/seed*/model.pt' --mosaic <mosaic.tif> --out <prefix>

# Reviewed polygons → masks (optionally clipped to the reviewed area)
docker compose run --rm labeler python scripts/masks_from_vectors.py --vectors <prefix>_labels.geojson \
    --mosaic <mosaic.tif> --out <reviewed_prefix> [--area <area.geojson>]
```

`train.py` writes to `runs/<model>/<protocol>/<split>/seed<k>/` the model (`model.pt`), the metrics
(`metrics.json`, IoU counts per class and test site) and the class probabilities over the test
mosaic (`prob_<site>.tif`).

`auto_label.py` reduces the mosaic to the resolution the models were trained at. Models saved
without that resolution need it through `--train-res-m`. It writes:

| file | content |
|---|---|
| `<prefix>_posidonia.png`, `<prefix>_rock.png` | masks in the format of the manual ones |
| `<prefix>_prob.tif` | class probabilities (uint8 0..250; bands other, Posidonia, rock) |
| `<prefix>_doubt.tif` | band 1: 1 − highest probability; band 2: standard deviation of P(Posidonia) between models |
| `<prefix>_labels.geojson` | Posidonia and rock polygons (mosaic CRS) with a `class` property and their mean probability and doubt |

When reviewing in the GIS, every polygon keeps its `class` property: `posidonia`, `rock`, or
`ignore` for objects that must count neither as Posidonia nor as background (man-made targets, for
example). `masks_from_vectors.py` only reads that property and refuses any other value. Where
polygons overlap, `ignore` wins over `rock` and `rock` over `posidonia`. It writes
`<reviewed_prefix>_posidonia.png`, `_rock.png` and `_ignore.png`; with `--area` (a GeoJSON of the
reviewed area) every mask is clipped to that area and `_area.png` is written too. The polygons must
be in the CRS of the mosaic.

In `configs/sites.yaml` a site may then add `ignore_mask` (pixels left out of training and metrics)
and `area_mask` (outside it nothing is labeled) next to `posidonia_mask` and `rock_mask`.

## Tests

```bash
docker compose run --rm labeler          # runs python -m unittest discover
```

## Licenses

Dependencies are listed in `requirements.txt` with pinned versions and their licenses; torch and
torchvision are installed in the `Dockerfile`. None is GPL.
