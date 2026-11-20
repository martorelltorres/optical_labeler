# optical_labeler

Propuesta automática de etiquetas **Posidonia / roca / resto** sobre fotomosaicos ópticos
georreferenciados (cámara del AUV). Una U-Net sobre RGB propone las máscaras. Una persona las
revisa en un SIG y el resultado son máscaras con el mismo formato que las dibujadas a mano, que
se pueden usar donde se usen las manuales; por ejemplo, para etiquetar mosaicos de side-scan
sonar co-registrados.

## Flujo

1. **Entrenar** con fotomosaicos que tengan máscara manual (`configs/sites.yaml`).
2. **Proponer** etiquetas sobre un fotomosaico nuevo: máscaras, probabilidades, un mapa de duda y
   polígonos GeoJSON.
3. **Revisar** los polígonos en un SIG (QGIS) y corregir la clase o la forma.
4. **Convertir** los polígonos revisados en máscaras.

## Datos

Los datos no se versionan. `docker-compose.yml` monta el directorio `OPTICAL_DATA` (por defecto
`../sss_segmentation/bagfiles`) en solo lectura en `/workspace/data`, y las rutas de
`configs/sites.yaml` son relativas a `/workspace`. Ningún script escribe en `data/`.

Formato de entrada:

- **Fotomosaico**: GeoTIFF RGB o RGBA. La banda alfa marca los píxeles válidos (sin alfa, son los
  píxeles no negros).
- **Máscaras manuales**: PNG en escala de grises, 255 = clase, dibujadas sobre una copia del
  mosaico reducida por cualquier factor (se remuestrean a la rejilla de trabajo). Una por clase
  (`posidonia_mask`, `rock_mask`); `rock_mask: null` si la roca no está anotada (cuenta como resto).

La resolución de trabajo es la del mosaico reducida por `factor` (con mosaicos x4, ~2,6 cm/px).

## Puesta en marcha

```bash
docker compose build                                   # imagen CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build   # imagen CUDA
```

Con GPU, `OPTICAL_GPU` elige la tarjeta por su índice de `nvidia-smi`:

```bash
OPTICAL_GPU=1 docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm labeler <orden>
```

## Uso

```bash
# Validación cruzada espacial en un sitio (bloques de configs/sites.yaml)
docker compose run --rm labeler python scripts/train.py --protocol cv --site andratx_135222

# Entrenamiento con todo un sitio y test en otros
docker compose run --rm labeler python scripts/train.py --protocol transfer \
    --site andratx_135222 --test pollenca_134616

# Propuesta de etiquetas sobre un fotomosaico (media de los modelos dados)
docker compose run --rm labeler python scripts/auto_label.py \
    --models 'runs/optical_r34/transfer/all/seed*/model.pt' --mosaic <fotomosaico.tif> --out <prefijo>

# Polígonos revisados → máscaras
docker compose run --rm labeler python scripts/masks_from_vectors.py --vectors <prefijo>_labels.geojson \
    --mosaic <fotomosaico.tif> --out <prefijo_revisado>
```

`train.py` escribe en `runs/<modelo>/<protocolo>/<partición>/seed<k>/` el modelo (`model.pt`), las
métricas (`metrics.json`, recuentos de IoU por clase y sitio de test) y las probabilidades por clase
sobre el mosaico de test (`prob_<sitio>.tif`).

`auto_label.py` reduce el mosaico a la resolución con la que se entrenaron los modelos. Para
modelos guardados sin esa resolución hay que darla con `--train-res-m`. Escribe:

| fichero | contenido |
|---|---|
| `<prefijo>_posidonia.png`, `<prefijo>_rock.png` | máscaras con el formato de las manuales |
| `<prefijo>_prob.tif` | probabilidades por clase (uint8 0..250; bandas resto, Posidonia, roca) |
| `<prefijo>_doubt.tif` | banda 1: 1 − probabilidad máxima; banda 2: desviación de P(Posidonia) entre modelos |
| `<prefijo>_labels.geojson` | polígonos de Posidonia y roca (CRS del mosaico) con propiedad `class`, probabilidad y duda medias |

Al revisar en el SIG, cada polígono conserva la propiedad `class` (`posidonia` o `rock`);
`masks_from_vectors.py` solo lee esa propiedad. Los polígonos tienen que estar en el CRS del
mosaico.

## Tests

```bash
docker compose run --rm labeler          # ejecuta python -m unittest discover
```

## Licencias

Dependencias en `requirements.txt`, con la versión fijada y la licencia anotada; torch y
torchvision se instalan en el `Dockerfile`. Ninguna es GPL.
