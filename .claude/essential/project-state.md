# project-state — optical_labeler

## Estado actual

Paquete `optical_labeler/` con cinco clases (Posidonia, mata muerta, roca, arena, otro), etiquetas parciales,
revisión en QGIS o en **CVAT** (`cvat.py`, `scripts/cvat_export.py`/`cvat_import.py`), 30 tests en verde. Imagen
`optical_labeler:cu121` construida.

**Andratx 13_52_22 etiquetado en CVAT** (tarea 2, export de las 12:47 del 2026-11-23): 94 % de la zona válida;
máscaras en `runs/review_masks/andratx_13_52_22/cvat_20261123_1247/` y copia para `sss_segmentation` en
`runs/labels/andratx_13_52_22_cvat/` (montada allí como `labels/`). **Modelo óptico de 5 clases**
`optical_r34_c5_cvat` (CV en Andratx, 3 folds × 3 semillas): Posidonia 0,965 ± 0,009, roca 0,857, arena 0,953,
otro 0,629, mata muerta 0,317. Sin transfer todavía. Fuera de Andratx los modelos anteriores fallaban (Es Trenc
~60 %, 2025 ~3 % de la pradera).

## Cambios recientes (sesión del 2026-11-23)

- Etiquetado completo de Andratx en CVAT con SAM; tres revisiones de las etiquetas (quedan huecos: ~11 m²
  alrededor de las manchas de mata muerta y una franja de 15 m² en el borde arena–roca del este).
- Conversión CVAT ↔ máscaras (`cvat.py`; los contornos cerrados se rellenan; prioridad ignore > roca > mata >
  arena > otro > Posidonia). Configs del entrenamiento en `runs/review_masks/andratx_13_52_22/`
  (`sites_andratx_cvat.yaml`, `unet_c5_cvat.yaml`); lanzador `scratch/run_optical_c5_cvat.sh`.
- `sss_segmentation` consume estas máscaras (sus resultados, en su propio `.claude/`).
- `origin/main` ya está sincronizado con la historia reescrita.

## Decisiones en curso (revisables)

- La mata muerta no se separa bien ni en el óptico (0,32): revisar la definición de la clase antes de invertir en
  ella.
- Hipótesis por comprobar con el `d1prod`: parte del fallo en Es Trenc viene de la falta de corrección fotométrica.
- Los modelos `transfer` antiguos se guardaron sin `res_m`: `auto_label.py` necesita `--train-res-m 0.0263`.

## Problemas latentes

- El recorte de 2025 tiene mata muerta dentro de los polígonos de Posidonia (tarea 8).

## Siguiente acción

Rellenar los huecos de Andratx en CVAT (usuario). Después: transfer del modelo de 5 clases a Pollença y al recorte
de 2025 (tarea 10). Es Trenc sigue en pausa hasta el `mosaic_d1prod` de Haddock.
