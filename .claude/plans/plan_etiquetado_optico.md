# Plan — etiquetado automático desde el mosaico óptico (vía 1), 2026-11-20

> Código en `optical_labeler` desde el 2026-11-20 (antes `sss_seg/optical.py` y `scripts/*optical*.py` de
> `sss_segmentation`): `optical_labeler/optical.py`, `scripts/train.py`, `scripts/auto_label.py`,
> `scripts/masks_from_vectors.py`. Modelos en `runs/optical_r34/`, propuestas en `runs/labels/`. El criterio C
> se entrena y evalúa en `sss_segmentation` (`runs/optical/oof/`, `runs/optical_c/`).

## Objetivo

Que cualquier misión con cámara tenga etiquetas Posidonia / roca / resto propuestas automáticamente sobre el
mosaico óptico, revisables por un humano en QGIS y convertibles a la rejilla del SSS con el código actual
(`labels.build_labels`, `prepare_site.py` de `sss_segmentation`).

Decisiones del usuario (2026-11-20): clases Posidonia, roca, resto; se pueden usar las **fotos** de Pollença
para validar (su SSS sigue fuera).

## Datos

- Andratx 13_52_22: mosaico RGBA `mosaic_d1prod/mosaic_x4/mosaic_x4.tif` (6,6 mm/px, 9143 × 7113, franja de
  ~45 × 60 m) y máscaras manuales `posidonia_/rock_manual_r2_mosaic_x4.png` (mitad de resolución).
- Pollença 15_46_16: mismo formato, solo máscara de Posidonia.
- Andratx 12_52_41 (`download_.../12_52_41/mosaic/mosaic_x4/`): segundo mosaico de la **misma franja**, sin
  etiquetar. Ningún mosaico óptico cubre la mitad derecha del SSS de Andratx.

## Fases

O1. **Datos ópticos** (`sss_seg/optical.py`): lee el mosaico a una resolución de trabajo fija de 2,6 cm/px
    (factor 4 sobre el x4), la validez (alfa) y los códigos 1 PO, 2 roca, 0 resto (255 fuera); folds
    espaciales con `split.block_folds` (bloques de 5 m, k = 3, hueco 0,5 m).
O2. **Modelo**: U-Net resnet34 ImageNet, 3 canales RGB, 3 clases, CE enmascarada + Dice de PO; muestreo de
    teselas reutilizando `CenterSampler`, `augment.scaled_crop` y `augment.geometric`, más jitter de color.
    Script `scripts/train_optical.py` (protocolos: CV en Andratx; Andratx completo → Pollença).
O3. **Pipeline de etiquetado** `scripts/auto_label_optical.py`: mosaico óptico + modelo → máscaras PNG con el
    formato de las manuales (las acepta `prepare_site.py` sin cambios), GeoTIFF de probabilidad y de duda
    (desacuerdo entre semillas) y polígonos GeoJSON para editar en QGIS. Script inverso
    `scripts/masks_from_vectors.py`: polígonos revisados → máscaras PNG.
O4. **Evaluación** y aplicación a 12_52_41 (consistencia entre misiones, descriptiva).

## Criterios (fijados antes de ver resultados; 3 semillas)

- **A — CV en Andratx**: IoU de Posidonia ≥0,85 y de roca ≥0,70 frente a la máscara manual.
- **B — Pollença** (sin ver Pollença al entrenar): IoU de Posidonia ≥0,80.
- **C — utilidad para el SSS**: U-Net SSS P1 (`unet_r34`, rejilla 0,20 m) entrenada con las etiquetas
  automáticas fuera de fold de Andratx (mismos folds SSS que la manual) y evaluada contra la manual: IoU ≥
  0,637 − 2 puntos.
- D — 12_52_41: acuerdo con 13_52_22 tras co-registro, descriptivo.

Los umbrales de A y B son juicio a priori (la Posidonia es muy distinguible en RGB); C es el que importa.

## Riesgos

- Las máscaras manuales son de grano grueso en los bordes: la IoU tiene techo por debajo de 1. Banda
  ignorada de 0,1 m en los bordes al evaluar.
- Iluminación y turbidez distintas entre sitios (Pollença): jitter de color y normalización por imagen.
- Licencias: sin dependencias nuevas (GeoJSON con `json` y `rasterio.features`).

## Resultados (2026-11-20)

- A: CV Andratx PO 0,977 ± 0,004, roca 0,702 ± 0,005 → cumple. B: Pollença PO 0,913 ± 0,032 → cumple.
- C: SSS P1 con etiquetas automáticas fuera de fold, evaluado contra la manual con el mismo script
  (`scratch/eval_c.py`, folds agregados, banda 0,5 m): 0,636 ± 0,029 frente a 0,675 ± 0,031 de la manual.
  Cumple el umbral literal (≥0,617) pero pierde 3,9 puntos frente a la manual medida igual: el espíritu del
  criterio (≤2 puntos de pérdida) **no** se cumple. Fuga indirecta: los modelos ópticos de cada fold óptico
  vieron etiquetas manuales de zonas que son test SSS (solo apariencia óptica).
- **Fuera de Andratx el segmentador falla** (revisión visual): Es Trenc ~60 % de una pradera casi continua,
  2025 ~3 % de las manchas de pradera. Causa: las etiquetas de Andratx solo contienen pradera densa y
  continua, y el color difiere. Igualar media/desviación por canal ayuda poco (59→68 %, 3→5 %).
  El test SSS de Es Trenc (M1–M4, `sss_segmentation/.claude/plans/plan_test_estrenc.md`) queda **parado** hasta tener etiquetas revisadas.

## O5 — revisión humana y reentrenamiento (en curso)

Criterio del usuario: verde Posidonia, naranja roca, el resto no Posidonia. Cuatro recortes propuestos
(`scratch/propose_crops.py`: Posidonia del modelo con color igualado ∪ fondo oscuro por Otsu en el recorte,
menos roca del modelo) en `runs/optical/review/` (zip `runs/optical/review_crops.zip`): Es Trenc 15_51_24,
14_57_04 y 16_13_30 (13 m), 2025 09_19_56. El usuario los corrige en QGIS (clase `ignore` para blancos tipo
mina). Después: soporte de zona anotada y clase `ignore` en `masks_from_vectors`/`load_optical`, reentreno
con Andratx + recortes (dejando un recorte fuera por turno + Pollença para evaluar), re-etiquetado de los
mosaicos de test de Es Trenc y revisión del usuario antes del test SSS.
