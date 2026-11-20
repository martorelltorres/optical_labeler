# project-state — optical_labeler

## Estado actual

Código separado de `sss_segmentation` el 2026-11-20: paquete `optical_labeler/` (`optical`,
`codes`, `tiling`, `split`, `infer`), scripts `train.py`, `auto_label.py`, `masks_from_vectors.py`,
15 tests en verde (ejecutados en la imagen `sss_segmentation:cpu`; la imagen propia
`optical_labeler:cpu/cu121` aún no se ha construido). Re-etiquetar `optical_10_06_31` con el código
movido da máscaras idénticas y probabilidades idénticas a las anteriores. Primer commit en `main`.

Resultados del plan (`plans/plan_etiquetado_optico.md`): A y B cumplen (Andratx PO 0,977 / roca
0,702; Pollença PO 0,913); C pierde 3,9 puntos frente a la manual (dentro de ~1 desviación entre
semillas). **Fuera de Andratx falla** (Es Trenc ~60 %, 2025 ~3 % de la pradera): las etiquetas de
Andratx solo enseñan pradera densa.

## Cambios recientes (sesión del 2026-11-20)

- Separación del repositorio; modelos y propuestas movidos a `runs/`.
- Análisis OOF frente a la manual en Andratx (`context/analisis_oof_andratx.md`): la PO es casi
  perfecta y los errores están en los bordes; el punto débil es roca ↔ resto.

## Decisiones en curso (revisables)

- Los modelos `transfer` actuales se guardaron sin `res_m`: `auto_label.py` necesita
  `--train-res-m 0.0263`.

## Problemas latentes

- **Los recortes de revisión siguen en `sss_segmentation/runs/optical/review/`** (y
  `review_crops.zip`) porque el usuario los está editando en QGIS desde esa ruta. Se mueven a
  `runs/review/` cuando termine.
- `masks_from_vectors.py` hace ganar la Posidonia sobre la roca: contradice la decisión de
  intent (gana la roca). No soporta todavía la zona anotada (`_area.geojson`) ni la clase `ignore`.

## Siguiente acción

El usuario termina de corregir en QGIS los 4 recortes (`16_13_30` guardado a las 13:13;
`14_57_04` abierto; `15_51_24` y 2025 `09_19_56` sin tocar). Después, fase O5 del plan:
1. `masks_from_vectors`/`load_optical`: zona anotada, clase `ignore`, roca gana sobre Posidonia.
2. Reentrenar con Andratx + recortes; evaluar dejando un recorte fuera y en Pollença.
3. Re-etiquetar los mosaicos de test de Es Trenc para el test SSS de `sss_segmentation`.
