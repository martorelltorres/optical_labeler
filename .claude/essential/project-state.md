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

- **Cinco clases y etiquetas parciales implementadas** (`plan_clases_ampliadas.md`): Posidonia, mata muerta,
  roca, arena, otro; `annotated` por sitio; `load_optical(mosaic, masks, …)`; `optical_loss` con clases permitidas
  para el fondo sin clase; `class_iou` por clase anotada; `auto_label.py` lee también los modelos antiguos de 3
  clases (`LEGACY_CODES`). 24 tests en verde; prueba en CPU de `train.py` y `auto_label.py` (modelo nuevo y
  antiguo) correcta. **Sin modelo de 5 clases entrenado todavía** (faltan ejemplos de arena y mata).
- README con sección de resultados y dos figuras (`docs/images/`, script `scratch/readme_figures.py`).
- Historial reescrito sin las líneas `Co-Authored-By` (petición del usuario); `origin/main` aún tiene los 2
  commits antiguos: hace falta `git push --force-with-lease` del usuario.
- Revisión movida a `runs/review/`; capa vacía `runs/review/andratx_13_52_22/andratx_13_52_22_extra_labels.geojson`.

- **Recorte de 2025 `09_19_56` revisado por el usuario** (15:24) y convertido: `runs/review_masks/optical_2025_04_09/`
  (factor 8 sobre 3,3 mm → 2,6 cm, con zona anotada). Posidonia 43,6 % → 41,9 % (IoU 0,96 con la propuesta), roca
  0,9 % sin cambios. La propuesta venía de `scratch/propose_crops.py` (modelo con color igualado ∪ fondo oscuro), no
  del modelo solo.

- Polígonos revisados → máscaras (`optical_labeler/vectors.py`, `masks_from_vectors.py --area`): clase
  `ignore`, zona anotada, prioridad ignore > roca > Posidonia; `load_optical` y `configs/sites.yaml` aceptan
  `ignore_mask` y `area_mask`. 19 tests en verde; en el recorte revisado `16_13_30`, 48,8 % de Posidonia.
  Commiteado.

- Separación del repositorio; modelos y propuestas movidos a `runs/`.
- Análisis OOF frente a la manual en Andratx (`context/analisis_oof_andratx.md`): la PO es casi
  perfecta y los errores están en los bordes; el punto débil es roca ↔ resto.

## Decisiones en curso (revisables)

- El usuario indica que en el recorte de 2025 hay fondo oscuro que es mata muerta, hoy dentro de los
  polígonos de Posidonia: hay que redibujarlo como `dead_posidonia` antes de usar ese recorte.
- Hipótesis por comprobar con el `d1prod`: parte del fallo en Es Trenc («el color difiere») viene de la
  falta de corrección fotométrica (Andratx, el entrenamiento, es `d1prod`).

- Los modelos `transfer` actuales se guardaron sin `res_m`: `auto_label.py` necesita
  `--train-res-m 0.0263`.

## Problemas latentes

- `origin/main` conserva 2 commits con co-autoría de Claude hasta el push forzado del usuario.

## Siguiente acción

El usuario anota arena y mata muerta: recorte de 2025 (`runs/review/optical_2025_04_09/`, redibujar la mata como
`dead_posidonia`) y unos polígonos en Andratx (`runs/review/andratx_13_52_22/`, instrucciones en `LLEGEIX.txt`).
Después: convertir, declarar en `configs/sites.yaml` y entrenar el modelo de 5 clases (tarea 10). Es Trenc sigue
en pausa hasta el `mosaic_d1prod` de Haddock (semana del 2026-11-23); lo ya revisado (`16_13_30`, parte de
`14_57_04`) se reutiliza si encaja.
