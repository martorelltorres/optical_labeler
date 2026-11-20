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

- Polígonos revisados → máscaras (`optical_labeler/vectors.py`, `masks_from_vectors.py --area`): clase
  `ignore`, zona anotada, prioridad ignore > roca > Posidonia; `load_optical` y `configs/sites.yaml` aceptan
  `ignore_mask` y `area_mask`. 19 tests en verde; en el recorte revisado `16_13_30`, 48,8 % de Posidonia.
  Commiteado.

- Separación del repositorio; modelos y propuestas movidos a `runs/`.
- Análisis OOF frente a la manual en Andratx (`context/analisis_oof_andratx.md`): la PO es casi
  perfecta y los errores están en los bordes; el punto débil es roca ↔ resto.

## Decisiones en curso (revisables)

- Hipótesis por comprobar con el `d1prod`: parte del fallo en Es Trenc («el color difiere») viene de la
  falta de corrección fotométrica (Andratx, el entrenamiento, es `d1prod`).

- Los modelos `transfer` actuales se guardaron sin `res_m`: `auto_label.py` necesita
  `--train-res-m 0.0263`.

## Problemas latentes

- **Los recortes de revisión siguen en `sss_segmentation/runs/optical/review/`** (y
  `review_crops.zip`) porque el usuario los está editando en QGIS desde esa ruta. Se mueven a
  `runs/review/` cuando termine.

## Siguiente acción

**Revisión de los recortes de Es Trenc en pausa**: sus fotomosaicos son de la versión antigua (sin corrección
fotométrica ni homografías refinadas) y se han pedido en `mosaic_d1prod` a Haddock (respuesta la semana del
2026-11-23; `sss_segmentation/.claude/context/productos_haddock.md`). Lo corregido (`16_13_30`, parte de
`14_57_04`) se conserva; hay que comprobar su encaje con el `d1prod` antes de reutilizarlo. El recorte de 2025
(`09_19_56`) no depende del reprocesado y se puede revisar ya.

Fase O5:
1. Con el `d1prod`: re-proponer y re-recortar Es Trenc, reutilizar lo revisado si encaja.
2. Reentrenar con Andratx + recortes; evaluar dejando un recorte fuera y en Pollença.
3. Re-etiquetar los mosaicos de test de Es Trenc para el test SSS de `sss_segmentation`.
