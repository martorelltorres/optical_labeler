# todo_list — optical_labeler

Planes vigentes: `.claude/plans/plan_clases_ampliadas.md` (cinco clases) y `.claude/plans/plan_etiquetado_optico.md` (fase O5).

| # | tarea | prioridad | estado |
|---|---|---|---|
| 1b | Es Trenc: re-proponer y re-recortar sobre `mosaic_d1prod`; comprobar el encaje de lo ya revisado (`16_13_30`, `14_57_04`) | Alta | BLOCKED (`d1prod` de Haddock) |
| 4 | Re-etiquetar los mosaicos de test de Es Trenc y revisión del usuario (entrada del test SSS de `sss_segmentation`) | Alta | BLOCKED (1b, 10) |
| 6 | Construir las imágenes `optical_labeler:cpu` y `:cu121` y pasar los tests en ellas | Media | Pending |
| 8 | Usuario: en el recorte de 2025, redibujar la mata muerta como `dead_posidonia` y añadir `sand`; decir si toda la arena quedó dibujada (`annotated`) | Alta | BLOCKED (usuario) |
| 9 | Usuario: polígonos de `sand` y `dead_posidonia` (y `other`) en Andratx 13_52_22 (`runs/review/andratx_13_52_22/`) | Alta | BLOCKED (usuario) |
| 10 | Convertir 8 y 9 (`masks_from_vectors`), añadir a `configs/sites.yaml` y entrenar el modelo de 5 clases: CV en Andratx (criterio: IoU de Posidonia ≥ modelo de 3 clases − 1 punto) + transfer | Alta | BLOCKED (8, 9) |

## Detalle
