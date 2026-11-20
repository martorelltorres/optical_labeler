# todo_list — optical_labeler

Plan vigente: `.claude/plans/plan_etiquetado_optico.md` (fase O5).

| # | tarea | prioridad | estado |
|---|---|---|---|
| 1 | Revisión humana del recorte de 2025 `09_19_56` en QGIS | Alta | BLOCKED (usuario) |
| 1b | Es Trenc: re-proponer y re-recortar sobre `mosaic_d1prod`; comprobar el encaje de lo ya revisado (`16_13_30`, `14_57_04`) | Alta | BLOCKED (`d1prod` de Haddock) |
| 3 | Reentrenar con Andratx + recortes revisados; evaluar dejando un recorte fuera y en Pollença | Alta | BLOCKED (1, 1b) |
| 4 | Re-etiquetar los mosaicos de test de Es Trenc y revisión del usuario (entrada del test SSS de `sss_segmentation`) | Alta | BLOCKED (3) |
| 5 | Mover `sss_segmentation/runs/optical/review/` y `review_crops.zip` a `runs/review/` | Media | Pending (Es Trenc en pausa: se puede mover ya, avisando al usuario) |
| 6 | Construir las imágenes `optical_labeler:cpu` y `:cu121` y pasar los tests en ellas | Media | Pending |

## Detalle
