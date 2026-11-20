# todo_list — optical_labeler

Plan vigente: `.claude/plans/plan_etiquetado_optico.md` (fase O5).

| # | tarea | prioridad | estado |
|---|---|---|---|
| 1 | Revisión humana de los 4 recortes en QGIS | Alta | BLOCKED (usuario) |
| 2 | `masks_from_vectors`/`load_optical`: zona anotada (`_area.geojson`), clase `ignore`, roca gana sobre Posidonia | Alta | Pending |
| 3 | Reentrenar con Andratx + recortes revisados; evaluar dejando un recorte fuera y en Pollença | Alta | BLOCKED (1, 2) |
| 4 | Re-etiquetar los mosaicos de test de Es Trenc y revisión del usuario (entrada del test SSS de `sss_segmentation`) | Alta | BLOCKED (3) |
| 5 | Mover `sss_segmentation/runs/optical/review/` y `review_crops.zip` a `runs/review/` | Media | BLOCKED (1) |
| 6 | Construir las imágenes `optical_labeler:cpu` y `:cu121` y pasar los tests en ellas | Media | Pending |

## Detalle

**2.** Prometido al usuario: donde roca y Posidonia se solapan, gana la roca. La regla «fondo
oscuro» de las propuestas mete roca oscura en Posidonia; el usuario la dibuja a mano.
