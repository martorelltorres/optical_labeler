# todo_list — optical_labeler

Planes vigentes: `.claude/plans/plan_clases_ampliadas.md` (cinco clases) y `.claude/plans/plan_etiquetado_optico.md` (fase O5).

| # | tarea | prioridad | estado |
|---|---|---|---|
| 1b | Es Trenc: re-proponer y re-recortar sobre `mosaic_d1prod`; comprobar el encaje de lo ya revisado (`16_13_30`, `14_57_04`) | Alta | BLOCKED (`d1prod` de Haddock) |
| 4 | Re-etiquetar los mosaicos de test de Es Trenc y revisión del usuario (entrada del test SSS de `sss_segmentation`) | Alta | BLOCKED (1b, 10) |
| 6 | Pasar los tests en la imagen `optical_labeler:cu121` (la CPU ya los pasa) | Baja | Pending |
| 8 | Usuario, en CVAT (tarea 1): pintar la mata muerta como `dead_posidonia` y añadir `sand` en el recorte de 2025; decir si toda la arena quedó pintada (`annotated`) | Alta | BLOCKED (usuario) |
| 9 | Usuario, en CVAT (tarea 2): rellenar los huecos de Andratx (alrededor de la mata muerta, franja este arena–roca) | Media | BLOCKED (usuario) |
| 10 | Transfer del modelo de 5 clases (`optical_r34_c5_cvat`, CV en Andratx hecha) a Pollença y al recorte de 2025 | Alta | Pending |

## Detalle
