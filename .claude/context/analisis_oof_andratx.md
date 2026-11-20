# Etiquetas automáticas fuera de fold frente a la máscara manual — Andratx 13_52_22 (2026-11-20)

Script: `/home/azken/antoni_martorell/scratch/compare_oof_manual.py` (entrada
`sss_segmentation/runs/optical/oof/andratx_seed{0,1,2}_*.png`, compuestas por `scratch/compose_oof.py`
desde los modelos `cv`). 1317 m² anotados, 2,63 cm/px, 3 semillas.

| clase | IoU sin banda | IoU banda 0,1 m | por fold |
|---|---|---|---|
| Posidonia | 0,95 | 0,98 | 0,974–0,980 |
| roca | 0,70 | 0,70 | 0,66–0,76 |
| resto | 0,88 | 0,89 | 0,87–0,92 |

Confusión (semilla 0, % de la fila manual): resto 93,8 / PO 1,8 / roca 4,4; PO 1,4 / 97,7 / 0,9;
roca 19,0 / 0,1 / 80,9. Cobertura auto/manual: PO 1,009, roca 0,953.

- El 73 % de los errores está a <25 cm de un borde manual; a >1 m, tasa de error 0,2 %. La PO
  perdida está casi toda en el borde (2,6 % a ≥25 cm).
- Error dominante roca ↔ resto (37 m² de roca no detectada, 26 m² de roca de más), un tercio
  lejos de bordes. Roca >5 m² detectada al 100 %, <1 m² al 62 %. Sin revisión visual no se
  separa error del modelo de ambigüedad del trazo manual.
- Acuerdo entre semillas en el 95 % de los píxeles; error 47 % donde discrepan y 5 % donde
  coinciden: el desacuerdo (`_doubt.tif`) localiza los errores.
- Es el mismo dominio que el entrenamiento: cota superior, no estimación para sitios nuevos
  (Es Trenc ~60 %, 2025 ~3 % de la pradera detectada).
