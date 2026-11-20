# intent — optical_labeler

## Objetivo

Que cualquier misión con cámara tenga etiquetas Posidonia / roca / resto propuestas
automáticamente sobre el fotomosaico óptico, revisables por una persona en QGIS y convertibles
en máscaras con el formato de las manuales.

## Dentro y fuera

**Dentro**: lectura de fotomosaicos y máscaras manuales, entrenamiento y evaluación del
segmentador RGB, propuesta de etiquetas (máscaras, probabilidad, duda, polígonos), conversión de
polígonos revisados a máscaras, preparación de recortes de revisión y reentrenamiento con
etiquetas revisadas.

**Fuera**: llevar las etiquetas a la rejilla del SSS y todo lo que se entrena con ellas
(`sss_segmentation`), generar los fotomosaicos.

## Decisiones permanentes, en vigor

- **Clases**: Posidonia, roca, resto, más `ignore` en la revisión humana (blancos artificiales).
  En polígonos revisados: ignore > roca > Posidonia donde se solapan.
- **Contrato con `sss_segmentation`**: máscaras PNG 255 = clase sobre la rejilla del mosaico y
  GeoJSON con propiedad `class`. `sss_segmentation` las consume como si fueran manuales y no
  importa código de aquí.
- **La propiedad intelectual del código es de la UIB.** Sin código GPL ni de licencia sin verificar.
- **Todo en Docker**, imagen CPU y CUDA. Los datos (`data/`) se montan en solo lectura desde
  `sss_segmentation/bagfiles` (variable `OPTICAL_DATA`).
- **Partición espacial** por bloques de 5 m (k = 3, hueco 0,5 m); banda ignorada de 0,1 m en los
  bordes de Posidonia. Resolución de trabajo 2,6 cm/px (mosaicos x4 reducidos por 4).
- Las utilidades genéricas (`codes`, `tiling`, `split`, `infer`) se copiaron de `sss_segmentation`
  y evolucionan por separado.

## Dónde está cada cosa

| ruta | qué contiene |
|---|---|
| `CLAUDE.md` | cómo se trabaja |
| `.claude/plans/plan_etiquetado_optico.md` | plan vigente: fases, criterios A–D, resultados, fase O5 |
| `.claude/context/analisis_oof_andratx.md` | etiquetas automáticas frente a la manual en Andratx |
| `.claude/history/done.md` | la historia del proyecto, un hito por línea |
| `runs/optical_r34/` | modelos `cv` (3 folds × 3 semillas) y `transfer` (todo Andratx, 3 semillas) |
| `runs/labels/` | propuestas sobre Andratx 12_52_41, Es Trenc y 2025-04-09 |
