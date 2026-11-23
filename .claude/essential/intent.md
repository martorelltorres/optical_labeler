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

- **Clases** (decisión del usuario, 2026-11-20): Posidonia, Posidonia muerta (mata), roca, arena y otro
  fondo, más `ignore` en la revisión (blancos artificiales). En polígonos revisados la prioridad es
  ignore > roca > mata > arena > otro > Posidonia.
- **Etiquetas parciales**: cada sitio declara las clases anotadas completas (`annotated`); un píxel anotado sin
  clase solo se sabe que es una de las demás (pérdida marginal, como el `ce_dice` de `sss_segmentation`).
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
| `.claude/plans/plan_etiquetado_optico.md` | plan de la vía 1: fases, criterios A–D, resultados, fase O5 |
| `.claude/plans/plan_clases_ampliadas.md` | plan de las cinco clases y las etiquetas parciales |
| `runs/review/` | recortes de revisión en QGIS (movidos desde `sss_segmentation` el 2026-11-20) |
| `runs/cvat/` | imágenes y sidecars exportados para CVAT (carpeta compartida de CVAT) |
| `.claude/context/analisis_oof_andratx.md` | etiquetas automáticas frente a la manual en Andratx |
| `.claude/history/done.md` | la historia del proyecto, un hito por línea |
| `runs/optical_r34/` | modelos `cv` (3 folds × 3 semillas) y `transfer` (todo Andratx, 3 semillas) |
| `runs/labels/` | propuestas sobre Andratx 12_52_41, Es Trenc y 2025-04-09 |
