# Plan — etiquetador con clases Posidonia, mata muerta, roca, arena (2026-11-20)

Petición del usuario: el etiquetador debe reconocer al menos roca, arena, Posidonia y Posidonia muerta (mata).
Hoy predice resto / Posidonia / roca.

## Problema de datos

Ninguna fuente tiene las cuatro clases anotadas:
- Andratx 13_52_22: Posidonia y roca; el resto (52 %) es «no Posidonia ni roca», sin distinguir arena de mata muerta
  u otro fondo.
- Pollença: solo Posidonia; el resto puede ser roca, arena o mata.
- Recortes revisados: lo que el usuario dibuje (2025 `09_19_56`; Es Trenc cuando llegue el `d1prod`).

## Diseño propuesto

- **Clases de salida**: Posidonia, mata muerta, roca, arena y «otro» (fango, grava, algas, lo que no sea ninguna de las
  cuatro). Cinco salidas softmax.
- **Etiquetas parciales por sitio** (generaliza el `ce_dice` marginal de `sss_segmentation`): cada sitio declara qué
  clases tiene anotadas (`annotated: [posidonia, rock]` en Andratx). Un píxel anotado sin clase («resto») aporta
  −log Σ p(clases NO anotadas en ese sitio): en Andratx empuja hacia mata/arena/otro sin decir cuál; en Pollença,
  hacia todo menos Posidonia. Así Andratx y Pollença siguen sirviendo sin redibujarlos.
- **Revisión en QGIS**: `class` ∈ {posidonia, dead_posidonia, rock, sand, other, ignore}; el resto del área anotada
  cuenta como «sin clase» con las clases declaradas del recorte.
- **Métricas**: IoU por clase solo donde esa clase está anotada; IoU de Posidonia frente a la manual sigue siendo la
  comparación con el modelo actual (criterio: no perder más de 1 punto en Andratx out of fold).
- **Contrato con `sss_segmentation`**: se añaden `_sand.png` y `_dead_posidonia.png`; `_posidonia.png` y `_rock.png`
  no cambian de significado.

## Lo que hace falta del usuario

Arena y mata muerta solo se aprenden si hay ejemplos dibujados: en el recorte de 2025 (dibujar `sand` y
`dead_posidonia`) y, si se puede, unas pocas zonas de arena y mata en Andratx 13_52_22 (bastaría un polígono por zona
típica, no el mosaico entero).

## Pasos

1. Código: clases y códigos (`codes.py`), pérdida con etiquetas parciales por sitio, `annotated` en `sites.yaml`,
   `sand`/`other` en `vectors.py`, salidas nuevas en `auto_label.py`, métricas por clase anotada. Tests.
2. Revisión del usuario (arena y mata muerta).
3. Reentreno: CV en Andratx (comparar IoU de Posidonia con el modelo de 3 clases) + transfer a Pollença y a los
   recortes, dejando uno fuera.
