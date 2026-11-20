# optical_labeler

Propuesta de etiquetas Posidonia / roca / resto sobre fotomosaicos ópticos, revisables en un
SIG. Qué es y cómo se usa: `README.md`.

## Dónde está escrito qué

Público: este fichero (cómo se trabaja) y `README.md`. Estado vivo, en `.claude/`:
`essential/intent.md` objetivo, alcance y decisiones permanentes · `essential/project-state.md`
qué es verdad ahora · `essential/todo_list.md` lo abierto · `history/done.md` un hito por línea ·
`plans/` los planes vigentes.

**Fuera de `.claude/` no hay metadatos de desarrollo**, tampoco en los mensajes de commit ni en
los comentarios del código.

El entorno (servidor, directorios compartidos, QGIS) está descrito en el `CLAUDE.md` del
directorio padre.

## Datos

- `data/` es un montaje de solo lectura (`OPTICAL_DATA`). Nunca se escribe ahí. Ojo:
  `gdalinfo -stats` y abrir un raster con GDAL en modo actualización escriben un `.aux.xml`
  junto al fichero.
- Las salidas (modelos, propuestas, recortes de revisión) van en `runs/`, que no se versiona.

## Interfaz con quien consume las etiquetas

El formato de salida es un contrato: máscaras PNG 255 = clase sobre la rejilla del mosaico (a
cualquier reducción) y GeoJSON con la propiedad `class`. `sss_segmentation` las lee como si
fueran manuales. Un cambio de formato rompe a ese consumidor: dilo antes de hacerlo.

## Entorno

- Todo se ejecuta en Docker (`Dockerfile`, `docker-compose.yml`, variante CUDA en
  `docker-compose.gpu.yml`, GPU con `OPTICAL_GPU`).
- Dependencias en `requirements.txt`, con versión fijada y licencia anotada.
- Tests: `docker compose run --rm labeler`.

## Propiedad intelectual y licencias

El código es propiedad de la UIB.

- No entra código GPL/LGPL/AGPL en el paquete.
- No se importa ni copia código cuya licencia no esté verificada.
- Antes de añadir una dependencia, comprueba su licencia y justifícala.

## Reglas de evaluación

- **Partición espacial** (bloques de `configs/sites.yaml`), nunca teselas aleatorias del mismo
  mosaico.
- Métricas: **IoU por clase** frente a la máscara manual, con la banda ignorada alrededor de los
  bordes de Posidonia (`band_m`). La Posidonia es la clase principal; la roca se reporta aparte.
- Un sitio de test no se usa para elegir modelo, umbral ni receta.
- La IoU dentro del sitio de entrenamiento no predice la de un sitio nuevo: la transferencia se
  mide siempre en otro sitio.

## Reglas de trabajo

- Verifica antes de afirmar. Si no has leído un fichero, no describas su contenido.
- Si la petición admite dos interpretaciones razonables, pregunta.
- Si algo parece una mala idea, dilo antes de implementarlo.
- Un commit = un cambio conceptual. Antes de commitear, refresca
  `.claude/essential/project-state.md` y `todo_list.md` e inclúyelos en el mismo commit.
- Agentes en `.claude/agents/`: `code-reviewer`, `test-runner`, `docs-updater`,
  `research-analyst`.
