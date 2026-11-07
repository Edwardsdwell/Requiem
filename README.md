# Plataforma unificada de aprendizaje no supervisado
Una aplicación Streamlit que analiza el dataset PostgreSQL `aves_db` en **audio, imágenes, texto y coordenadas**
con las **12 opciones algorítmicas** y las **13 comparativas** , más dos comparativas nuevas
(varios algoritmos / varios valores de un parámetro). 

## Puesta en marcha

**Windows** (PowerShell o CMD, dentro de la carpeta del proyecto):
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python arrancar.py
```

También puedes iniciar con `streamlit run app.py`. No ejecutes `app.py` directamente con Python: necesita el servidor de Streamlit. Si ya existe `.venv`, instala o actualiza sus dependencias desde PowerShell con `& .\.venv\Scripts\python.exe -m pip install -r .\requirements.txt`.

`requirements.txt` reúne todas las dependencias de la aplicación, los complementos
(UMAP, lectura PDF, monitoreo de memoria y exportación PNG) y las pruebas controladas.

La aplicación requiere PostgreSQL `aves_db`. Copie `.env.example` como `.env` y configure
`DATABASE_URL` y `USER_AGENT` con sus valores locales. `.env` está excluido de Git;
no suba credenciales ni copias de la base de datos. La sincronización de inicio transforma
`observations`, `taxa`, `media`, `texts` y `vernacular_names` a las modalidades que analiza el proyecto.
Los registros locales de SQLite se reconstruyen desde PostgreSQL; SQLite solo conserva
el índice operativo, el historial de experimentos y las cachés. No se consultan GBIF,
iNaturalist ni Xeno-canto como fuentes de datos.

* Python **3.11 o superior**.
* **ffmpeg** en el PATH si los medios de `aves_db` incluyen archivos **M4A**.
* Pruebas controladas: `python -m pytest tests -q`.

La pestaña **Datos** muestra la cobertura de `aves_db` y permite descargar los medios
cuyas URLs ya están registradas en esa base. Las descargas completadas se guardan en
`media_content` de PostgreSQL y en `data_store/`, que está excluido del código versionado;
si se borra el almacén local, la app puede regenerarlo desde los BLOB de la base.

## Despliegue en Streamlit Community Cloud

1. Seleccione este repositorio, la rama `main` y `app.py` como archivo de entrada.
2. En **Advanced settings → Secrets**, configure las credenciales como TOML:

```toml
DATABASE_URL = "postgresql://USUARIO:CONTRASENA@HOST:PUERTO/aves_db"
USER_AGENT = "aves-dataset/0.2 (correo-de-contacto@ejemplo.com)"
```

Use la URL de una base PostgreSQL accesible desde Internet y limite su acceso con las
reglas de red de su proveedor. `127.0.0.1` apunta al contenedor de la app, no al equipo
local. La carpeta `data_store/` no se publica y el almacenamiento local de Community
Cloud no es persistente; conserve la fuente y los medios en PostgreSQL.

## Calidad de datos (pestaña «Calidad e inventario»)

* **Inventario**: registros sincronizados de `aves_db` (audio, imagen, texto y punto) como **válido / inválido / pendiente**, con su motivo e identificador de origen. Filtrable y exportable a CSV.
* **Informe de calidad por modalidad**: formatos, duraciones, resoluciones, longitud de textos, idioma, cobertura geográfica, completitud por evento, licencias y recomendaciones. Exportable a Markdown + CSV.
* **Verificar integridad**: comprueba que cada archivo válido exista, no esté vacío, se pueda abrir y coincida con su SHA-256.
* **Textos taxonómicos**: nombres comunes y descripciones existentes en las tablas de `aves_db`.
* **Auditoría**: registro de sincronización e informes de calidad.

## Qué es una muestra (barra lateral)

* **Fragmentos** (por defecto): parches de 32×32 px de cada imagen y ventanas de 1 s de cada audio.
* **Archivo completo**: un vector por imagen (79 descriptores de color, textura y composición) o por grabación (87 descriptores).
  Así cada muestra es un ejemplar con su taxonomía, y los resultados se pueden validar con ARI/NMI frente a las etiquetas disponibles.

## Estructura (sección 6 de la especificación)

| Carpeta | Contenido |
|---|---|
| `app.py`, `config.py` | interfaz única; configuración central (`config.toml` / variables `NS_*`) |
| `core/` | `engine` (Motor), `registry` (registro extensible), `results` (Datos/Res/Param), `cache`, `tasks`, `errors` |
| `data/` | Adaptador de `aves_db` (`postgres_source`), catálogo operativo SQLite, almacenamiento, validación y cargadores por modalidad |
| `ml/` | `algorithms` (12), `comparisons` (13 + ranking/sensibilidad/estabilidad), `recomendaciones`, `preprocessing`, `metrics`, `catalogo` |
| `visualization/` | `charts`, `maps`, `media`, `widgets` + `web/` (orbita, galería, mapa, audio, texto, hero…) |
| `ui/` | `pages.py`, `paneles.py` (calidad, grupos, sugerencias, comparación de ejecuciones), `styles.css` |
| `data_store/` | Estado operativo regenerable: índice local, medios asociados a `aves_db`, derivados, cachés y resultados |
| `tests/fixtures/manuales/` | Muestras de `datos/` para pruebas controladas; la app de investigación no las carga |

## Decisiones
* **Se retiró Tkinter** (`interfaz.py`) y se fusionaron `app.py` y `app(1).py` (eran idénticos): una sola interfaz.
* **Cuatro tipos reales:** audio, imágenes, texto y geografía (no tabulares/temporales genéricos).
* **Descargar ≠ analizar:** `data/download.py` no importa librosa, PIL ni scikit-learn. Las características se calculan al pedirlas y se guardan por SHA-256 + parámetros + versión.
* **TF-IDF disperso** de punta a punta; solo se densifica (con tope de MB) para algoritmos que lo exigen.
* **Fuente de investigación:** `aves_db` es la fuente de registros; `data_store/` guarda estado derivado y `tests/fixtures/manuales/` conserva un corpus independiente para pruebas controladas.
* **LDA discriminante** queda marcado como método supervisado/heredado; LDA de Dirichlet es el temático.
* Registro perezoso: listar los algoritmos no importa scikit-learn, Plotly ni UMAP (hay una prueba).

## Añadir un algoritmo sin tocar el núcleo
```python
from core.registry import registrar
from core.results import Param, Res
def mi_metodo(datos, X, p): return Res("informe", Z=X[:, :2], ejes=["a", "b"])
registrar("Mi método", mi_metodo, [Param("k", "k", 3, 2, 10)], tipos=("audio", "img"), descripcion="…", min_muestras=10)
```
La interfaz genera los controles sola. `retirar("Mi método")` lo quita.

## Trazabilidad requerimiento → implementación
| Módulo | Dónde | Pruebas |
|---|---|---|
| I Arquitectura | `core/*`, `config.py`, `ui/pages.py` | `test_algorithms` (registro, perezoso), `test_integration` |
| II Adquisición |  `core/engine.ingerir*`, `data/download.py`, `data/collect.py`, `data/storage.py` | `test_storage` |
| III Tipos de datos |  `data/loaders/*` | `test_loaders` |
| IV Preparación |  `ml/preprocessing.py`, `core/results.Datos` | `test_algorithms` |
| V Algoritmos |  `ml/algorithms.py`, `ml/catalogo.py`, `core/registry.py` | `test_algorithms` |
| VI Comparativas |  `ml/comparisons.py`, `core/engine.exportar` | `test_integration` |
| VII Gráficos |  `visualization/charts.py`, `ui/pages.mostrar_algoritmo` | `test_performance` |
| VIII Multimedia |  `visualization/media.py`, `widgets.py`, `web/` | prueba de interfaz manual |
| IX Base y almacenamiento |  `data/db.py`, `data/storage.py`, `core/cache.py` | `test_storage`, `test_integration` |
| X Rendimiento y seguridad |  `core/tasks.py`, `data/download.py`, `tests/` | `test_performance`, `test_storage` |

### Limitaciones conocidas (honestas)
* **Cancelación cooperativa:** se comprueba entre archivos, lotes, valores de k y descargas; un `fit` de scikit-learn ya iniciado termina antes de cancelarse.
* **PNG** requiere `kaleido` (opcional); HTML interactivo siempre está disponible.
* El monitoreo de memoria usa `psutil` si está instalado, y `resource` en Linux/macOS.
* `tests/` no incluye pruebas automáticas de los widgets web (se comprobó la interfaz en un navegador automatizado).
* El adaptador sincroniza el dataset de `aves_db` al iniciar; el servidor PostgreSQL debe estar disponible.


---

# Enfoque científico: biodiversidad multimodal y control de calidad

## Problema de investigación

Las colecciones de biodiversidad reúnen observaciones heterogéneas: fotografías, grabaciones, descripciones y ubicaciones geográficas. Sin embargo, estos datos suelen estar distribuidos entre repositorios y no siempre están completos. Esto dificulta comparar qué tipo de evidencia representa mejor las diferencias entre especies y detectar registros atípicos antes de utilizarlos.

La plataforma se orienta a estudiar esa estructura sin utilizar las etiquetas taxonómicas para construir los grupos. Las etiquetas conocidas (especie, género y clase) se reservan para la evaluación posterior.

Objetivo general
Evaluar la capacidad de algoritmos de aprendizaje no supervisado para descubrir patrones en registros de biodiversidad y detectar anomalías mediante características multimodales.

Objetivo específico 1 — Descubrimiento de patrones
Muestra comparable
- Audio e imagen sobre los mismos eventos.
- Texto y geografía mediante subconjuntos comparables.
- K-Means, clustering jerárquico, GMM, DBSCAN y HDBSCAN.
- ARI, NMI y métricas internas.
Objetivo específico 2 — Control de calidad
Corpus completo
- Todos los registros disponibles.
- Análisis independiente por modalidad.
- Detección de ruido, distancias e inconsistencias.
- Identificación de registros que requieren revisión.

Resultados esperados
1. Determinar qué modalidad presenta mayor capacidad de agrupamiento taxonómico.
2. Identificar qué información adicional aporta cada modalidad.
3. Detectar registros sospechosos que podrían afectar la calidad de los datos.
4. Establecer qué algoritmos son más adecuados para cada tipo de información.

## Preguntas de investigación

- ¿Qué modalidad produce agrupamientos que mejor coinciden con las categorías taxonómicas conocidas?
- ¿La combinación de modalidades mejora la separación respecto del uso individual?
- ¿Qué tipos de anomalías pueden detectarse sin etiquetas de error?
- ¿En qué medida la falta de fotografías, audios, textos o coordenadas sesga los resultados?

## Dataset de investigación

La fuente de registros es la base PostgreSQL `aves_db`. El adaptador sincroniza observaciones,
taxonomía, referencias de medios, textos, nombres comunes y coordenadas al catálogo operativo.
Las imágenes y audios se analizan cuando el archivo asociado está disponible; las descargas
usan solamente URLs registradas en esa base. No se consultan otras plataformas durante el análisis.

La pestaña Laboratorio compara audio e imagen sobre los mismos eventos cuando ambas modalidades
están disponibles. En las demás modalidades, los agrupamientos se evalúan con especie y clase
después de entrenar; esas etiquetas no forman parte de las variables de entrada.

## Diseño experimental recomendado

### Experimento A — Comparación por evento

Utilizar únicamente observaciones de `aves_db` que tengan dos o más modalidades asociadas al mismo evento. El laboratorio compara audio e imagen sobre esa intersección.

### Experimento B — Comparación por especie

Analizar cada modalidad por separado y reservar especie/clase para evaluación externa. Reportar cuántas observaciones y medios aportan a cada ejecución.

Mantener separadas las métricas por modalidad y usar eventos comunes al comparar audio con imagen. Las etiquetas taxonómicas solo evalúan los grupos después del ajuste.

### Validación del objetivo 1

- Interna: silueta (mayor es mejor), Davies-Bouldin (menor es mejor) y Calinski-Harabasz (mayor es mejor).
- Externa: ARI y NMI frente a especie y clase cuando están disponibles. Se calculan después de agrupar; no se entregan como variables de entrada.
- Comparación justa: mismas especies, mismo subconjunto de eventos cuando corresponda, misma semilla y parámetros documentados.
- No interpretar ARI/NMI altos como prueba de que el modelo identifica correctamente cualquier especie fuera del conjunto evaluado.

### Validación del objetivo 2

La detección de anomalías debe combinar:
- Ruido detectado por DBSCAN/HDBSCAN/OPTICS.
- Coordenadas inválidas, incertidumbre geográfica elevada o puntos espacialmente aislados.
- Medios que no se pueden descargar, decodificar o validar.
- Descripciones vacías, demasiado cortas o discordantes con la taxonomía.
- Duplicados, metadatos incompletos y desbalance por especie/fuente.

Una observación marcada como atípica no es automáticamente un error biológico. Se trata de una alerta para revisión. Para estimar precisión y sensibilidad, se recomienda crear un pequeño conjunto de casos revisados manualmente y/o inyectar anomalías controladas (coordenadas intercambiadas, fechas imposibles, archivos corruptos) como prueba técnica separada.

## Estrategia de almacenamiento ligero

1. PostgreSQL conserva los registros canónicos; SQLite mantiene el índice local y las ejecuciones.
2. Los medios solo se descargan desde URLs referenciadas por `aves_db`.
3. El almacén usa SHA-256 para deduplicar archivos.
4. Audio e imagen se procesan para producir características; las matrices derivadas y sus cachés son mucho más pequeñas que los originales.
5. El análisis puede trabajar con características y manifiestos. Los archivos multimedia pueden mantenerse fuera del ZIP y excluidos del control de versiones.
6. El ZIP de entrega debe incluir código, pruebas, documentación y, si se desea, una base pequeña de metadatos; no debe incluir `data_store/raw/`, temporales ni cachés regenerables.

No se eliminan automáticamente los originales: primero se debe comprobar que las características se calcularon correctamente y que el manifiesto permite reconstruir la procedencia.

## Uso

1. Instalar dependencias con `pip install -r requirements.txt`.
2. Ejecutar `streamlit run app.py`.
3. Abrir **Datos** para revisar la cobertura sincronizada desde `aves_db`.
4. Descargar los medios asociados que se quieran analizar.
5. Usar **Laboratorio de investigación** para comparar modalidades o analizar cada una.

## Limitaciones que deben declararse en el informe

- La cobertura de modalidades depende de los registros disponibles en `aves_db` y de los medios descargados.
- Los textos de especie proceden de las tablas del dataset; su contenido y cobertura dependen de esa base.
- Las coordenadas pueden ser aproximadas, generalizadas u ocultas.
- Una misma especie puede tener múltiples variedades, edades, sexos, comportamientos y condiciones ambientales; un grupo visual o acústico no necesariamente equivale a una especie.
- El dataset puede tener sesgos geográficos, taxonómicos y de cobertura de medios.
