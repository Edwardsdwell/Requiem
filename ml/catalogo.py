"""Catálogo de los 12 algoritmos y las 13 comparativas.

Aquí solo hay metadatos y referencias perezosas a las funciones («módulo:función»): importar este
archivo no carga scikit-learn, Plotly ni UMAP. Para añadir o retirar métodos: `core.registry.registrar/retirar`.
"""
from core.registry import registrar
from core.results import Param

TODOS = ("img", "audio", "texto", "geo")
SIN_GEO = ("img", "audio", "texto")
ALG = "ml.algorithms:"
CMP = "ml.comparisons:"

# ---------------------------------------------------------------------------- algoritmos (orden de la especificación)
registrar("Agrupación · KMeans", ALG + "kmeans_fijo", [
    Param("k", "Grupos (k)", 3, 2, 20),
    Param("init", "Inicialización", "k-means++", opciones=["k-means++", "random"]),
], TODOS, "KMeans con el k que elijas y su silueta por muestra.", min_muestras=4, acepta_dispersa=True)

registrar("KMeans (k óptimo)", ALG + "kmeans_auto", [
    Param("k_max", "k máximo", 10, 3, 25),
    Param("criterio", "Criterio", "Silueta", opciones=["Silueta", "Calinski-Harabasz", "Davies-Bouldin"]),
], TODOS, "Prueba varios k y se queda con el mejor según un criterio.", min_muestras=6, acepta_dispersa=True)

registrar("Métricas para agrupación", ALG + "metricas", [Param("k_max", "k máximo", 10, 3, 25)], TODOS,
          "Inercia, silueta, Calinski-Harabasz y Davies-Bouldin según k.", min_muestras=6, acepta_dispersa=True)

registrar("Clusterización jerárquica", ALG + "jerarquico", [
    Param("k", "Grupos (corte)", 3, 2, 20),
    Param("enlace", "Enlace", "ward", opciones=["ward", "average", "complete", "single"]),
    Param("hojas", "Hojas del dendrograma", 30, 10, 100),
], TODOS, "Árbol de uniones que se corta en k grupos.", min_muestras=4)

registrar("Agrupación por escaneo (OPTICS)", ALG + "optics", [
    Param("min_samples", "min_samples", 10, 2, 50),
    Param("xi", "xi", 0.05, 0.01, 0.5, 0.01),
    Param("tam_min", "Tamaño mínimo (fracción)", 0.05, 0.01, 0.5, 0.01),
], TODOS, "Ordena los puntos por densidad: los valles son grupos.", min_muestras=12)

registrar("DBSCAN · HDBSCAN", ALG + "dbscan_hdbscan", [
    Param("metodo", "Método", "DBSCAN", opciones=["DBSCAN", "HDBSCAN"]),
    Param("min_samples", "min_samples", 5, 2, 50),
    Param("eps", "eps (0 = automático)", 0.0, 0.0, 1000.0, 0.1, si=("metodo", "DBSCAN")),
    Param("tam_min", "Tamaño mínimo de grupo", 10, 2, 200, si=("metodo", "HDBSCAN")),
], TODOS, "Grupos por densidad; lo disperso queda como ruido.", min_muestras=8)

registrar("Modelo mixto gaussiano", ALG + "gmm", [
    Param("k", "Componentes", 3, 1, 20),
    Param("covarianza", "Covarianza", "diag", opciones=["full", "tied", "diag", "spherical"]),
    Param("k_max", "k máximo (curva BIC/AIC)", 8, 2, 15),
], TODOS, "Grupos como campanas gaussianas, con probabilidad de pertenencia.", min_muestras=8)

registrar("Reducción de dimensionalidad · PCA", ALG + "pca", [Param("n_comp", "Componentes", 3, 2, 20)], SIN_GEO,
          "Ejes de mayor varianza y las variables que los forman.", min_muestras=4, min_variables=2)

registrar("Descomposición en valores singulares", ALG + "svd", [
    Param("rango", "Rango de aproximación", 3, 2, 30),
    Param("centrar", "Centrar datos", "sí", opciones=["sí", "no"]),
], SIN_GEO, "Factoriza los datos y mide cuánta energía retiene cada rango.", min_muestras=4, min_variables=2)

registrar("Aproximación iterativa · PPCA", ALG + "ppca", [
    Param("n_latentes", "Variables latentes", 2, 2, 10),
    Param("iteraciones", "Iteraciones EM", 200, 10, 1000),
], SIN_GEO, "PCA probabilístico estimado paso a paso con EM.", min_muestras=6, min_variables=3)

registrar("t-SNE · UMAP", ALG + "tsne_umap", [
    Param("metodo", "Método", "t-SNE", opciones=["t-SNE", "UMAP"]),
    Param("dims", "Dimensiones", 2, 2, 3),
    Param("metrica", "Métrica", "euclidean", opciones=["euclidean", "cosine", "manhattan"]),
    Param("perplexity", "Perplejidad", 30.0, 2.0, 100.0, 1.0, si=("metodo", "t-SNE")),
    Param("iteraciones", "Iteraciones", 1000, 250, 3000, si=("metodo", "t-SNE")),
    Param("vecinos", "Vecinos", 15, 2, 100, si=("metodo", "UMAP")),
    Param("min_dist", "Distancia mínima", 0.1, 0.0, 1.0, 0.05, si=("metodo", "UMAP")),
], SIN_GEO, "Mapas no lineales que conservan los vecindarios.", min_muestras=12, min_variables=3)

registrar("LDA · temas", ALG + "lda_temas", [Param("temas", "Temas", 5, 2, 15)], ("texto",),
          "Descubre temas en texto y asigna a cada fragmento su tema dominante.",
          min_muestras=10, min_variables=3, acepta_dispersa=True)

# ---------------------------------------------------------------------------- comparativas (original vs procesado)
DBS = [Param("metodo", "Método", "DBSCAN", opciones=["DBSCAN", "HDBSCAN"]),
       Param("min_samples", "min_samples", 5, 2, 50),
       Param("eps", "eps (0 = automático)", 0.0, 0.0, 1000.0, 0.1, si=("metodo", "DBSCAN")),
       Param("tam_min", "Tamaño mínimo de grupo", 10, 2, 200, si=("metodo", "HDBSCAN"))]


def _c(nombre, fn, descripcion, params, tipos=TODOS, **kw):
    registrar(nombre, CMP + fn, params, tipos, descripcion, clase="comparativa", **kw)


_c("KMeans paso a paso (Voronoi)", "kmeans_pasos", "Datos originales frente a las celdas de Voronoi y los centroides en cada iteración, con la curva de inercia.",
   [Param("k", "Grupos (k)", 3, 2, 12), Param("init", "Inicialización", "random", opciones=["random", "k-means++"]),
    Param("iteraciones", "Iteraciones máximas", 15, 3, 40)], min_muestras=6)
_c("Selección de k (codo · silueta · CH)", "seleccion_k", "Codo, Silhouette, Calinski-Harabasz y un score combinado normalizado; datos originales frente al k elegido.",
   [Param("k_max", "k máximo", 10, 3, 25)], min_muestras=8)
_c("KMeans: sin estandarizar vs estandarizado", "kmeans_escala", "Mismo KMeans sobre los datos crudos y sobre los estandarizados.",
   [Param("k", "Grupos (k)", 3, 2, 15)], min_muestras=6)
_c("DBSCAN con ruido", "dbscan_comp", "Zonas densas como grupos y puntos aislados como ruido; en coordenadas, también sobre el mapa.", DBS, min_muestras=8)
_c("Dendrograma jerárquico", "dendro", "Árbol de fusiones del clustering jerárquico y los grupos resultantes del corte.",
   [Param("k", "Grupos (corte)", 3, 2, 20), Param("enlace", "Enlace", "ward", opciones=["ward", "average", "complete", "single"]),
    Param("hojas", "Hojas del dendrograma", 30, 10, 100), Param("muestra", "Muestras a usar", 600, 100, 1500, 100)], min_muestras=6)
_c("Modelo mixto gaussiano (elipses)", "gmm_elipses", "Datos originales, KMeans y GMM con elipses de covarianza sobre la proyección 2D.",
   [Param("k", "Componentes", 3, 2, 10), Param("k_max", "k máximo (curva BIC/AIC)", 8, 2, 12)], min_muestras=10)
_c("PCA: proyección y cargas", "pca_comp", "Variables originales frente a las dos primeras componentes principales, con varianza y cargas.",
   [Param("n_comp", "Componentes", 5, 2, 20)], SIN_GEO, min_muestras=4, min_variables=3)
_c("SVD: energía y reconstrucción", "svd_comp", "Original frente a su reconstrucción de rango k, mapa de error y energía acumulada (imagen, espectrograma de audio o matriz de datos).",
   [Param("rango", "Rango k (valores singulares)", 30, 1, 300)], SIN_GEO, archivo=("img", "audio"), min_muestras=4, min_variables=3)
_c("Reconstrucción lineal: PCA · FA · PPCA", "reconstruccion", "Compara la reconstrucción y el error de PCA, análisis factorial y PPCA con el mismo rango latente.",
   [Param("q", "Variables latentes", 8, 1, 20), Param("iter", "Iteraciones EM (PPCA)", 100, 10, 500)], SIN_GEO, min_muestras=12, min_variables=4)
_c("t-SNE · UMAP", "no_lineal", "Proyección lineal (PCA) frente a una no lineal, con trustworthiness; las imágenes añaden parches superpuestos.",
   [Param("metodo", "Método", "t-SNE", opciones=["t-SNE", "UMAP"]),
    Param("perplexity", "Perplejidad", 30.0, 2.0, 100.0, 1.0, si=("metodo", "t-SNE")),
    Param("iteraciones", "Iteraciones", 1000, 250, 3000, si=("metodo", "t-SNE")),
    Param("vecinos", "Vecinos", 15, 2, 100, si=("metodo", "UMAP")),
    Param("min_dist", "Distancia mínima", 0.1, 0.0, 1.0, 0.05, si=("metodo", "UMAP")),
    Param("k_color", "k para colorear (si hay un solo archivo)", 3, 2, 10)], SIN_GEO, min_muestras=12, min_variables=3)
_c("LDA (discriminante)", "lda_comp", "Proyección PCA frente a los ejes discriminantes (método supervisado/heredado: usa clases o pseudoclases).",
   [Param("k", "Clases KMeans (si hay un solo archivo)", 3, 2, 10)], SIN_GEO, min_muestras=10, min_variables=3, supervisado=True)
_c("Compresión de color con K-means", "cuantizacion", "Imagen original frente a la imagen con solo K colores representativos (cuantización).",
   [Param("k", "Colores (K)", 8, 2, 64)], ("img",), archivo=("img",), min_muestras=1)
_c("Escala de grises y negativo", "gris_negativo", "Imagen original, en grises y en negativo, con el histograma de intensidades.",
   [], ("img",), archivo=("img",), min_muestras=1)
