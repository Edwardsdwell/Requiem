"""Los 12 algoritmos clásicos de aprendizaje no supervisado.

Contrato público: cada función recibe `(datos, X preparada, parámetros)` y devuelve un `Res`.
No dependen de la interfaz. Los metadatos (parámetros, tipos admitidos…) viven en `ml/catalogo.py`
y se registran en `core/registry.py`; este módulo solo se importa al ejecutar un algoritmo (carga diferida).

Algoritmos (12): kmeans_fijo, kmeans_auto, metricas, jerarquico, optics, dbscan_hdbscan, gmm,
                 pca, svd, ppca, tsne_umap, lda_temas.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as sp
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, KMeans, OPTICS
from sklearn.decomposition import PCA, LatentDirichletAllocation, TruncatedSVD
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances_argmin, silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from core.errors import ErrorValidacion
from core.results import Res
from visualization.charts import ESCALA_CONTINUA, categorias, estilo

from .metrics import evaluar
from .preprocessing import _distintas, media_filas, muestra_densa, punto, rs

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Validación y utilidades comunes
def _validar_X(X, minimo_filas=2, minimo_columnas=1):
    """Comprueba forma, tipo y valores finitos. Devuelve X (ndarray o matriz dispersa)."""
    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ErrorValidacion("La matriz de entrada no es válida.", "Se esperaba una matriz bidimensional.")
    n, m = X.shape
    if n < minimo_filas or m < minimo_columnas:
        raise ErrorValidacion(
            "Hay muy pocos datos para ejecutar el algoritmo.",
            f"Se necesitan al menos {minimo_filas} filas y {minimo_columnas} columnas; se recibieron {n} y {m}.")
    if sp.issparse(X):
        if not np.isfinite(X.data).all():
            raise ErrorValidacion("La matriz contiene valores no finitos.", "Revise los datos y vuelva a prepararlos.")
        return X
    X = np.asarray(X)
    if not np.issubdtype(X.dtype, np.number):
        raise ErrorValidacion("La matriz contiene variables no numéricas.", "Convierta las variables antes del análisis.")
    if not np.isfinite(X).all():
        raise ErrorValidacion("La matriz contiene valores no finitos.", "Trate los NaN o infinitos en el preprocesamiento.")
    return X


def _densa(X, max_mb=1024, contexto="este algoritmo"):
    """Densifica solo si el tamaño estimado cabe en el límite indicado."""
    if not sp.issparse(X):
        return np.asarray(X, dtype=np.float64)
    est = int(X.shape[0]) * int(X.shape[1]) * 8
    if est > max_mb * 1024**2:
        raise ErrorValidacion(
            f"No hay memoria segura para ejecutar {contexto}.",
            f"La matriz densa necesitaría ≈{est / 1024**2:.0f} MB (límite {max_mb} MB). "
            "Reduzca dimensiones o use un método compatible con matrices dispersas (p. ej. SVD).")
    return X.toarray().astype(np.float64, copy=False)


def _nombres(d, m):
    """Nombres de las variables; si no coinciden con las columnas se generan V1…Vm."""
    nom = getattr(d, "nombres", None)
    nom = [] if nom is None else list(nom)
    return nom if len(nom) == m else [f"V{i + 1}" for i in range(m)]


def _limitar_componentes(q, X, nombre):
    q = int(q)
    if q < 1:
        raise ErrorValidacion(f"El número de componentes de {nombre} no es válido.", "Debe ser al menos 1.")
    return min(q, min(X.shape))


def _proy(d, X, C=None):
    """Proyecta a ≤3 ejes para graficar (PCA, o SVD truncada si X es dispersa)."""
    m = X.shape[1]
    if m <= 3:                                   # incluye coordenadas geográficas: se mantienen las unidades originales
        return (_densa(X, contexto="la visualización") if sp.issparse(X) else np.asarray(X)), _nombres(d, m), C
    q = max(1, min(3, X.shape[0]))
    if sp.issparse(X):
        mod = TruncatedSVD(q, random_state=rs()).fit(X)
        ejes = [f"SV{i + 1} ({v:.0%})" for i, v in enumerate(mod.explained_variance_ratio_)]
        return mod.transform(X), ejes, None if C is None else mod.transform(C)
    mod = PCA(q, random_state=rs()).fit(X)
    ejes = [f"PC{i + 1} ({v:.0%})" for i, v in enumerate(mod.explained_variance_ratio_)]
    return mod.transform(X), ejes, None if C is None else mod.transform(C)


def _perfil(X, lab, nombres, n=4):
    """Mapa de calor: variables que más distinguen a cada grupo respecto de la media."""
    ids = [g for g in np.unique(lab) if g != -1]
    if not ids or X.shape[1] <= 3:
        return None
    mu = media_filas(X)
    M = np.array([media_filas(X, lab == g) - mu for g in ids])
    cols = sorted({int(j) for fila in M for j in np.argsort(fila)[::-1][:n]})
    if not cols:
        return None
    fig = px.imshow(M[:, cols], x=[nombres[j] for j in cols], y=[f"G{g}" for g in ids], aspect="auto",
                    color_continuous_scale=ESCALA_CONTINUA, color_continuous_midpoint=0)
    return estilo(fig, "Perfil de grupos (desviación de la media)")


def _terminos_grupo(d, X, lab, n=8) -> dict:
    """Términos característicos de cada grupo de texto: mayor TF-IDF medio del grupo frente al resto."""
    out = {}
    if getattr(d, "tipo", None) != "texto":
        return out
    nom = _nombres(d, X.shape[1])
    lab = np.asarray(lab)
    mu = media_filas(X)
    for g in sorted(set(lab.tolist()) - {-1}):
        dif = media_filas(X, lab == g) - mu
        out[g] = ", ".join(nom[j] for j in np.argsort(dif)[::-1][:n] if dif[j] > 0)
    return out


def validacion_externa(d, lab) -> dict:
    """ARI y NMI frente a etiquetas conocidas (clase, especie). Solo evalúan: nunca intervienen en el agrupamiento."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    out, lab = {}, np.asarray(lab)
    for campo in ("clase", "especie"):
        y = d.etiquetas_externas(campo)
        if y is None or len(y) != len(lab):
            continue
        y = np.asarray(y)
        ok = pd.notna(y)
        if ok.sum() < 2:
            continue
        out[campo] = dict(ARI=float(adjusted_rand_score(y[ok], lab[ok])),
                          NMI=float(normalized_mutual_info_score(y[ok], lab[ok])), y=y)
    return out


def diagnostico(lab, n_total=None) -> list[str]:
    """Situaciones que invalidan o debilitan un agrupamiento, con una sugerencia concreta."""
    lab = np.asarray(lab)
    m = lab != -1
    g = len(np.unique(lab[m]))
    out = []
    if g == 0:
        out.append("⚠️ **Todas las muestras se consideraron ruido**: no se formó ningún grupo. Aumente `eps` (DBSCAN) o reduzca "
                   "`min_samples` / el tamaño mínimo de grupo, y revise la escala de las variables.")
    elif g == 1:
        out.append("⚠️ **Todas las muestras (no ruidosas) quedaron en un único grupo**: no se encontró estructura con estos "
                   "parámetros. Pruebe un k mayor, reduzca `eps` o revise si los datos tienen variabilidad suficiente.")
    if m.any() and (1 - m.mean()) > 0.5:
        out.append(f"⚠️ El {1 - m.mean():.0%} de las muestras es ruido: los grupos describen solo una minoría de los datos.")
    if g >= 2:
        _, tam = np.unique(lab[m], return_counts=True)
        if tam.max() / m.sum() > 0.9:
            out.append(f"⚠️ Un grupo concentra el {tam.max() / m.sum():.0%} de las muestras: agrupamiento muy desequilibrado.")
        if (tam == 1).sum() >= max(2, g // 2):
            out.append(f"⚠️ {(tam == 1).sum()} grupos tienen una sola muestra: probablemente son atípicos, no grupos reales.")
    return out


def _distribucion(lab):
    nom, orden, cmap = categorias(lab)
    cnt = pd.Series(nom).value_counts().reindex(orden).fillna(0)
    fig = px.bar(x=cnt.index, y=cnt.values, color=cnt.index, color_discrete_map=cmap, labels={"x": "grupo", "y": "muestras"})
    fig.update_layout(showlegend=False)
    return estilo(fig, "Distribución de muestras por grupo (gris = ruido)")


def _contingencia(lab, y, campo):
    nom, orden, _ = categorias(lab)
    t = pd.crosstab(pd.Series(nom, name="grupo"), pd.Series(np.asarray(y), name=campo)).reindex(orden)
    if t.shape[1] > 25:                                   # demasiadas categorías: se muestran las 25 más frecuentes
        t = t[t.sum().sort_values(ascending=False).index[:25]]
    fig = px.imshow(t.values, x=[str(c) for c in t.columns], y=list(t.index), aspect="auto", color_continuous_scale=ESCALA_CONTINUA,
                    text_auto=True, labels={"color": "muestras"})
    return estilo(fig, f"Grupos frente a {campo} conocida (solo para evaluar)", h=320 + 12 * len(t))


def _agrupar(d, X, lab, cabecera, C=None, figuras=(), kpis=None):
    """Resultado común a todos los algoritmos de agrupamiento (métricas, tabla, diagnóstico, validación externa)."""
    lab = np.asarray(lab).astype(int)
    if len(lab) != X.shape[0]:
        raise ErrorValidacion("El algoritmo devolvió una cantidad incorrecta de etiquetas.",
                              "El número de etiquetas debe coincidir con el de muestras.")
    Z, ejes, C = _proy(d, X, C)
    m = lab != -1
    k = {"Grupos": len(np.unique(lab[m])), **(kpis or {})}
    if not m.all():
        k["Ruido"] = f"{1 - m.mean():.0%}"
    q = evaluar(X, lab)
    if np.isfinite(q["silueta"]):
        k["Silueta"], k["Davies-Bouldin"], k["Calinski-Harabasz"] = (
            f"{q['silueta']:.2f}", f"{q['davies_bouldin']:.2f}", f"{q['calinski_harabasz']:.0f}")
    n = pd.Series(["ruido" if g == -1 else f"G{g}" for g in lab]).value_counts()
    tabla = pd.DataFrame({"grupo": n.index, "muestras": n.values, "%": (100 * n.values / len(lab)).round(1)})
    terminos = _terminos_grupo(d, X, lab)
    if terminos:
        tabla["términos característicos"] = [terminos.get(int(g[1:]), "") if g != "ruido" else "" for g in tabla.grupo]
    origen = getattr(d, "origen", None)
    if origen is not None and len(set(origen)) > 1:       # archivo o categoría de origen más frecuente en cada grupo
        org = pd.Series(origen)
        def _origen(g):
            sel = lab == (-1 if g == "ruido" else int(g[1:]))
            return org[sel].value_counts().index[0] if sel.any() else ""
        tabla["origen más frecuente"] = [_origen(g) for g in tabla.grupo]
    figs = [*figuras, _distribucion(lab)]
    ext = validacion_externa(d, lab)
    for campo, v in ext.items():
        k[f"ARI vs {campo}"] = f"{v['ARI']:.2f}"
        k[f"NMI vs {campo}"] = f"{v['NMI']:.2f}"
    if "clase" in ext:
        figs.append(_contingencia(lab, ext["clase"]["y"], "clase"))
    perfil = _perfil(X, lab, _nombres(d, X.shape[1]))
    if perfil is not None:
        figs.append(perfil)
    diag = diagnostico(lab)
    informe = cabecera + ("\n\n" + "\n\n".join(diag) if diag else "")
    if ext:
        informe += ("\n\n**Validación externa** (las etiquetas no se usan para agrupar): " + " · ".join(
            f"{c}: ARI {v['ARI']:.2f}, NMI {v['NMI']:.2f}" for c, v in ext.items()) +
            ". ARI≈0 = coincidencia al azar; 1 = los grupos reproducen exactamente las categorías.")
    return Res(informe, Z, ejes, lab, C, figs, tabla, k,
               extra=dict(diagnostico=diag, externa={c: {a: b for a, b in v.items() if a != "y"} for c, v in ext.items()}))


def _barrido(X, k_max):
    """KMeans para k = 2…k_max; devuelve inercia, silueta, CH y DB por k."""
    X = _validar_X(X)
    max_k = min(int(k_max), X.shape[0] - 1, _distintas(X))
    if max_k < 2:
        raise ErrorValidacion("No hay suficientes muestras distintas para evaluar K-Means.",
                              "Se requieren al menos tres muestras distintas. Revise duplicados o variables constantes.")
    filas = []
    for k in range(2, max_k + 1):
        punto(None, f"KMeans k={k}")
        km = KMeans(k, n_init=5, random_state=rs()).fit(X)
        q = evaluar(X, km.labels_)
        filas.append(dict(k=k, inercia=km.inertia_, silueta=q["silueta"], CH=q["calinski_harabasz"], DB=q["davies_bouldin"]))
    t = pd.DataFrame(filas, columns=["k", "inercia", "silueta", "CH", "DB"])
    if t[["silueta", "CH", "DB"]].isna().all().all():
        raise ErrorValidacion("No se pudo evaluar ningún k.", "Revise duplicados, variables constantes y el tamaño del conjunto.")
    return t


# ---------------------------------------------------------------------------
# Agrupamiento
def kmeans_fijo(d, X, p):
    X = _validar_X(X)
    k, init = int(p["k"]), p.get("init", "k-means++")
    tope = min(X.shape[0], _distintas(X))
    if k < 2 or k > tope:
        raise ErrorValidacion("El valor de k no es válido.", f"k debe estar entre 2 y {tope} según el tamaño y la diversidad de los datos.")
    km = KMeans(k, init=init, n_init=10, random_state=rs()).fit(X)
    kp = {"Inercia": f"{km.inertia_:.0f}"}
    if len(np.unique(km.labels_)) < 2:
        return _agrupar(d, X, km.labels_, f"**K-Means** k={k}: todas las muestras cayeron en el mismo grupo", km.cluster_centers_, [], kp)
    Xs, idx = muestra_densa(X, min(5000, X.shape[0]), rs())       # silueta por muestra: O(n²), se limita a 5000 filas
    lab_s = km.labels_[idx]
    s = silhouette_samples(Xs, lab_s) if 1 < len(np.unique(lab_s)) < len(lab_s) else np.zeros(len(lab_s))
    fig, y0 = go.Figure(), 0
    for g in np.unique(lab_s):
        v = np.sort(s[lab_s == g])
        fig.add_scatter(x=v, y=np.arange(y0, y0 + len(v)), mode="lines", fill="tozerox", name=f"G{g}")
        y0 += len(v) + 10
    fig.add_vline(x=float(s.mean()), line_dash="dash", annotation_text="media")
    fig.update_yaxes(showticklabels=False)
    estilo(fig, "Silueta por muestra", x="silueta")
    return _agrupar(d, X, km.labels_, f"**K-Means** k={k}, inicialización `{init}`", km.cluster_centers_, [fig], kp)


def kmeans_auto(d, X, p):
    criterios = {"Silueta": "silueta", "Calinski-Harabasz": "CH", "Davies-Bouldin": "DB"}
    crit = p.get("criterio", "Silueta")
    if crit not in criterios:
        raise ErrorValidacion("Criterio de selección desconocido.", "Elija Silueta, Calinski-Harabasz o Davies-Bouldin.")
    col = criterios[crit]
    t = _barrido(X, p["k_max"])
    ok = t[col].notna()
    if ok.any():
        serie = t.loc[ok, col]
        k = int(t.loc[serie.idxmin() if col == "DB" else serie.idxmax(), "k"])
    else:
        k = int(t.loc[t["inercia"].idxmin(), "k"])
    km = KMeans(k, n_init=20, random_state=rs()).fit(X)
    fig = px.line(t, x="k", y=col, markers=True)
    fig.add_vline(x=k, line_dash="dash", line_color="crimson", annotation_text=f"k={k}")
    estilo(fig, f"Selección de k · {crit}")
    return _agrupar(d, X, km.labels_, f"**K-Means** con k óptimo = {k} (criterio: {crit})", km.cluster_centers_, [fig], {"k óptimo": k})


def metricas(d, X, p):
    t = _barrido(X, p["k_max"])
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Inercia (codo)", "Silueta ↑", "Calinski-Harabasz ↑", "Davies-Bouldin ↓"))
    mejores = {}
    for i, (c, sentido) in enumerate([("inercia", None), ("silueta", "max"), ("CH", "max"), ("DB", "min")]):
        r, q = divmod(i, 2)
        fig.add_scatter(x=t.k, y=t[c], mode="lines+markers", showlegend=False, row=r + 1, col=q + 1)
        ok = t[c].notna()
        if sentido and ok.any():
            serie = t.loc[ok, c]
            j = serie.idxmax() if sentido == "max" else serie.idxmin()
            mejores[c] = int(t.loc[j, "k"])
            fig.add_scatter(x=[t.loc[j, "k"]], y=[t.loc[j, c]], mode="markers", showlegend=False, row=r + 1, col=q + 1,
                            marker=dict(symbol="star", size=15, color="crimson"))
    estilo(fig, "Métricas de K-Means según k", x="k", h=540)
    return Res("**Métricas de K-Means** para cada k; la estrella marca el mejor valor.", figuras=[fig], tabla=t.round(4),
               kpis={f"Mejor k ({c})": k for c, k in mejores.items()})


def _asignar_centros(X, centros, bloque=2000):
    """Centro más cercano para cada fila, por bloques (sin densificar toda la matriz)."""
    out = np.empty(X.shape[0], dtype=int)
    for i in range(0, X.shape[0], bloque):
        B = X[i:i + bloque]
        out[i:i + bloque] = pairwise_distances_argmin(B.toarray() if sp.issparse(B) else np.asarray(B), centros)
    return out


def jerarquico(d, X, p, max_mb: float | None = None):
    """Enlace jerárquico. La matriz de distancias ocupa n·(n−1)/2·8 bytes: si supera el límite configurado se construye
    el árbol con una submuestra y el resto se asigna al centro de grupo más cercano."""
    from config import CFG

    X = _validar_X(X)
    n = X.shape[0]
    k, met = int(p["k"]), p["enlace"]
    if met not in {"single", "complete", "average", "weighted", "centroid", "median", "ward"}:
        raise ErrorValidacion("Método de enlace no válido.", "Elija un enlace admitido: ward, complete, average, single…")
    if not 2 <= k < n:
        raise ErrorValidacion("El número de grupos no es válido.", f"k debe estar entre 2 y {n - 1}.")
    max_mb = max(16.0, float(max_mb or getattr(CFG, "max_distancias_mb", 512)))
    m_max = max(3, int((1 + np.sqrt(1 + max_mb * 2**20)) / 2))
    aviso = ""
    if n > m_max:
        idx = np.sort(np.random.default_rng(rs()).choice(n, m_max, replace=False))
        aviso = (f" · árbol construido con {m_max:,} de {n:,} muestras (límite de {max_mb:g} MB para las distancias); "
                 "el resto se asignó al centro más cercano")
    else:
        idx = np.arange(n)
    Xs = _densa(X[idx], contexto="el enlace jerárquico")          # solo se densifica la submuestra acotada
    L = linkage(Xs, method=met)
    lab_s = fcluster(L, k, "maxclust") - 1
    if len(idx) < n:
        centros = np.vstack([Xs[lab_s == g].mean(axis=0) for g in np.unique(lab_s)])
        lab = _asignar_centros(X, centros)
        lab[idx] = lab_s
    else:
        lab = lab_s
    coph, _ = cophenet(L, pdist(Xs))
    hojas = int(np.clip(p.get("hojas", 30), 2, max(2, len(Xs))))
    dd = dendrogram(L, truncate_mode="lastp", p=hojas, no_plot=True)
    xs, ys = [], []
    for ic, dc in zip(dd["icoord"], dd["dcoord"]):
        xs += [*ic, None]
        ys += [*dc, None]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#64748b", width=1), hoverinfo="skip"))
    if len(L) >= k:
        fig.add_hline(y=(L[-k, 2] + L[-(k - 1), 2]) / 2, line_dash="dash", line_color="crimson", annotation_text=f"corte k={k}")
    fig.update_xaxes(showticklabels=False)
    estilo(fig, f"Dendrograma ({met}, {hojas} hojas)", y="distancia de unión")
    return _agrupar(d, X, lab, f"**Jerárquico** enlace `{met}`, k={k}{aviso}", None, [fig], {"Cofenética": f"{coph:.3f}"})


def optics(d, X, p):
    X = _validar_X(X)
    n = X.shape[0]
    ms, xi, tam = int(p["min_samples"]), float(p["xi"]), p.get("tam_min")
    if not 2 <= ms < n:
        raise ErrorValidacion("min_samples no es válido.", f"Debe estar entre 2 y {n - 1}.")
    if not 0 < xi < 1:
        raise ErrorValidacion("xi no es válido.", "Debe estar entre 0 y 1.")
    if tam is not None and not 0 < tam <= 1:
        raise ErrorValidacion("El tamaño mínimo de grupo no es válido.", "Debe ser una fracción entre 0 y 1 de las muestras.")
    o = OPTICS(min_samples=ms, xi=xi, min_cluster_size=tam).fit(X)
    r = o.reachability_[o.ordering_]
    fin = np.isfinite(r)
    r = np.where(fin, r, r[fin].max() if fin.any() else 0)
    nom, orden, cmap = categorias(o.labels_[o.ordering_])
    fig = px.bar(x=np.arange(len(r)), y=r, color=nom, color_discrete_map=cmap, category_orders={"color": orden},
                 labels={"x": "orden OPTICS", "y": "distancia de alcance", "color": ""})
    fig.update_traces(marker_line_width=0)
    fig.update_layout(bargap=0)
    estilo(fig, "Diagrama de alcance (valles = grupos)")
    return _agrupar(d, X, o.labels_, f"**OPTICS** min_samples={ms}, xi={xi:g}", None, [fig])


def _rodilla(v):
    """Valor de la curva ordenada con mayor distancia a la recta que une sus extremos (codo)."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0.0
    if len(v) < 3 or np.isclose(v[-1], v[0]):
        return float(np.median(v))
    x, y = np.linspace(0, 1, len(v)), (v - v[0]) / (v[-1] - v[0])
    return float(v[int(np.argmax(x - y))])


def dbscan_hdbscan(d, X, p):
    X = _validar_X(X)
    n = X.shape[0]
    ms = int(p["min_samples"])
    if not 2 <= ms < n:
        raise ErrorValidacion("min_samples no es válido.", f"Debe estar entre 2 y {n - 1}.")
    dist = np.sort(NearestNeighbors(n_neighbors=ms).fit(X).kneighbors(X)[0][:, -1])
    fig = px.line(x=np.arange(len(dist)), y=dist, labels={"x": "muestras ordenadas", "y": f"distancia al vecino {ms}"})
    metodo = p["metodo"]
    if metodo == "DBSCAN":
        eps_p = p.get("eps")
        auto = eps_p in (None, 0, 0.0)
        eps = _rodilla(dist) if auto else float(eps_p)
        if not np.isfinite(eps) or eps <= 0:                    # muchos duplicados exactos: el codo cae en 0
            pos = dist[dist > 0]
            eps = float(np.median(pos)) if len(pos) else 1e-3
        lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
        fig.add_hline(y=eps, line_dash="dash", line_color="crimson", annotation_text=f"eps={eps:.3g}")
        cab = f"**DBSCAN** eps={eps:.3g}{' (automático)' if auto else ''}, min_samples={ms}"
        if (lab == -1).all() or len(np.unique(lab)) == 1:
            cab += f". Sugerencia: el codo de la curva está en eps≈{_rodilla(dist):.3g}"
    elif metodo == "HDBSCAN":
        tam = int(p["tam_min"])
        if not 2 <= tam <= n:
            raise ErrorValidacion("min_cluster_size no es válido.", f"Debe estar entre 2 y {n}.")
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError:
            try:
                from hdbscan import HDBSCAN
            except ImportError as e:
                raise ErrorValidacion("HDBSCAN no está disponible.", "Actualice scikit-learn (≥1.3) o ejecute: pip install hdbscan") from e
        lab = HDBSCAN(min_cluster_size=tam, min_samples=ms).fit_predict(X)
        cab = f"**HDBSCAN** min_cluster_size={tam}, min_samples={ms}"
    else:
        raise ErrorValidacion("Método desconocido.", "Elija DBSCAN o HDBSCAN.")
    estilo(fig, "Distancia al k-ésimo vecino (el codo sugiere eps)")
    return _agrupar(d, X, lab, cab, None, [fig])


def gmm(d, X, p):
    X = _validar_X(X)
    Xd = _densa(X, contexto="GMM")
    n, m = Xd.shape
    k, cov = int(p["k"]), p["covarianza"]
    if not 1 <= k < n:
        raise ErrorValidacion("El número de componentes de GMM no es válido.", f"k debe estar entre 1 y {n - 1}.")
    if cov not in {"full", "tied", "diag", "spherical"}:
        raise ErrorValidacion("Tipo de covarianza no válido.", "Use full, tied, diag o spherical.")
    nota, C = "", None
    if m > 100:                                           # alta dimensión: las covarianzas serían singulares
        Xg = PCA(min(50, n - 1, m), random_state=rs()).fit_transform(Xd)
        nota = f" · ajustado sobre {Xg.shape[1]} componentes principales (datos de {m} dimensiones)"
    else:
        Xg, C = Xd, None
    modelo = GaussianMixture(k, covariance_type=cov, reg_covar=1e-4, n_init=3, max_iter=300, random_state=rs()).fit(Xg)
    figs = []
    ks = [] if p.get("_rapido") else list(range(1, min(int(p.get("k_max", 8)), n - 1) + 1))
    if ks:
        bic, aic = [], []
        for i in ks:
            punto(None, f"GMM con {i} componentes")
            aj = GaussianMixture(i, covariance_type=cov, reg_covar=1e-4, max_iter=200, random_state=rs()).fit(Xg)
            bic.append(aj.bic(Xg))
            aic.append(aj.aic(Xg))
        crit = go.Figure([go.Scatter(x=ks, y=bic, mode="lines+markers", name="BIC"),
                          go.Scatter(x=ks, y=aic, mode="lines+markers", name="AIC")])
        crit.add_vline(x=k, line_dash="dash", line_color="crimson")
        estilo(crit, "Selección de componentes (menor = mejor)", x="componentes")
        figs.append(crit)
    conf = px.histogram(x=modelo.predict_proba(Xg).max(1), nbins=30, labels={"x": "probabilidad máxima de pertenencia"})
    estilo(conf, "Confianza de la asignación", y="muestras")
    figs.append(conf)
    C = modelo.means_ if Xg is Xd else None               # los centros solo se proyectan si están en el espacio original
    pesos = ", ".join(f"{w:.2f}" for w in modelo.weights_)
    return _agrupar(d, X, modelo.predict(Xg), f"**GMM** {k} componentes, covarianza `{cov}` · pesos: {pesos}{nota}",
                    C, figs, {"BIC": f"{modelo.bic(Xg):.0f}", "AIC": f"{modelo.aic(Xg):.0f}"})


# ---------------------------------------------------------------------------
# Reducción de dimensionalidad
def _scree(v, nombre="PC"):
    et = [f"{nombre}{i + 1}" for i in range(len(v))]
    fig = go.Figure([go.Bar(x=et, y=v, name="individual"),
                     go.Scatter(x=et, y=np.cumsum(v), mode="lines+markers", name="acumulada")])
    fig.update_yaxes(tickformat=".0%")
    return fig


def pca(d, X, p):
    X = _validar_X(X)
    X = _densa(X, contexto="PCA")                         # PCA necesita centrar: solo si cabe en memoria
    q = _limitar_componentes(p["n_comp"], X, "PCA")
    pc = PCA(q, random_state=rs()).fit(X)
    v = pc.explained_variance_ratio_
    L = pc.components_[:5]
    top = np.argsort(np.abs(L).max(0))[::-1][:15]
    nom = _nombres(d, X.shape[1])
    carga = px.imshow(L[:, top], x=[nom[j] for j in top], y=[f"PC{i + 1}" for i in range(len(L))], aspect="auto",
                      color_continuous_scale="RdBu_r", color_continuous_midpoint=0)
    estilo(carga, "Cargas de las variables más influyentes")
    return Res(f"**PCA** con {q} componentes", pc.transform(X), [f"PC{i + 1} ({x:.0%})" for i, x in enumerate(v)],
               figuras=[estilo(_scree(v), "Varianza explicada"), carga], kpis={"Varianza explicada": f"{v.sum():.0%}"})


def svd(d, X, p):
    """SVD exacta si la matriz es pequeña; aleatorizada (TruncatedSVD) si es grande o dispersa."""
    X = _validar_X(X)
    centrar = p.get("centrar", "no") == "sí"
    if sp.issparse(X) and centrar:
        X = _densa(X, contexto="la SVD centrada")         # centrar vuelve densa la matriz: solo si cabe
    if centrar:
        X = np.asarray(X, dtype=float)
        X = X - X.mean(0)
    q = _limitar_componentes(p["rango"], X, "SVD")
    total = float(X.multiply(X).sum()) if sp.issparse(X) else float(np.sum(np.asarray(X, dtype=float) ** 2))
    total = max(total, 1e-15)
    exacta = (not sp.issparse(X)) and min(X.shape) <= 400
    if exacta:
        U, s, _ = np.linalg.svd(np.asarray(X, dtype=float), full_matrices=False)
        Z, titulo = U[:, :q] * s[:q], "SVD exacta"
    else:
        r = max(1, min(min(X.shape) - 1, max(q, 50)))
        mod = TruncatedSVD(r, n_iter=7, random_state=rs())
        Zf = mod.fit_transform(X)
        s, q = mod.singular_values_, min(q, r)
        Z, titulo = Zf[:, :q], "SVD truncada"
    e = s ** 2 / total
    err = np.sqrt(np.clip(1 - np.cumsum(e), 0, None))
    n = min(len(s), 50)
    ix = np.arange(1, n + 1)
    sv = go.Figure([go.Bar(x=ix, y=s[:n], name="σ"),
                    go.Scatter(x=ix, y=np.cumsum(e)[:n], mode="lines+markers", name="energía acumulada", yaxis="y2")])
    sv.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%", range=[0, 1.02]))
    estilo(sv, "Valores singulares", x="índice")
    rec = px.line(x=ix, y=err[:n], markers=True, labels={"x": "rango", "y": "error relativo"})
    rec.add_vline(x=q, line_dash="dash", line_color="crimson", annotation_text=f"r={q}")
    estilo(rec, "Error de reconstrucción según el rango")
    kp = {"Energía retenida": f"{e[:q].sum():.0%}", "Error de reconstrucción": f"{err[q - 1]:.0%}"}
    if exacta:
        kp["Número de condición"] = f"{s[0] / max(s[-1], 1e-12):.3g}"
    return Res(f"**{titulo}** de rango {q}", Z, [f"SV{i + 1}" for i in range(q)], figuras=[sv, rec], kpis=kp)


def _ppca(Xc, q, iters, tol=1e-6):
    """PPCA por EM (Tipping & Bishop, 1999). Modelo probabilístico lineal, no es una red neuronal."""
    n, d = Xc.shape
    tr = float(np.sum(Xc ** 2))
    s2 = max(tr / max(n * d, 1), 1e-8)
    W = np.random.default_rng(rs()).normal(scale=np.sqrt(s2), size=(d, q))
    eps = 1e-9 * max(tr / max(n * d, 1), 1e-12)           # regularización mínima: datos (casi) constantes no rompen la inversa
    hist = []
    for _ in range(int(iters)):
        Minv = np.linalg.pinv(W.T @ W + (s2 + eps) * np.eye(q))
        Ez = Xc @ W @ Minv
        S = n * s2 * Minv + Ez.T @ Ez
        Wn = Xc.T @ Ez @ np.linalg.pinv(S + eps * np.eye(q))
        s2n = max(float((tr - 2 * np.sum(Ez * (Xc @ Wn)) + np.trace(S @ Wn.T @ Wn)) / (n * d)), eps)
        delta = float(np.linalg.norm(Wn - W) / (np.linalg.norm(W) + eps))
        W, s2 = Wn, s2n
        hist.append((s2, delta))
        if delta < tol:
            break
    Z = Xc @ W @ np.linalg.pinv(W.T @ W + (s2 + eps) * np.eye(q))
    return Z, W, s2, hist


def ppca(d, X, p):
    X = _validar_X(X)
    X = _densa(X, contexto="PPCA")
    q = min(int(p["n_latentes"]), max(X.shape[1] - 1, 1), X.shape[0] - 1)
    if q < 1:
        raise ErrorValidacion("El número de variables latentes no es válido.", "Debe ser al menos 1 y menor que el número de muestras.")
    Ez, W, s2, h = _ppca(X - X.mean(0), q, max(1, int(p["iteraciones"])))
    Z = PCA(q, random_state=rs()).fit_transform(Ez)       # ejes principales del espacio latente
    it = np.arange(1, len(h) + 1)
    conv = make_subplots(rows=1, cols=2, subplot_titles=("Varianza del ruido σ²", "Cambio relativo en W (log)"))
    conv.add_scatter(x=it, y=[a for a, _ in h], mode="lines", showlegend=False, row=1, col=1)
    conv.add_scatter(x=it, y=[max(b, 1e-16) for _, b in h], mode="lines", showlegend=False, row=1, col=2)
    conv.update_yaxes(type="log", row=1, col=2)
    estilo(conv, "Convergencia del algoritmo EM", x="iteración")
    w2 = np.sort((W ** 2).sum(0))[::-1]
    aporte = px.bar(x=[f"z{i + 1}" for i in range(q)], y=w2, labels={"x": "variable latente", "y": "‖w‖²"})
    estilo(aporte, "Varianza que aporta cada variable latente")
    return Res(f"**PPCA** con {q} variables latentes", Z, [f"z{i + 1}" for i in range(q)], figuras=[conv, aporte],
               kpis={"σ²": f"{s2:.4g}", "Iteraciones EM": len(h)})


def tsne_umap(d, X, p):
    X = _validar_X(X, minimo_filas=5)
    n = X.shape[0]
    metodo, dims = p["metodo"], int(p["dims"])
    metrica = p.get("metrica", "euclidean")
    if dims not in (2, 3):
        raise ErrorValidacion("Dimensiones de salida no válidas.", "t-SNE y UMAP deben generar 2 o 3 dimensiones.")
    if metodo == "t-SNE":
        if sp.issparse(X):
            Xp = TruncatedSVD(min(50, X.shape[1] - 1, n - 1), random_state=rs()).fit_transform(X) if X.shape[1] > 51 else X
        else:
            Xp = PCA(min(50, X.shape[1], n - 1), random_state=rs()).fit_transform(X) if X.shape[1] > 50 else X
        perp = min(float(p["perplexity"]), max(1.0, (n - 1) / 3))      # regla práctica: perplejidad < n/3
        clave = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
        mod = TSNE(n_components=dims, perplexity=perp, metric=metrica, random_state=rs(),
                   init="random" if sp.issparse(Xp) else "pca", **{clave: max(250, int(p["iteraciones"]))})
        Z = mod.fit_transform(Xp)
        extra = f" · perplejidad {perp:g}" + (" (ajustada al tamaño)" if perp < float(p["perplexity"]) else "") + \
                f" · KL {mod.kl_divergence_:.3f}"
    elif metodo == "UMAP":
        try:
            import umap  # carga diferida: biblioteca pesada 
        except ImportError as e:
            raise ErrorValidacion("UMAP no está instalado.", "Ejecute: pip install umap-learn") from e
        vec = min(int(p["vecinos"]), n - 1)
        Z = umap.UMAP(n_components=dims, n_neighbors=vec, min_dist=float(p["min_dist"]), metric=metrica,
                      random_state=rs()).fit_transform(X)
        extra = f" · vecinos {vec} · min_dist {p['min_dist']}"
        Xp = X
    else:
        raise ErrorValidacion("Método de reducción desconocido.", "Elija t-SNE o UMAP.")
    try:                                                   # fidelidad de vecindarios, sobre ≤2000 muestras (coste O(n²))
        idx = np.sort(np.random.default_rng(rs()).choice(n, min(n, 2000), replace=False))
        A = Xp[idx]
        A = A.toarray() if sp.issparse(A) else np.asarray(A)
        tw = float(trustworthiness(A, Z[idx], n_neighbors=max(1, min(10, (len(idx) - 1) // 2)), metric=metrica))
    except Exception:
        tw = np.nan
    return Res(f"**{metodo}** a {dims}D{extra}", Z, [f"{metodo}-{i + 1}" for i in range(dims)],
               kpis={"Trustworthiness": f"{tw:.3f}" if np.isfinite(tw) else "No disponible"})


def lda_temas(d, X, p):
    """LDA de temas (Latent Dirichlet Allocation): no supervisado, solo para texto."""
    if getattr(d, "tipo", None) != "texto":
        raise ErrorValidacion("LDA de temas requiere datos de texto.",
                              "Cargue textos, o use PCA / SVD / t-SNE / UMAP para datos numéricos.")
    X = _validar_X(X)
    negativo = (X.data.size and X.data.min() < 0) if sp.issparse(X) else bool(np.any(X < 0))
    if negativo:
        raise ErrorValidacion("LDA requiere valores no negativos.", "Use conteos o TF-IDF sin centrar.")
    n, m = X.shape
    temas = int(p["temas"])
    if not 1 <= temas <= min(n, m):
        raise ErrorValidacion("Número de temas no válido.", f"Debe estar entre 1 y {min(n, m)}.")
    lda = LatentDirichletAllocation(temas, learning_method="batch", max_iter=int(p.get("iteraciones", 20)),
                                    random_state=rs()).fit(X)
    theta = lda.transform(X)
    lab = theta.argmax(1)
    nom = _nombres(d, m)
    top = [", ".join(nom[j] for j in np.argsort(c)[::-1][:8]) for c in lda.components_]
    cuenta = np.bincount(lab, minlength=temas)
    tam = px.bar(x=[f"T{i}" for i in range(temas)], y=cuenta, labels={"x": "tema dominante", "y": "fragmentos"})
    estilo(tam, "Fragmentos por tema dominante")
    q = min(3, theta.shape[1], n)
    if q >= 2:
        Z, ejes = PCA(q, random_state=rs()).fit_transform(theta), [f"PC{i + 1}" for i in range(q)]
    else:
        Z, ejes = np.c_[theta[:, 0], np.zeros(n)], ["Tema", "Eje 2"]
    tabla = pd.DataFrame({"tema": [f"T{i}" for i in range(temas)], "términos principales": top, "fragmentos": cuenta})
    return Res(f"**LDA (Latent Dirichlet Allocation)** con {temas} temas", Z, ejes, lab, None, [tam], tabla, {"Temas": temas},
               extra=dict(distribucion_documentos=cuenta.tolist()))
