"""Recomendaciones de parámetros según las características de los datos (n, dimensiones, densidad, varianza).

Son puntos de partida razonados, no óptimos garantizados: cada sugerencia trae su justificación para que
el usuario decida. Los cálculos son baratos (submuestreo a ≤1000 filas) para poder pedirlos antes de ejecutar.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .preprocessing import muestra_densa, rs


def _k_silueta(Xd: np.ndarray, k_max: int = 10) -> tuple[int, float]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    mejor, val = 2, -1.0
    for k in range(2, max(3, min(k_max, len(Xd) - 1)) + 1):
        lab = KMeans(k, n_init=3, random_state=rs()).fit_predict(Xd)
        if len(np.unique(lab)) < 2:
            continue
        s = silhouette_score(Xd, lab, sample_size=min(len(Xd), 1000), random_state=rs())
        if s > val:
            mejor, val = k, float(s)
    return mejor, val


def _rodilla(v: np.ndarray) -> float:
    v = np.sort(v)
    x, y = np.linspace(0, 1, len(v)), (v - v[0]) / (v[-1] - v[0] + 1e-12)
    return float(v[np.argmax(x - y)])


def _componentes(Xd: np.ndarray, umbral: float = 0.9, tope: int = 20) -> tuple[int, float]:
    from sklearn.decomposition import PCA

    q = min(Xd.shape) - 1
    if q < 2:
        return 2, 1.0
    v = np.cumsum(PCA(min(q, 50), random_state=rs()).fit(Xd).explained_variance_ratio_)
    k = int(np.searchsorted(v, umbral) + 1)
    k = int(np.clip(k, 2, tope))
    return k, float(v[min(k, len(v)) - 1])


def recomendar(nombre: str, d, X, params: dict | None = None) -> dict:
    """{parámetro: (valor sugerido, justificación)} para el algoritmo `nombre` sobre estos datos."""
    n, p = X.shape
    Xd, _ = muestra_densa(X, 1000, rs())
    Xd = np.nan_to_num(np.asarray(Xd, dtype=float))
    out: dict = {}
    metodo = (params or {}).get("metodo")

    def k_por_silueta(clave="k"):
        k, s = _k_silueta(Xd)
        out[clave] = (k, f"mejor silueta ({s:.2f}) entre k=2 y k={min(10, n - 1)} en una muestra de {len(Xd)} filas")

    if nombre in ("Agrupación · KMeans", "Clusterización jerárquica"):
        k_por_silueta()
    if nombre == "Modelo mixto gaussiano":
        k_por_silueta()
        out["covarianza"] = ("diag" if p > 20 else "full",
                             "con muchas variables «full» estima demasiados parámetros; «diag» es más estable" if p > 20
                             else "pocas variables: «full» permite elipses inclinadas")
    if nombre in ("KMeans (k óptimo)", "Métricas para agrupación"):
        out["k_max"] = (int(np.clip(round(np.sqrt(n / 2)), 4, 15)), "regla √(n/2), acotada entre 4 y 15")
    if nombre in ("DBSCAN · HDBSCAN", "Agrupación por escaneo (OPTICS)"):
        ms = int(np.clip(2 * min(p, 10), 4, max(4, n // 10)))
        out["min_samples"] = (ms, f"≈ 2 × dimensión efectiva (máx. 10 dims) y ≤ n/10; aquí p={p}, n={n}")
        if nombre == "DBSCAN · HDBSCAN" and metodo != "HDBSCAN":
            from sklearn.neighbors import NearestNeighbors

            kk = min(ms, len(Xd) - 1)
            dist = NearestNeighbors(n_neighbors=kk).fit(Xd).kneighbors(Xd)[0][:, -1]
            eps = _rodilla(dist)
            if eps <= 0:
                pos = dist[dist > 0]
                eps = float(np.median(pos)) if len(pos) else 1e-3
            out["eps"] = (round(float(eps), 4), f"codo de la curva de distancias al {kk}.º vecino")
        tam = int(np.clip(round(0.02 * n), 5, max(5, n // 3)))
        if nombre == "DBSCAN · HDBSCAN":
            out["tam_min"] = (tam, "≈ 2 % de las muestras (mín. 5): grupos más pequeños se consideran ruido")
        else:
            out["tam_min"] = (round(float(np.clip(tam / n, 0.01, 0.3)), 2), "≈ 2 % de las muestras como fracción")
    if nombre in ("Reducción de dimensionalidad · PCA",):
        k, v = _componentes(Xd)
        out["n_comp"] = (k, f"componentes necesarias para explicar ≥ 90 % de la varianza ({v:.0%})")
    if nombre == "Descomposición en valores singulares":
        k, v = _componentes(Xd, tope=30)
        out["rango"] = (k, f"rango que retiene ≈ 90 % de la energía ({v:.0%})")
    if nombre == "Aproximación iterativa · PPCA":
        k, v = _componentes(Xd, tope=10)
        out["n_latentes"] = (k, f"igual que PCA: dimensiones para ≈ 90 % de la varianza ({v:.0%}), máx. 10")
    if nombre == "t-SNE · UMAP":
        out["perplexity"] = (float(np.clip(np.floor((n - 1) / 3), 2, 50)), "entre 5 y 50 y nunca mayor que (n−1)/3")
        out["vecinos"] = (int(np.clip(round(np.sqrt(n)), 5, 50)), "≈ √n: más vecinos = estructura más global")
        out["metrica"] = ("cosine" if (d.tipo == "texto" or sp.issparse(X)) else "euclidean",
                          "TF-IDF se compara mejor por ángulo (coseno)" if d.tipo == "texto" else "variables continuas estandarizadas")
    return out
