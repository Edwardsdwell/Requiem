"""Métricas internas de agrupación: silueta, Davies-Bouldin, Calinski-Harabasz y proporción de ruido."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from .preprocessing import muestra_densa, rs


def evaluar(X, lab, max_muestras: int = 2000) -> dict:
    """Métricas internas ignorando el ruido (-1). Admite dispersas (submuestreo denso y reproducible)."""
    lab = np.asarray(lab)
    m = lab != -1
    out = dict(grupos=int(len(np.unique(lab[m]))), ruido=float(1 - m.mean()),
               silueta=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
    if out["grupos"] < 2 or m.sum() <= out["grupos"]:
        return out
    Xd, idx = muestra_densa(X if m.all() else X[m], max_muestras, rs())
    y = lab[m][idx]
    if not 2 <= len(np.unique(y)) < len(y):
        return out
    try:
        out["silueta"] = float(silhouette_score(Xd, y))
        out["davies_bouldin"] = float(davies_bouldin_score(Xd, y))
        out["calinski_harabasz"] = float(calinski_harabasz_score(Xd, y))
    except (ValueError, FloatingPointError):
        pass
    return out
