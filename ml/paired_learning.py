"""Evaluación emparejada de descriptores acústicos y visuales por evento."""
from __future__ import annotations

from collections import Counter

import numpy as np


def _por_evento(salida):
    d, etiquetas = salida.datos, salida.res.etiquetas
    filas = {}
    for i, meta in enumerate(d.meta or []):
        evento = meta.get("event_id")
        if evento is None:
            continue
        fila = filas.setdefault(evento, {"x": [], "species": meta.get("especie"), "cluster": []})
        fila["x"].append(np.asarray(d.X[i].toarray() if hasattr(d.X[i], "toarray") else d.X[i]).ravel())
        if etiquetas is not None:
            fila["cluster"].append(etiquetas[i])
    return {
        evento: {
            "x": np.mean(fila["x"], axis=0),
            "species": fila["species"],
            "cluster": Counter(fila["cluster"]).most_common(1)[0][0] if fila["cluster"] else None,
        }
        for evento, fila in filas.items()
    }


def comparar(resultados, semilla=0):
    """Alinea filas por event_id, compara agrupamientos y evalúa especies con CV por grupos."""
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline

    por_tipo = {tipo: _por_evento(salida) for tipo, salida in resultados.items()}
    comunes = sorted(set.intersection(*(set(filas) for filas in por_tipo.values())))
    if not comunes:
        return {"n_comunes": 0, "tabla": [], "acuerdo_ari": None, "supervisado": []}

    audio, imagen = por_tipo["audio"], por_tipo["img"]
    etiquetables = [e for e in comunes if audio[e]["cluster"] is not None and imagen[e]["cluster"] is not None]
    ari = float(adjusted_rand_score([audio[e]["cluster"] for e in etiquetables],
                                    [imagen[e]["cluster"] for e in etiquetables])) if len(etiquetables) >= 2 else None

    por_modalidad = {}
    base = {}
    for tipo, filas in por_tipo.items():
        ids = [e for e in comunes if filas[e]["species"] and len(filas[e]["x"]) > 0]
        conteo = Counter(filas[e]["species"] for e in ids)
        ids = [e for e in ids if conteo[filas[e]["species"]] >= 2]
        if len(ids) < 4 or len({filas[e]["species"] for e in ids}) < 2:
            continue
        X = np.vstack([filas[e]["x"] for e in ids])
        y = np.asarray([filas[e]["species"] for e in ids])
        grupos = np.asarray(ids)
        n_splits = min(5, min(Counter(y).values()))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(semilla))
        pred, real = np.empty(len(y), dtype=object), np.empty(len(y), dtype=object)
        for train, test in cv.split(X, y, grupos):
            modelo = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(
                n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=int(semilla), n_jobs=-1))
            modelo.fit(X[train], y[train])
            pred[test], real[test] = modelo.predict(X[test]), y[test]
        por_modalidad[tipo] = {
            "eventos": len(ids), "especies": len(set(y)), "pliegues": n_splits,
            "balanced_accuracy": float(balanced_accuracy_score(real, pred)),
            "f1_macro": float(f1_score(real, pred, average="macro", zero_division=0)),
        }
        base[tipo] = float(1 / len(set(y)))

    return {"n_comunes": len(comunes), "acuerdo_ari": ari, "supervisado": por_modalidad,
            "referencia_azar": base, "cohorte_taxonomica": sum(
                1 for e in comunes if audio[e]["species"] and audio[e]["species"] == imagen[e]["species"])}
