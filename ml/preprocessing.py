"""Preparación, normalización y control de calidad de datos.

Nunca modifica los archivos originales: trabaja sobre copias de la matriz en memoria.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from core.cache import Cache, hash_obj
from core.errors import ErrorValidacion

# --- semilla y contexto de tarea por hilo (reproducibilidad, cancelación) --------
_SEMILLA: ContextVar[int] = ContextVar("semilla", default=0)
_CTX: ContextVar = ContextVar("ctx", default=None)


def rs() -> int:
    """Semilla aleatoria de la ejecución en curso."""
    return _SEMILLA.get()


def punto(frac=None, msg=None) -> None:
    """Punto de control cooperativo: actualiza el progreso y lanza `Cancelado` si se pidió cancelar."""
    ctx = _CTX.get()
    if ctx is not None:
        ctx.punto(frac, msg)


@dataclass
class Prep:
    escalado: str = "auto"          # auto | standard | minmax | l2 | ninguno
    ausentes: str = "imputar"       # excluir | imputar | señalar
    quitar_constantes: bool = True

    def clave(self):
        return [self.escalado, self.ausentes, self.quitar_constantes]


ESCALADOS = ["auto", "standard", "minmax", "l2", "ninguno"]
AUSENTES = ["imputar", "excluir", "señalar"]


# ------------------------------------------------------------------ utilidades
def datos_finitos(X) -> np.ndarray:
    """Vector con los valores almacenados (para comprobar NaN/inf también en matrices dispersas)."""
    return X.data if sp.issparse(X) else np.asarray(X)


def densificar(X, max_mb: int = 512):
    """Convierte a densa solo si es imprescindible y cabe en el límite de memoria."""
    if not sp.issparse(X):
        return X
    mb = X.shape[0] * X.shape[1] * 8 / 2**20
    if mb > max_mb:
        raise ErrorValidacion(f"La matriz densa ocuparía {mb:.0f} MB (límite {max_mb} MB)",
                              "Reduzca las muestras o los términos, o use un algoritmo que admita matrices dispersas.")
    return X.toarray()


def media_filas(X, mask=None) -> np.ndarray:
    M = X if mask is None else X[mask]
    return np.asarray(M.mean(0)).ravel()


def muestra_densa(X, n: int = 2000, semilla: int = 0):
    """(submatriz densa, índices) para métricas que no admiten dispersas."""
    if X.shape[0] > n:
        idx = np.sort(np.random.default_rng(semilla).choice(X.shape[0], n, replace=False))
        M = X[idx]
    else:
        idx, M = np.arange(X.shape[0]), X
    return (M.toarray() if sp.issparse(M) else M), idx


def _distintas(X, tope: int = 50) -> int:
    """Número de filas distintas (hasta `tope`): KMeans no puede formar más grupos que puntos distintos."""
    M = X[: min(X.shape[0], 5000)]
    M = M.toarray() if sp.issparse(M) else np.asarray(M)
    return int(min(len(np.unique(np.round(M, 9), axis=0)), tope))


def _duplicados_dispersa(X) -> int:
    """Filas repetidas en una matriz dispersa (sin densificarla)."""
    X = X.tocsr()
    vistas = set()
    for i in range(X.shape[0]):
        a, b = X.indptr[i], X.indptr[i + 1]
        vistas.add((X.indices[a:b].tobytes(), X.data[a:b].tobytes()))
    return X.shape[0] - len(vistas)


# ------------------------------------------------------------------ calidad
def _constantes(X) -> np.ndarray:
    """Índices de las columnas sin variabilidad (en dispersas: columnas siempre cero)."""
    if sp.issparse(X):
        return np.flatnonzero(X.getnnz(0) == 0)
    sd = np.nanstd(np.where(np.isfinite(X), X, np.nan), 0)
    return np.flatnonzero(np.nan_to_num(sd, nan=0.0) == 0)


def calidad(d) -> dict:
    X = d.X
    v = datos_finitos(X)
    nan, inf = int(np.isnan(v).sum()), int(np.isinf(v).sum())
    if sp.issparse(X):
        filas_malas = np.unique(X.tocoo().row[~np.isfinite(X.tocoo().data)]) if (nan or inf) else np.array([], int)
        dup = _duplicados_dispersa(X)
        estad = dict(minimo=float(v.min()) if len(v) else 0.0, maximo=float(v.max()) if len(v) else 0.0,
                     no_nulos=f"{X.nnz / (X.shape[0] * X.shape[1]):.1%}")
        desc = None
    else:
        ok = np.isfinite(X)
        filas_malas = np.flatnonzero(~ok.all(1))
        dup = int(len(X) - len(np.unique(np.ascontiguousarray(X), axis=0))) if X.shape[0] <= 20000 else -1
        f = np.where(ok, X, np.nan)
        estad = dict(minimo=float(np.nanmin(f)), maximo=float(np.nanmax(f)))
        desc = pd.DataFrame({"variable": d.nombres, "mín": np.nanmin(f, 0), "máx": np.nanmax(f, 0), "media": np.nanmean(f, 0),
                             "desv.": np.nanstd(f, 0)}).round(4)
    constantes = _constantes(X)
    adv = []
    if len(filas_malas):
        adv.append(f"{len(filas_malas)} muestras con valores ausentes o infinitos")
    if len(constantes):
        adv.append(f"{len(constantes)} variables sin variabilidad (constantes)")
    if dup > 0:
        adv.append(f"{dup} muestras duplicadas (no se eliminan automáticamente)")
    if d.n < 10:
        adv.append("Muy pocas muestras: muchos métodos no serán fiables")
    adv += list(d.avisos or [])
    return dict(muestras=d.n, variables=X.shape[1], nan=nan, inf=inf, filas_incompletas=filas_malas, constantes=constantes,
                nombres_constantes=[d.nombres[i] for i in constantes[:10]], duplicados=dup, dispersa=sp.issparse(X),
                rango=estad, descriptivas=desc, advertencias=adv)


# ------------------------------------------------------------------ preparación
def tratar_ausentes(d, modo: str):
    """excluir: quita las muestras incompletas · imputar: mediana por variable · señalar: imputa y marca la muestra en `meta`."""
    X = d.X
    if sp.issparse(X):
        if np.isfinite(X.data).all():
            return d, 0
        X = X.copy()
        X.data[~np.isfinite(X.data)] = 0.0
        d2 = d.tomar(np.arange(d.n))
        d2.X = X
        return d2, int((~np.isfinite(d.X.data)).sum())
    ok = np.isfinite(X)
    if ok.all():
        return d, 0
    malas = np.flatnonzero(~ok.all(1))
    if modo == "excluir":
        return d.tomar(np.flatnonzero(ok.all(1))), len(malas)
    med = np.nanmedian(np.where(ok, X, np.nan), 0)
    med = np.nan_to_num(med, nan=0.0)
    X2 = np.where(ok, X, med)
    d2 = d.tomar(np.arange(d.n))
    d2.X = X2
    if modo == "señalar" and d2.meta is not None:
        for i in malas:
            d2.meta[i] = dict(d2.meta[i], incompleta=True)
    return d2, len(malas)


def _escalar(Xd, metodo: str, clave: str, cache: Cache | None):
    """Escala y guarda los parámetros ajustados (media y desviación) para reutilizarlos sin recalcular."""
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer

    disp = sp.issparse(Xd)
    if metodo == "l2":
        return Normalizer().fit_transform(Xd), {}
    if metodo == "minmax":
        return (MaxAbsScaler() if disp else MinMaxScaler()).fit_transform(Xd), {}
    if metodo != "standard":
        raise ErrorValidacion(f"Escalado desconocido: {metodo}")
    z = cache.leer_npz(clave) if (cache and clave) else None
    if z is None:
        if disp:
            from sklearn.utils.sparsefuncs import mean_variance_axis

            _, var = mean_variance_axis(Xd, axis=0)
            mean, scale = np.zeros(Xd.shape[1]), np.sqrt(var)
        else:
            mean, scale = Xd.mean(0), Xd.std(0)
        scale = np.where(scale == 0, 1.0, scale)
        if cache and clave:
            cache.guardar_npz(clave, mean=mean, scale=scale)
    else:
        mean, scale = z["mean"], z["scale"]
    if disp:                                    # sin centrar: conserva la dispersión
        return Xd.multiply(1.0 / scale).tocsr(), dict(scale=scale)
    return (Xd - mean) / scale, dict(mean=mean, scale=scale)


def preparar(d, prep: Prep | None = None, cache: Cache | None = None):
    """(Datos preparados, X escalada, informe). Aplica ausentes → constantes → escalado selectivo.

    `d.X` queda sin escalar (pero limpio y con las mismas columnas que `X`), porque varias comparativas
    contrastan «sin estandarizar» frente a «estandarizado»; `X` es la matriz que consumen los algoritmos.
    """
    prep = prep or Prep()
    info = dict(prep=prep.clave())
    d2, n_aus = tratar_ausentes(d, prep.ausentes)
    info["ausentes_tratados"] = n_aus
    if prep.quitar_constantes:
        c = _constantes(d2.X)
        if len(c) and len(c) < d2.X.shape[1] - 1:
            keep = np.setdiff1d(np.arange(d2.X.shape[1]), c)
            d2 = d2.con_columnas(keep)
            info["constantes_quitadas"] = len(c)
    metodo = prep.escalado
    if metodo == "auto":                                      # escalado selectivo 
        metodo = "standard" if d2.escalar else "ninguno"
    info["escalado"] = metodo
    if metodo == "ninguno":
        return d2, d2.X, info
    clave = hash_obj("escalador", d2.huella, metodo, d2.n, d2.X.shape[1], prep.clave())
    X, _ = _escalar(d2.X, metodo, clave if d2.huella else "", cache)
    return d2, X, info


# ------------------------------------------------------------------ validación 
def validar(entrada, d, X, params: dict) -> tuple[list, list]:
    """(errores, advertencias) antes de ejecutar. Los errores bloquean la ejecución."""
    err, adv = [], []
    n, p = X.shape
    if d.tipo and d.tipo not in entrada.tipos:
        err.append(f"«{entrada.nombre}» no es aplicable a datos de tipo «{d.tipo}» (admite: {', '.join(entrada.tipos)}).")
    if n < entrada.min_muestras:
        err.append(f"Se necesitan al menos {entrada.min_muestras} muestras y hay {n}.")
    if p < 1:
        err.append("No quedan variables después de la preparación.")
    elif n >= 2:                                                   # ¿hay variabilidad? (todas las filas iguales = nada que agrupar)
        sin_var = (X.nnz == 0 or (X.max(0) - X.min(0)).nnz == 0) if sp.issparse(X) else bool(np.all(np.ptp(np.asarray(X), 0) == 0))
        if sin_var:
            err.append("Los datos no tienen variabilidad: todas las muestras son idénticas, no hay estructura que encontrar.")
        else:
            nd = _distintas(X)
            v = params.get("k")
            if isinstance(v, int) and v > nd and nd < 50:
                err.append(f"«k» = {v} supera el número de muestras distintas ({nd}): hay filas repetidas.")
    for kk in ("min_samples",):
        v = params.get(kk)
        if isinstance(v, int) and v >= n:
            err.append(f"«{kk}» = {v} debe ser menor que el número de muestras ({n}).")
    if isinstance(params.get("perplexity"), float) and params["perplexity"] >= n:
        adv.append(f"La perplejidad ({params['perplexity']}) debe ser menor que el número de muestras ({n}): se reducirá a {n - 1}.")
    if p < entrada.min_variables:
        err.append(f"Se necesitan al menos {entrada.min_variables} variables y hay {p}.")
    for k in ("k", "temas"):
        v = params.get(k)
        if isinstance(v, (int, float)) and v >= n:
            err.append(f"«{k}» = {v} debe ser menor que el número de muestras ({n}).")
    for k in ("n_comp", "rango", "n_latentes", "q"):
        v = params.get(k)
        if isinstance(v, (int, float)) and v > min(n, p):
            adv.append(f"«{k}» = {v} supera min(muestras, variables) = {min(n, p)}: se reducirá automáticamente.")
    if "temas" in params and X.min() < 0:                         # LDA de temas
        err.append("LDA requiere valores no negativos: use el escalado «auto» o «ninguno» con TF-IDF.")
    if params.get("metodo") == "HDBSCAN" and params.get("tam_min", 0) > n:
        err.append("El tamaño mínimo de grupo supera el número de muestras.")
    if params.get("metodo") == "t-SNE" and n < 10:
        err.append("t-SNE necesita al menos 10 muestras.")
    if sp.issparse(X) and not entrada.acepta_dispersa:
        adv.append("La matriz es dispersa y este método necesita datos densos: se convertirá (con límite de memoria).")
    adv += list(d.avisos or [])
    return err, adv
