"""Imágenes -> parches de 32×32 reducidos a 8×8×3 = 192 características."""
from __future__ import annotations

import gc

import numpy as np

from core.cache import hash_obj
from core.results import Datos
from . import Recurso, cache_caracteristicas, ensamblar

TIPO, NOMBRE, VERSION = "img", "Imágenes", "img-3"


def abrir_rgb(ruta, lado: int):
    """Abre, convierte a RGB y reduce con `draft` (JPEG se decodifica ya reducido: menos RAM)."""
    from PIL import Image  # carga diferida

    Image.MAX_IMAGE_PIXELS = 100_000_000                   # protege contra «bombas de descompresión»
    im = Image.open(ruta)
    im.draft("RGB", (lado * 2, lado * 2))
    return im.convert("RGB").resize((lado, lado))


def _extraer(ruta, cfg) -> dict:
    lado, p, red = cfg.img_lado, cfg.img_parche, cfg.img_reducido
    if p % red or lado % p:
        raise ValueError("Se requiere: lado múltiplo del parche y parche múltiplo de la reducción")
    f = p // red
    a = np.asarray(abrir_rgb(ruta, lado), dtype=np.float32) / 255            # normalización a [0, 1] (req. 49)
    filas, pos = [], []
    for y in range(0, lado, p):
        for x in range(0, lado, p):
            filas.append(a[y:y + p, x:x + p].reshape(red, f, red, f, 3).mean((1, 3)).ravel())
            pos.append((y, x))
    return dict(X=np.asarray(filas, dtype=np.float32), pos=np.asarray(pos, dtype=np.int32))


NOMBRES_ARCHIVO = ([f"{c}_media" for c in "RGB"] + [f"{c}_desv" for c in "RGB"] + [f"tono_{i}" for i in range(12)]
                   + [f"saturacion_{i}" for i in range(4)] + [f"brillo_{i}" for i in range(4)]
                   + ["borde_medio", "borde_desv", "densidad_bordes", "nitidez", "colorido"]
                   + [f"{c}[{i},{j}]" for i in range(4) for j in range(4) for c in "RGB"])


def _descriptores_archivo(ruta, cfg) -> dict:
    """Un vector por imagen: color (medias, desviaciones, histogramas HSV), textura (bordes, nitidez),
    colorido y la distribución espacial del color en una rejilla 4×4."""
    im = abrir_rgb(ruta, cfg.img_lado)
    a = np.asarray(im, dtype=np.float32) / 255
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255
    h, s_, v = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
    hist = lambda x, b, w=None: np.histogram(x, b, (0, 1), weights=w)[0] / max(len(x), 1)
    gris = a.mean(2)
    gy, gx = np.gradient(gris)
    mag = np.hypot(gx, gy)
    lap = gris[:-2, 1:-1] + gris[2:, 1:-1] + gris[1:-1, :-2] + gris[1:-1, 2:] - 4 * gris[1:-1, 1:-1]
    rg, yb = a[..., 0] - a[..., 1], 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    colorido = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
    L = cfg.img_lado // 4
    rejilla = a[: 4 * L, : 4 * L].reshape(4, L, 4, L, 3).mean((1, 3)).ravel()
    x = np.r_[a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0), hist(h, 12, s_), hist(s_, 4), hist(v, 4),
              mag.mean(), mag.std(), (mag > 0.1).mean(), lap.var(), colorido, rejilla].astype(np.float32)
    return dict(X=x[None, :], pos=np.array([[-1, -1]], dtype=np.int32))


def cargar(recursos: list[Recurso], cfg, ctx=None) -> Datos:
    cache = cache_caracteristicas(cfg, TIPO)
    por_archivo = cfg.img_granularidad == "archivo"
    params = [cfg.img_lado, cfg.img_parche, cfg.img_reducido, "archivo" if por_archivo else "fragmentos"]
    bloques, avisos, usados, descartes = [], [], [], []
    for i, r in enumerate(recursos):
        if ctx is not None:
            ctx.punto(i / len(recursos), f"Imagen {i + 1}/{len(recursos)}: {r.nombre}")
        clave = hash_obj(VERSION, r.sha256, params)
        z = cache.leer_npz(clave)
        if z is None:
            try:
                z = _descriptores_archivo(r.ruta, cfg) if por_archivo else _extraer(r.ruta, cfg)
            except Exception as e:  # noqa: BLE001 - imagen corrupta: se omite con aviso
                avisos.append(f"Omitida {r.nombre}: {e}")
                descartes.append((r.nombre, "archivo", str(e)[:120]))
                continue
            cache.guardar_npz(clave, **z)
        usados.append(r)
        p = cfg.img_parche
        if por_archivo:
            esp = r.meta.get("especie")
            bloques.append(dict(X=z["X"], hover=np.array([f"{r.nombre}" + (f" · {esp}" if esp else "")]), ref=[(r.nombre, -1, -1)]))
            continue
        bloques.append(dict(X=z["X"], hover=np.array([f"{r.nombre} · parche ({y // p}, {x // p})" for y, x in z["pos"]]),
                            ref=[(r.nombre, int(y), int(x)) for y, x in z["pos"]]))
        if (i + 1) % cfg.lote == 0:
            gc.collect()
    if not bloques:
        from core.errors import ErrorDatos

        raise ErrorDatos("No se pudo leer ninguna imagen", "; ".join(avisos[:3]))
    recursos = usados
    red = cfg.img_reducido
    nombres = NOMBRES_ARCHIVO if por_archivo else [f"{'RGB'[c]}[{i},{j}]" for i in range(red) for j in range(red) for c in range(3)]
    X = np.vstack([b["X"] for b in bloques])
    desc = (f"{len(X)} imágenes completas · {X.shape[1]} descriptores de color, textura y composición" if por_archivo
            else f"{len(X)} parches de {cfg.img_parche}×{cfg.img_parche} px")
    d = ensamblar(TIPO, recursos, bloques, nombres, cfg, [VERSION, *params], desc, True)
    d.avisos, d.descartes, d.granularidad = avisos, descartes, "archivo" if por_archivo else "fragmentos"
    return d
