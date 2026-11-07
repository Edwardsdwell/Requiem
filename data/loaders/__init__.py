"""Cargadores especializados (audio, imágenes, texto, geografía) con una interfaz común
Cada módulo expone:  `TIPO`, `NOMBRE`, `EXTENSIONES`, `VERSION` y `cargar(recursos, cfg, ctx) -> Datos`.
Los recursos salen de la base de datos (descargados) y de los archivos locales ingeridos por contenido.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.cache import Cache, hash_obj
from core.errors import ErrorDatos, ErrorValidacion
from core.results import Datos

MODULOS = {"img": "data.loaders.image", "audio": "data.loaders.audio", "texto": "data.loaders.text", "geo": "data.loaders.geo"}
NOMBRES = {"img": "Imágenes", "audio": "Audio", "texto": "Texto", "geo": "Coordenadas"}


@dataclass
class Recurso:
    tipo: str
    nombre: str                      # nombre visible y único
    ruta: Path | None                # None para fuentes virtuales (textos y puntos de la BD)
    sha256: str
    meta: dict = field(default_factory=dict)   # event_id, especie, licencia, creador, fuente…

    @property
    def virtual(self) -> bool:
        return self.ruta is None


def modulo(tipo: str):
    """Importa el cargador del tipo solo cuando se necesita."""
    if tipo not in MODULOS:
        raise ErrorValidacion(f"Tipo de datos desconocido: {tipo}")
    return importlib.import_module(MODULOS[tipo])


def extensiones(tipo: str) -> tuple:
    from ..storage import EXT

    return tuple(sorted(EXT[tipo]))


# ------------------------------------------------------------------ listado de recursos
def listar(tipo: str, con, almacen) -> list[Recurso]:
    """Recursos utilizables de un tipo: registros descargados + archivos locales + fuentes virtuales de la BD."""
    out: list[Recurso] = []
    vistos: set[str] = set()
    if tipo in ("audio", "img"):
        tabla, idc = ("audio", "audio_id") if tipo == "audio" else ("images", "image_id")
        q = (f"SELECT t.{idc} AS id, t.sha256, t.local_path, t.license, t.creator, e.event_id, e.source_id, e.species, e.class, e.observed_raw "
             f"FROM {tabla} t JOIN events e USING(event_id) WHERE t.download_status='correcto' ORDER BY e.species, t.{idc}")
        for f in con.execute(q):
            ruta = almacen.resolver(f["local_path"], tipo)
            if ruta is None or not f["sha256"] or f["sha256"] in vistos:
                continue                                    # sin archivo, o contenido repetido: se usa una sola vez
            vistos.add(f["sha256"])
            esp = (f["species"] or "sin especie").replace(" ", "_")
            out.append(Recurso(tipo, f"{esp}_{f['sha256'][:8]}{ruta.suffix}", ruta, f["sha256"],
                               dict(event_id=f["event_id"], id_origen=f["source_id"], especie=f["species"], clase=f["class"], licencia=f["license"],
                                    creador=f["creator"], fecha=f["observed_raw"], registro=f["id"], fuente="gbif")))
        for d in con.execute("SELECT name, local_path, sha256 FROM documents WHERE kind=? ORDER BY name", (tipo,)):
            ruta = almacen.resolver(d["local_path"], tipo)
            if ruta and d["sha256"] not in vistos:
                vistos.add(d["sha256"])
                out.append(Recurso(tipo, d["name"], ruta, d["sha256"], dict(fuente="local")))
    else:
        for d in con.execute("SELECT name, local_path, sha256 FROM documents WHERE kind=? ORDER BY name", (tipo,)):
            ruta = almacen.resolver(d["local_path"], tipo)
            if ruta and d["sha256"] not in vistos:
                vistos.add(d["sha256"])
                out.append(Recurso(tipo, d["name"], ruta, d["sha256"], dict(fuente="local")))
        if tipo == "texto":
            filas = con.execute("SELECT event_id, field, original FROM texts WHERE TRIM(COALESCE(original,''))<>'' ORDER BY event_id, field").fetchall()
            if filas:
                sha = hash_obj([tuple(f) for f in filas])
                out.append(Recurso("texto", "Textos de la base de datos", None, sha, dict(fuente="bd", n=len(filas))))
            esp = con.execute("SELECT species, source, lang, text FROM species_texts WHERE TRIM(COALESCE(text,''))<>'' ORDER BY 1, 2, 3").fetchall()
            if esp:
                out.append(Recurso("texto", "Descripciones de especies", None, hash_obj([tuple(f) for f in esp]), dict(fuente="especies", n=len(esp))))
        else:
            n = con.execute("SELECT COUNT(*) FROM geo").fetchone()[0]
            if n:
                sha = hash_obj([tuple(f) for f in con.execute("SELECT event_id, lat, lon FROM geo ORDER BY event_id")])
                out.append(Recurso("geo", "Puntos de la base de datos", None, sha, dict(fuente="bd", n=n)))
    return out


# ------------------------------------------------------------------ ensamblado común
def ensamblar(tipo, recursos, bloques, nombres, cfg, params, descripcion, escalar=True, X_dispersa=None, **kw) -> Datos:
    """Une los bloques por recurso en un `Datos`, aplica el submuestreo reproducible."""
    X = X_dispersa if X_dispersa is not None else np.vstack([b["X"] for b in bloques])
    n = X.shape[0]
    origen = np.concatenate([[r.nombre] * len(b["hover"]) for r, b in zip(recursos, bloques)]) if kw.get("origen") is None else kw.pop("origen")
    hover = np.concatenate([b["hover"] for b in bloques]) if kw.get("hover") is None else kw.pop("hover")
    ref = kw.pop("ref", None) or [x for b in bloques for x in b["ref"]]
    meta = kw.pop("meta", None) or [dict(r.meta, archivo=r.nombre, sha256=r.sha256) for r, b in zip(recursos, bloques) for _ in b["hover"]]
    d = Datos(X, list(nombres), np.asarray(origen), np.asarray(hover), escalar, tipo=tipo, ref=ref, meta=meta,
              recursos={r.nombre: r.ruta for r in recursos if r.ruta}, descripcion=descripcion, **kw)
    d.huella = hash_obj(tipo, sorted(r.sha256 for r in recursos), params, cfg.max_muestras, cfg.semilla)
    return d


def submuestrear(d: Datos, max_muestras: int, semilla: int) -> Datos:
    if d.n <= max_muestras:
        return d
    idx = np.sort(np.random.default_rng(semilla).choice(d.n, max_muestras, replace=False))
    s = d.tomar(idx)
    s.descripcion += f" (submuestreo reproducible a {max_muestras}, semilla {semilla})"
    s.huella = hash_obj(d.huella, "sub", max_muestras, semilla)
    return s


def cargar(tipo: str, recursos: list[Recurso], cfg, ctx=None, max_muestras: int | None = None) -> Datos:
    if not recursos:
        raise ErrorDatos("No hay recursos seleccionados")
    m = modulo(tipo)
    d = m.cargar(recursos, cfg, ctx)
    d.tipo = tipo
    return submuestrear(d, max_muestras or cfg.max_muestras, cfg.semilla)


def cache_caracteristicas(cfg, tipo: str) -> Cache:
    return Cache(cfg.subdir("cache", "features", tipo))
