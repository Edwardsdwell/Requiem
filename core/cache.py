"""Cachés de características y resultados en disco 

La clave de cada entrada se deriva del *contenido* (SHA-256 del archivo), del procesador, de su
versión y de los parámetros de extracción: si cualquiera cambia, la clave cambia y la entrada anterior
deja de usarse. Las escrituras son atómicas (archivo temporal único + replace),
de modo que dos procesos nunca comparten un temporal .
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
import uuid
from pathlib import Path

import numpy as np

_LOCKS: dict[str, threading.Lock] = {}
_GLOBAL = threading.Lock()


def hash_obj(*partes) -> str:
    """Hash estable (SHA-256 abreviado) de una combinación de valores JSON-serializables."""
    h = hashlib.sha256()
    for p in partes:
        h.update(json.dumps(p, sort_keys=True, default=_def, ensure_ascii=False).encode())
        h.update(b"\x1f")
    return h.hexdigest()[:32]


def _def(o):
    if isinstance(o, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(o).tobytes()).hexdigest()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def _lock(clave: str) -> threading.Lock:
    with _GLOBAL:
        return _LOCKS.setdefault(clave, threading.Lock())


def _escribir_atomico(destino: Path, escribir) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(f".{destino.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as f:
            escribir(f)
        os.replace(tmp, destino)
    finally:
        tmp.unlink(missing_ok=True)


class Cache:
    """Caché genérica por carpeta: `npz` para arrays y `pkl` para objetos."""

    def __init__(self, carpeta: Path):
        self.dir = Path(carpeta)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _p(self, clave: str, ext: str) -> Path:
        if not clave.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Clave de caché no válida")
        return self.dir / f"{clave}.{ext}"

    # --- arrays -------------------------------------------------------------------
    def leer_npz(self, clave: str):
        p = self._p(clave, "npz")
        if not p.is_file():
            return None
        try:
            with np.load(p, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
        except Exception:
            p.unlink(missing_ok=True)       # entrada corrupta: se descarta y se regenera
            return None

    def guardar_npz(self, clave: str, **arrays) -> None:
        with _lock(str(self.dir / clave)):
            _escribir_atomico(self._p(clave, "npz"), lambda f: np.savez_compressed(f, **arrays))

    # --- objetos ------------------------------------------------------------------
    def leer_pkl(self, clave: str):
        p = self._p(clave, "pkl")
        if not p.is_file():
            return None
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            p.unlink(missing_ok=True)
            return None

    def guardar_pkl(self, clave: str, obj) -> None:
        with _lock(str(self.dir / clave)):
            _escribir_atomico(self._p(clave, "pkl"), lambda f: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL))

    # --- mantenimiento ------------------------------------------------------------
    def tamano(self) -> int:
        return sum(p.stat().st_size for p in self.dir.rglob("*") if p.is_file())

    def n_entradas(self) -> int:
        return sum(1 for p in self.dir.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))

    def limpiar(self) -> int:
        """Elimina todo lo regenerable. Devuelve los bytes liberados."""
        n = 0
        for p in self.dir.rglob("*"):
            if p.is_file():
                n += p.stat().st_size
                p.unlink(missing_ok=True)
        return n
