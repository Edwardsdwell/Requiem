"""Almacenamiento por contenido en `data_store/`.

    data_store/raw/<tipo>/<sha256>.<ext>   originales (solo lectura lógica: nunca se modifican)
    data_store/raw/_tmp/                   descargas en curso, con nombres únicos
    data_store/derived/                    miniaturas y fragmentos regenerables
    data_store/cache/                      características y resultados
    data_store/results/                    exportaciones, registro de errores

El nombre de cada original es su SHA-256: un mismo contenido ocupa un solo archivo aunque lo
referencien varios registros o se suba dos veces.
"""
from __future__ import annotations

import hashlib
import os
import time
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath

from core.errors import ErrorValidacion

EXT = {
    "audio": {".mp3", ".wav", ".flac", ".ogg", ".m4a"},
    "img": {".jpg", ".jpeg", ".png", ".bmp", ".webp"},
    "texto": {".txt", ".md", ".pdf"},
    "geo": {".json", ".geojson", ".csv", ".txt"},
}
CARPETA = {"audio": "audio", "img": "images", "texto": "text", "geo": "geo"}
PELIGROSAS = {".exe", ".dll", ".bat", ".cmd", ".com", ".scr", ".msi", ".ps1", ".sh", ".py", ".js", ".jar", ".vbs", ".lnk", ".html", ".htm", ".svg"}
MIME_EXT = {
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
    "audio/flac": ".flac", "audio/x-flac": ".flac", "audio/ogg": ".ogg", "application/ogg": ".ogg",
    "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/aac": ".m4a",
    "application/pdf": ".pdf", "text/plain": ".txt", "text/markdown": ".md", "application/json": ".json",
    "application/geo+json": ".geojson", "text/csv": ".csv",
    "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/bmp": ".bmp", "image/webp": ".webp",
}


def sha256_archivo(ruta: Path, bloque: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while b := f.read(bloque):
            h.update(b)
    return h.hexdigest()


def detectar_tipo(ruta: str | Path) -> str | None:
    """Tipo de dato por extensión. `.txt` y `.csv` ambiguos se resuelven mirando el contenido."""
    ext = Path(ruta).suffix.lower()
    if ext in EXT["audio"]:
        return "audio"
    if ext in EXT["img"]:
        return "img"
    if ext in (".pdf", ".md"):
        return "texto"
    if ext in (".geojson",):
        return "geo"
    if ext in (".json", ".csv", ".txt"):
        try:
            muestra = Path(ruta).read_text(encoding="utf-8", errors="ignore")[:4000].lower()
        except OSError:
            return None
        if ext == ".json" or '"coordinates"' in muestra:
            return "geo"
        cab = muestra.splitlines()[0] if muestra else ""
        if any(c in cab for c in ("lat", "lon", "lng")) or (ext == ".txt" and _parece_coordenadas(muestra)):
            return "geo"
        return "texto" if ext == ".txt" else "geo"
    return None


def _parece_coordenadas(t: str) -> bool:
    import re

    lineas = [l for l in t.splitlines() if l.strip()][:20]
    return bool(lineas) and sum(len(re.findall(r"-?\d+\.\d+", l)) >= 2 and len(l) < 80 for l in lineas) > len(lineas) * 0.6


class Almacen:
    def __init__(self, cfg):
        self.cfg = cfg
        self.raiz = cfg.subdir("raw")
        self.tmp = cfg.subdir("raw", "_tmp")
        self.derivados = cfg.subdir("derived")
        for c in CARPETA.values():
            cfg.subdir("raw", c)

    # ------------------------------------------------------------------ rutas
    def dir_tipo(self, tipo: str) -> Path:
        return self.raiz / CARPETA[tipo]

    def relativa(self, ruta: Path) -> str:
        return ruta.resolve().relative_to(self.cfg.store_dir).as_posix()

    def resolver(self, local_path: str | None, tipo: str | None = None) -> Path | None:
        """Convierte una ruta guardada en la BD (relativa, o heredada de Windows) en un archivo real dentro
        de `raw/`; devuelve None si no existe o si apuntara fuera del almacén."""
        if not local_path:
            return None
        from .validacion import partes_de_ruta

        partes = partes_de_ruta(local_path)                  # admite «data\\raw\\audio\\x.wav», «C:\\…» y rutas POSIX
        if "raw" in partes:                                  # …/raw/audio/xxx.wav  →  audio/xxx.wav
            partes = partes[partes.index("raw") + 1:]
        elif tipo:
            partes = [CARPETA[tipo], partes[-1]]
        if not partes:
            return None
        p = (self.raiz.joinpath(*partes)).resolve()
        if self.raiz.resolve() not in p.parents or not p.is_file():
            return None
        return p

    def buscar_sha(self, tipo: str, sha: str) -> Path | None:
        if len(sha) != 64 or not all(c in "0123456789abcdef" for c in sha):
            return None
        for p in self.dir_tipo(tipo).glob(f"{sha}.*"):
            return p
        return None

    # ------------------------------------------------------------------ validación 
    def validar_nombre(self, nombre: str, tipo: str) -> str:
        base = PureWindowsPath(PurePosixPath(nombre).name).name       # sin carpetas ni traversal
        ext = Path(base).suffix.lower()
        if not base or base.startswith(".") or ".." in base:
            raise ErrorValidacion(f"Nombre de archivo no válido: {nombre!r}")
        if ext in PELIGROSAS:
            raise ErrorValidacion(f"Tipo de archivo no permitido: {ext}")
        if ext not in EXT[tipo]:
            raise ErrorValidacion(f"{base}: formato {ext or '(sin extensión)'} no compatible con «{tipo}» (admite {', '.join(sorted(EXT[tipo]))})")
        return base

    def limite_bytes(self, tipo: str) -> int:
        mb = {"audio": self.cfg.max_audio_mb, "img": self.cfg.max_imagen_mb}.get(tipo, self.cfg.max_texto_mb)
        return int(mb * 2**20)

    # ------------------------------------------------------------------ colocación atómica 
    def nuevo_temporal(self, ext: str = ".part") -> Path:
        return self.tmp / f"{uuid.uuid4().hex}{ext}"        # nombre único: sin colisiones entre hilos 

    def colocar(self, tmp: Path, tipo: str, sha: str, ext: str) -> tuple[Path, bool]:
        """Mueve un temporal completo a su ubicación definitiva. Devuelve (ruta, era_duplicado)."""
        existente = self.buscar_sha(tipo, sha)
        if existente:
            tmp.unlink(missing_ok=True)
            return existente, True
        destino = self.dir_tipo(tipo) / f"{sha}{ext}"
        os.replace(tmp, destino)                               # atómico dentro del mismo volumen
        return destino, False

    def ingerir(self, origen: Path | str, tipo: str, nombre: str | None = None) -> tuple[Path, str, bool]:
        """Copia un archivo local al almacén por contenido, en bloques. Devuelve (ruta, sha, duplicado)."""
        origen = Path(origen)
        base = self.validar_nombre(nombre or origen.name, tipo)
        if origen.stat().st_size == 0:
            raise ErrorValidacion(f"{base}: archivo vacío (0 bytes)")
        if origen.stat().st_size > self.limite_bytes(tipo):
            raise ErrorValidacion(f"{base} supera el límite de {self.limite_bytes(tipo) // 2**20} MB")
        tmp, h = self.nuevo_temporal(), hashlib.sha256()
        try:
            with open(origen, "rb") as fi, open(tmp, "wb") as fo:
                while b := fi.read(self.cfg.chunk_bytes):
                    h.update(b)
                    fo.write(b)
            ruta, dup = self.colocar(tmp, tipo, h.hexdigest(), Path(base).suffix.lower())
        finally:
            tmp.unlink(missing_ok=True)
        return ruta, h.hexdigest(), dup

    def ingerir_bytes(self, datos: bytes, tipo: str, nombre: str) -> tuple[Path, str, bool]:
        tmp = self.nuevo_temporal()
        try:
            tmp.write_bytes(datos)
            return self.ingerir(tmp, tipo, nombre)
        finally:
            tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------ mantenimiento
    def limpiar_temporales(self, edad_s: float = 3600, forzar: bool = False) -> int:
        """Elimina descargas abandonadas; no toca archivos válidos de `raw/<tipo>/`."""
        n, ahora = 0, time.time()
        for p in self.tmp.glob("*"):
            if forzar or ahora - p.stat().st_mtime > edad_s:
                p.unlink(missing_ok=True)
                n += 1
        return n

    def verificar(self, ruta: Path, sha: str) -> bool:
        """Comprueba la integridad antes de reutilizar un archivo."""
        return ruta.is_file() and sha256_archivo(ruta) == sha

    def uso_disco(self) -> dict:
        def tam(d: Path) -> int:
            return sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) if d.exists() else 0

        s = self.cfg.store_dir
        return {"originales": tam(self.raiz), "derivados": tam(self.derivados), "caché": tam(s / "cache"), "resultados": tam(s / "results")}

    def limpiar_derivados(self) -> int:
        n = 0
        for p in self.derivados.rglob("*"):
            if p.is_file():
                n += p.stat().st_size
                p.unlink(missing_ok=True)
        return n
