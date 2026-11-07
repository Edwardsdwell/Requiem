"""Validación de entradas: rutas escritas por el usuario, coordenadas y archivos multimedia.

Todo lo que entra al sistema pasa por aquí antes de considerarse **válido**. Nada lanza una
excepción por un dato malo: cada función devuelve un veredicto con el **motivo**, para que el
registro quede como «inválido» (con su razón) en lugar de perderse o de interrumpir la importación.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

# ------------------------------------------------------------------ rutas
_UNIDAD = re.compile(r"^[A-Za-z]:[\\/]")


def limpiar_ruta_usuario(txt: str | Path) -> Path:
    """Normaliza una ruta pegada por el usuario.

    * Quita comillas (el «Copiar como ruta» de Windows las añade) y espacios.
    * Acepta `file:///C:/...`, `~` y separadores `\\` o `/`.
    * En Windows conserva la letra de unidad; en Linux/macOS convierte `\\` en `/`.
    """
    s = str(txt).strip().strip('"').strip("'").strip()
    if s.lower().startswith("file:///"):
        s = s[8:] if _UNIDAD.match(s[8:]) else s[7:]
    s = s.replace("%20", " ")
    import os

    if os.name != "nt":
        s = s.replace("\\", "/")
    return Path(s).expanduser()


def nombre_de_ruta(local_path: str | None) -> str:
    """Último componente de una ruta guardada en la BD, sea de Windows (`data\\raw\\audio\\x.wav`) o POSIX."""
    if not local_path:
        return ""
    txt = str(local_path).strip()
    if "\\" in txt or _UNIDAD.match(txt):
        return PureWindowsPath(txt).name
    return PurePosixPath(txt).name


def partes_de_ruta(local_path: str | None) -> list[str]:
    """Componentes de una ruta Windows o POSIX, sin unidad, sin `.` ni `..` (no permite salir del almacén)."""
    if not local_path:
        return []
    txt = str(local_path).strip().replace("\\", "/")
    txt = re.sub(r"^[A-Za-z]:/", "", txt)
    return [p for p in PurePosixPath(txt).parts if p not in ("", "/", ".", "..")]


# ------------------------------------------------------------------ coordenadas
_HEMI = re.compile(r"^\s*([-+−]?\d+(?:[.,]\d+)?)\s*°?\s*([NSEWOnsewo])?\s*$")


def numero(v) -> float | None:
    """Convierte a número una coordenada: 17.5 · "17,5" · "−17.5" · "17.5 S" · "68.1 O". None si no se puede."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    s = str(v).strip().replace("−", "-")
    if not s:
        return None
    m = _HEMI.match(s)
    if not m:
        try:
            f = float(s)
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    f = float(m.group(1).replace(",", ".").replace("−", "-"))
    if m.group(2) and m.group(2).upper() in ("S", "W", "O"):
        f = -abs(f)
    return f if math.isfinite(f) else None


def validar_coordenada(lat, lon) -> tuple[float | None, float | None, str]:
    """(lat, lon, motivo). `motivo` vacío = válida. Nunca lanza excepciones."""
    a, b = numero(lat), numero(lon)
    if a is None or b is None:
        return None, None, "latitud o longitud ausente o no numérica"
    if not -90 <= a <= 90:
        return None, None, f"latitud fuera de rango ({a})"
    if not -180 <= b <= 180:
        return None, None, f"longitud fuera de rango ({b})"
    if a == 0 and b == 0:
        return None, None, "coordenada nula (0, 0)"
    return a, b, ""


# ------------------------------------------------------------------ archivos
@dataclass
class Veredicto:
    ok: bool
    motivo: str = ""
    info: dict = field(default_factory=dict)       # dimensiones, duración, páginas, puntos…


_FIRMAS = {
    "jpg": (b"\xff\xd8\xff",), "png": (b"\x89PNG\r\n\x1a\n",), "bmp": (b"BM",), "webp": (b"RIFF",),
    "wav": (b"RIFF",), "flac": (b"fLaC",), "ogg": (b"OggS",), "pdf": (b"%PDF",),
}


def _cabecera(ruta: Path, n: int = 16) -> bytes:
    with open(ruta, "rb") as f:
        return f.read(n)


def formato_real(ruta: Path) -> str | None:
    """Formato según los bytes iniciales (no según la extensión)."""
    h = _cabecera(ruta, 16)
    if h.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if h.startswith(b"\x89PNG"):
        return "png"
    if h.startswith(b"BM"):
        return "bmp"
    if h[:4] == b"RIFF" and h[8:12] == b"WEBP":
        return "webp"
    if h[:4] == b"RIFF" and h[8:12] == b"WAVE":
        return "wav"
    if h.startswith(b"fLaC"):
        return "flac"
    if h.startswith(b"OggS"):
        return "ogg"
    if h[4:8] == b"ftyp":
        return "m4a"
    if h.startswith(b"ID3") or (len(h) > 1 and h[0] == 0xFF and (h[1] & 0xE0) == 0xE0):
        return "mp3"
    if h.startswith(b"%PDF"):
        return "pdf"
    return None


_COMPATIBLES = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "bmp": "bmp", "webp": "webp", "wav": "wav", "flac": "flac",
                "ogg": "ogg", "m4a": "m4a", "mp4": "m4a", "aac": "m4a", "mp3": "mp3", "pdf": "pdf"}


def validar_imagen(ruta: Path) -> Veredicto:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(ruta) as im:
            im.verify()                                     # detecta archivos truncados o corruptos
        with Image.open(ruta) as im:
            w, h = im.size
            im.draft("RGB", (64, 64))
            im.convert("RGB").load()                        # decodificación real (reducida)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        return Veredicto(False, f"imagen ilegible o corrupta ({type(e).__name__}: {str(e)[:80]})")
    if w < 8 or h < 8:
        return Veredicto(False, f"imagen demasiado pequeña ({w}×{h} px)", dict(ancho=w, alto=h))
    return Veredicto(True, info=dict(ancho=w, alto=h))


def _ffprobe(ruta: Path) -> dict | None:
    import json
    import shutil
    import subprocess

    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                              "stream=sample_rate,channels:format=duration", "-of", "json", str(ruta)],
                             capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return {"error": out.stderr.decode(errors="ignore")[:120]}
    j = json.loads(out.stdout or b"{}")
    s = (j.get("streams") or [{}])[0]
    if not s:
        return {"error": "sin pista de audio"}
    return dict(duracion_s=float(j.get("format", {}).get("duration") or 0), sr=int(s.get("sample_rate") or 0), canales=int(s.get("channels") or 0))


def validar_audio(ruta: Path) -> Veredicto:
    """Comprueba que el audio se puede abrir y que tiene duración. Usa soundfile (WAV, FLAC, OGG, MP3) y, para
    M4A u otros, ffprobe si está instalado. Sin ffprobe un M4A con cabecera correcta se acepta con aviso."""
    fmt = formato_real(ruta)
    try:
        import soundfile as sf

        i = sf.info(str(ruta))
        if i.frames <= 0 or i.samplerate <= 0:
            return Veredicto(False, "audio sin muestras (duración 0)")
        return Veredicto(True, info=dict(duracion_s=round(i.frames / i.samplerate, 3), sr=i.samplerate, canales=i.channels, formato=fmt))
    except Exception:  # noqa: BLE001 - formato no soportado por libsndfile: se intenta con ffprobe
        pass
    p = _ffprobe(ruta)
    if p is None:
        if fmt in ("m4a", "mp3"):
            return Veredicto(True, "no verificado a fondo: instale ffmpeg para comprobar M4A/MP3", dict(formato=fmt))
        return Veredicto(False, "formato de audio no reconocido")
    if "error" in p:
        return Veredicto(False, f"audio ilegible o corrupto ({p['error'].strip() or 'ffprobe'})")
    if p["duracion_s"] <= 0:
        return Veredicto(False, "audio sin duración")
    return Veredicto(True, info=dict(p, formato=fmt))


def validar_texto(ruta: Path) -> Veredicto:
    ext = ruta.suffix.lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader

            r = PdfReader(str(ruta))
            paginas = len(r.pages)
            txt = "".join((p.extract_text() or "") for p in r.pages[:20])
            if not txt.strip():
                return Veredicto(False, "PDF sin texto extraíble (¿escaneado?)", dict(paginas=paginas))
            return Veredicto(True, info=dict(paginas=paginas, caracteres=len(txt)))
        txt = ruta.read_bytes().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return Veredicto(False, f"texto ilegible ({type(e).__name__}: {str(e)[:80]})")
    if txt.count("\ufffd") > max(20, len(txt) * 0.05):
        return Veredicto(False, "codificación no reconocida (no es UTF-8)")
    if not txt.strip():
        return Veredicto(False, "archivo de texto vacío")
    return Veredicto(True, info=dict(caracteres=len(txt)))


def validar_geo(ruta: Path) -> Veredicto:
    from data.loaders.geo import leer_geo_informe

    try:
        pts, descartes = leer_geo_informe(ruta)
    except Exception as e:  # noqa: BLE001
        return Veredicto(False, f"archivo de coordenadas ilegible ({type(e).__name__}: {str(e)[:80]})")
    if not pts:
        motivo = descartes[0][1] if descartes else "no se encontraron columnas de latitud/longitud"
        return Veredicto(False, f"sin coordenadas válidas ({motivo})", dict(puntos=0, descartados=len(descartes)))
    return Veredicto(True, info=dict(puntos=len(pts), descartados=len(descartes)))


def validar_archivo(ruta: str | Path, tipo: str) -> Veredicto:
    """Veredicto completo de un archivo: existe, no está vacío, el contenido coincide con la extensión y se puede leer."""
    ruta = Path(ruta)
    if not ruta.is_file():
        return Veredicto(False, "archivo no encontrado")
    tam = ruta.stat().st_size
    if tam == 0:
        return Veredicto(False, "archivo vacío (0 bytes)")
    ext = ruta.suffix.lower().lstrip(".")
    real = formato_real(ruta)
    esperado = _COMPATIBLES.get(ext)
    if tipo in ("img", "audio") and esperado and real and real != esperado and not (esperado == "m4a" and real == "mp3"):
        return Veredicto(False, f"el contenido es {real.upper()} pero la extensión es .{ext}")
    try:
        v = {"img": validar_imagen, "audio": validar_audio, "texto": validar_texto, "geo": validar_geo}[tipo](ruta)
    except Exception as e:  # noqa: BLE001 - un validador nunca debe tumbar la importación
        v = Veredicto(False, f"error al validar ({type(e).__name__}: {str(e)[:80]})")
    v.info.setdefault("bytes", tam)
    return v
