"""Medios para los widgets web: miniaturas, fragmentos de audio, texto y parches.

Todo lo generado se guarda en `data_store/derived/` con clave = SHA-256 del original + parámetros, de modo que
se reutiliza mientras el archivo no cambie  y se puede borrar y regenerar. Del audio solo se
decodifica un fragmento, no el archivo completo.
"""
from __future__ import annotations

import base64
import io
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

from config import CFG
from core.cache import hash_obj
from data.storage import sha256_archivo


def _uri(mime: str, datos: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(datos).decode()}"


def _clave(ruta: Path) -> str:
    """El nombre ya es el SHA-256 en el almacén; para otros archivos se usa ruta + fecha."""
    n = Path(ruta).stem
    return n if len(n) == 64 else hash_obj(str(ruta), Path(ruta).stat().st_mtime)


def _derivado(sub: str, nombre: str, cfg=CFG) -> Path:
    return cfg.subdir("derived", sub) / nombre


def miniatura(ruta, lado: int | None = None, cfg=CFG) -> str:
    """Data-URI JPEG de una miniatura, con caché en disco."""
    from PIL import Image

    lado = lado or cfg.miniatura_px
    destino = _derivado("thumbs", f"{_clave(Path(ruta))}_{lado}.jpg", cfg)
    if not destino.is_file():
        im = Image.open(ruta)
        im.draft("RGB", (lado * 2, lado * 2))
        im = im.convert("RGB")
        im.thumbnail((lado, lado))
        tmp = destino.with_suffix(".tmp")
        im.save(tmp, "JPEG", quality=80)
        tmp.replace(destino)
    return _uri("image/jpeg", destino.read_bytes())


def fragmento_audio(ruta, inicio: float = 0.0, dur: float | None = None, cfg=CFG, sr: int = 16_000) -> str:
    """Data-URI WAV de un fragmento (por defecto, los primeros `clip_audio_s` s), decodificando solo ese tramo."""
    return _uri("audio/wav", fragmento_audio_bytes(ruta, inicio, dur, cfg, sr))


def fragmento_audio_bytes(ruta, inicio: float = 0.0, dur: float | None = None, cfg=CFG, sr: int = 16_000) -> bytes:
    from data.loaders.audio import decodificar

    dur = dur or cfg.clip_audio_s
    destino = _derivado("clips", f"{_clave(Path(ruta))}_{int(inicio * 10)}_{int(dur)}.wav", cfg)
    if not destino.is_file():
        y = decodificar(ruta, sr, dur, inicio)
        pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
        tmp = destino.with_suffix(".tmp")
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        tmp.replace(destino)
    return destino.read_bytes()


def espectrograma_fig(ruta, inicio: float = 0.0, dur: float = 3.0, sr: int = 22_050):
    """Espectrograma mel (dB) de un tramo de audio como figura Plotly: muestra la «huella» acústica del fragmento."""
    import librosa
    import plotly.graph_objects as go

    from data.loaders.audio import decodificar
    from visualization.charts import estilo

    y = decodificar(ruta, sr, dur, inicio)
    if len(y) < 512:
        y = np.pad(y, (0, 512 - len(y)))
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=256), ref=np.max)
    t = np.arange(S.shape[1]) * 256 / sr + inicio
    f = librosa.mel_frequencies(n_mels=64, fmax=sr / 2)
    fig = go.Figure(go.Heatmap(z=S, x=t, y=f, colorscale="Greys", colorbar=dict(title="dB")))
    fig.update_yaxes(type="log", title_text="Hz")
    return estilo(fig, "Espectrograma mel", x="s", h=230)


def texto(ruta, n: int = 1400) -> str:
    from data.loaders.text import leer_texto

    return _texto(str(ruta), Path(ruta).stat().st_mtime, n)


@lru_cache(maxsize=16)
def _texto(ruta, mtime, n):
    from data.loaders.text import leer_texto

    return " ".join(leer_texto(Path(ruta)).split())[:n]


@lru_cache(maxsize=8)
def _img_reducida(ruta, mtime, lado):
    from data.loaders.image import abrir_rgb

    return abrir_rgb(ruta, lado)


def parche(ruta, y: int, x: int, p: int | None = None, cfg=CFG):
    """Recorte de un parche de la imagen reducida (exactamente lo que vio el algoritmo."""
    p = p or cfg.img_parche
    ruta = Path(ruta)
    im = _img_reducida(str(ruta), ruta.stat().st_mtime, cfg.img_lado)
    if y < 0 or x < 0:                                       # granularidad «archivo»: la muestra es la imagen completa
        return im.resize((96, 96))
    return im.crop((x, y, x + p, y + p))


def parches_png(ruta, cfg=CFG):
    """Todos los parches de una imagen como lista de (y, x, PIL.Image) para el explorador de parches."""
    p = cfg.img_parche
    return [(y, x, parche(ruta, y, x, p, cfg)) for y in range(0, cfg.img_lado, p) for x in range(0, cfg.img_lado, p)]


# ------------------------------------------------------------------ explorador de mapa
def _prop(p, *claves):
    k = {str(a).lower(): v for a, v in p.items()}
    return next((str(k[c]) for c in claves if k.get(c) not in (None, "")), None)


def explorador_mapa(recursos_geo, recursos_img=(), recursos_audio=(), cfg=CFG, con=None, max_puntos=1500, max_medios_audio=24) -> dict:
    """Puntos del mapa con sus medios asociados (imagen y audio del mismo evento en la BD) para `mapa.html`."""
    from data import db

    c = con or db.connect(cfg)
    ev_img = {r.meta.get("event_id"): r for r in recursos_img if r.meta.get("event_id")}
    ev_aud = {r.meta.get("event_id"): r for r in recursos_audio if r.meta.get("event_id")}
    puntos, img, aud = [], {}, {}
    for f in c.execute("""SELECT g.event_id, g.lat, g.lon, e.species,
        COALESCE(
            (SELECT t.original FROM texts t WHERE t.event_id=g.event_id AND TRIM(COALESCE(t.original,''))<>'' ORDER BY t.field LIMIT 1),
            (SELECT s.text FROM species_texts s WHERE s.species=e.species AND TRIM(COALESCE(s.text,''))<>'' ORDER BY s.source LIMIT 1)
        ) AS texto
        FROM geo g JOIN events e USING(event_id) ORDER BY e.species, g.event_id LIMIT ?""", (max_puntos,)):
        ri, ra = ev_img.get(f["event_id"]), ev_aud.get(f["event_id"])
        if ri and ri.nombre not in img:
            img[ri.nombre] = miniatura(ri.ruta, 220, cfg)
        if ra and ra.nombre not in aud and len(aud) < max_medios_audio:
            try:
                aud[ra.nombre] = fragmento_audio(ra.ruta, 0, 8, cfg)
            except Exception:  # noqa: BLE001 - un audio ilegible no rompe el mapa
                ra = None
        elif ra and ra.nombre not in aud:
            ra = None
        texto = " ".join((f["texto"] or "").split())[:500] or None
        puntos.append(dict(la=f["lat"], lo=f["lon"], n=f["species"], i=ri.nombre if ri else None,
                           a=ra.nombre if ra else None, t=texto))
    return dict(puntos=puntos, img=img, audio=aud)
