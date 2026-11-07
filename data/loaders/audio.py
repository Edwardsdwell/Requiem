"""Audio -> ventanas descritas con MFCC (media y desviación) + centroide, ZCR y RMS.

Los descriptores se calculan una sola vez por archivo y se guardan en caché (clave = SHA-256 del
contenido + parámetros + versión); un archivo que no cambia no se vuelve a decodificar.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from core.results import Datos
from . import Recurso, cache_caracteristicas, ensamblar
from core.cache import hash_obj

TIPO, NOMBRE, VERSION = "audio", "Audio", "audio-2"


def decodificar(ruta, sr: int, max_s: float, inicio: float = 0.0) -> np.ndarray:
    """Decodifica a mono float32. Usa librosa/soundfile y, si el formato no se reconoce (p. ej. M4A), ffmpeg."""
    import librosa  # carga diferida: solo si se usa audio 

    try:
        return librosa.load(str(ruta), sr=sr, mono=True, offset=inicio, duration=max_s)[0]
    except Exception as e:  # noqa: BLE001
        import shutil
        import subprocess

        if not shutil.which("ffmpeg"):
            raise RuntimeError(f"No se pudo decodificar {Path(ruta).name}: instale ffmpeg para leer este formato") from e
        cmd = ["ffmpeg", "-v", "error", "-ss", str(inicio), "-t", str(max_s), "-i", str(ruta), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
        out = subprocess.run(cmd, capture_output=True, timeout=120)
        if out.returncode != 0 or not out.stdout:
            raise RuntimeError(f"No se pudo decodificar {Path(ruta).name}") from e
        return np.frombuffer(out.stdout, dtype=np.float32).copy()


def _extraer(ruta, cfg) -> dict:
    import librosa

    sr, n_mfcc, hop = cfg.audio_sr, cfg.audio_n_mfcc, 512
    y = decodificar(ruta, sr, cfg.audio_max_s)
    if len(y) < sr // 4:
        y = np.pad(y, (0, sr // 4 - len(y)))
    F = np.vstack([librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop),
                   librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop),
                   librosa.feature.zero_crossing_rate(y, hop_length=hop),
                   librosa.feature.rms(y=y, hop_length=hop)])
    w, s = max(1, int(cfg.audio_ventana_s * sr / hop)), max(1, int(cfg.audio_salto_s * sr / hop))   # ventana y salto (solapamiento)
    w = min(w, F.shape[1])
    filas, tiempos = [], []
    for i in range(0, F.shape[1] - w + 1, s):
        g = F[:, i:i + w]
        filas.append(np.r_[g[:n_mfcc].mean(1), g[:n_mfcc].std(1), g[n_mfcc:].mean(1)])
        tiempos.append(i * hop / sr)
    return dict(X=np.asarray(filas, dtype=np.float32), t=np.asarray(tiempos, dtype=np.float32), dur=np.float32(len(y) / sr))


def cargar(recursos: list[Recurso], cfg, ctx=None) -> Datos:
    cache = cache_caracteristicas(cfg, TIPO)
    params = [cfg.audio_sr, cfg.audio_n_mfcc, cfg.audio_ventana_s, cfg.audio_salto_s, cfg.audio_max_s]
    bloques, aciertos, avisos, usados, descartes = [], 0, [], [], []
    por_archivo = cfg.audio_granularidad == "archivo"
    for i, r in enumerate(recursos):
        if ctx is not None:
            ctx.punto(i / len(recursos), f"Audio {i + 1}/{len(recursos)}: {r.nombre}")
        clave = hash_obj(VERSION, r.sha256, params)
        z = cache.leer_npz(clave)
        if z is None:
            try:
                z = _extraer(r.ruta, cfg)
            except Exception as e:  # noqa: BLE001 - un archivo ilegible no aborta toda la carga
                avisos.append(f"Omitido {r.nombre}: {e}")
                descartes.append((r.nombre, "archivo", str(e)[:160]))
                continue
            cache.guardar_npz(clave, **z)
        else:
            aciertos += 1
        usados.append(r)
        if por_archivo:                                      # un vector por grabación: estadísticos de sus ventanas
            Xa = np.asarray(z["X"], dtype=np.float32)
            fila = np.r_[Xa.mean(0), Xa.std(0), float(z["dur"])].astype(np.float32)
            esp = r.meta.get("especie")
            bloques.append(dict(X=fila[None, :], hover=np.array([f"{r.nombre} · {float(z['dur']):.1f} s" + (f" · {esp}" if esp else "")]),
                                ref=[(r.nombre, 0.0)]))
        else:
            bloques.append(dict(X=z["X"], hover=np.array([f"{r.nombre} · {t:.1f} s" for t in z["t"]]),
                                ref=[(r.nombre, float(t)) for t in z["t"]]))
        if (i + 1) % cfg.lote == 0:
            gc.collect()                                   # procesamiento por lotes: libera memoria entre lotes
    if not bloques:
        from core.errors import ErrorDatos

        raise ErrorDatos("No se pudo leer ningún archivo de audio", "; ".join(avisos[:3]))
    recursos = usados
    n = cfg.audio_n_mfcc
    nombres = [f"mfcc{j}_media" for j in range(n)] + [f"mfcc{j}_std" for j in range(n)] + ["centroide", "zcr", "rms"]
    if por_archivo:
        nombres = [f"{c}·media" for c in nombres] + [f"{c}·desv" for c in nombres] + ["duracion_s"]
    X = np.vstack([b["X"] for b in bloques]).astype(np.float32)
    desc = (f"{len(X)} grabaciones completas · {X.shape[1]} descriptores (media y desviación de sus ventanas + duración)" if por_archivo
            else f"{len(X)} ventanas de {cfg.audio_ventana_s} s (salto {cfg.audio_salto_s} s)")
    d = ensamblar(TIPO, recursos, bloques, nombres, cfg, [VERSION, *params, "archivo" if por_archivo else "fragmentos"], desc, True)
    d.avisos, d.aciertos_cache, d.descartes, d.granularidad = avisos, aciertos, descartes, "archivo" if por_archivo else "fragmentos"
    return d
