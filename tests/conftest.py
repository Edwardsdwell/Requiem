import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    return Config(store=str(tmp_path / "store"), db=str(tmp_path / "store" / "t.db"), pausa_descarga_s=0, espera_base_s=0.01,
                  permitir_privadas=True, max_muestras=3000)


@pytest.fixture
def motor(cfg):
    from core.engine import Motor

    return Motor(cfg)


def escribir_wav(ruta: Path, frec: float, seg: float = 4.0, sr: int = 22050):
    t = np.arange(int(seg * sr)) / sr
    y = 0.4 * np.sin(2 * np.pi * frec * t) + 0.05 * np.random.default_rng(int(frec)).normal(size=len(t))
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())


@pytest.fixture
def archivos(tmp_path):
    """Datos sintéticos de los cuatro tipos con estructura conocida (3 grupos)."""
    from PIL import Image

    d = tmp_path / "src"
    d.mkdir()
    for i, f in enumerate((220, 880, 3000)):
        escribir_wav(d / f"tono{i}.wav", f)
    rng = np.random.default_rng(0)
    for i, base in enumerate(((220, 30, 30), (30, 220, 30), (30, 30, 220))):
        a = np.clip(np.array(base) + rng.normal(0, 25, (128, 128, 3)), 0, 255).astype("uint8")
        Image.fromarray(a).save(d / f"img{i}.png")
    temas = {"mar": "el barco navega por el mar azul con olas y marineros en la costa lejana",
             "cocina": "la receta lleva harina huevos azucar y horno para hornear el pastel dulce",
             "codigo": "el programa compila funciones variables y algoritmos del lenguaje de programacion"}
    parrafos = []
    for k, t in temas.items():
        parrafos += [f"{t} numero {j} sobre {k} {k} {k} y mas detalles de {k} para el texto de prueba" for j in range(12)]
    (d / "texto.txt").write_text("\n\n".join(parrafos), encoding="utf-8")
    pts = np.vstack([rng.normal(c, 0.05, (25, 2)) for c in ((10, 20), (10.8, 20.9), (9.2, 19.1))])
    (d / "puntos.csv").write_text("latitud,longitud,nombre\n" + "\n".join(f"{a:.5f},{b:.5f},p{i}" for i, (a, b) in enumerate(pts)), encoding="utf-8")
    return d
