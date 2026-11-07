"""Mapas geográficos y puntos del explorador de mapa."""
from __future__ import annotations

import numpy as np


def zoom_para(lat: np.ndarray, lon: np.ndarray) -> float:
    return float(np.clip(min(np.log2(527 / max(np.ptp(lon), 1e-3)), np.log2(351 / max(np.ptp(lat), 1e-3))), 1, 16))


def centro_de(lat, lon) -> dict:
    return dict(lat=float(np.mean(lat)), lon=float(np.mean(lon)))
