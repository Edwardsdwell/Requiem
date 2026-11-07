"""Estructuras comunes: datos de entrada, parámetros y resultados.

No dependen de Streamlit, Tkinter ni de ningún componente web. Plotly solo aparece
dentro de `Res.figuras` como objetos opacos.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Datos:
    """Conjunto procesado que entregan todos los cargadores."""

    X: object                        # (n, p): ndarray o matriz dispersa (TF-IDF)
    nombres: list                    # nombre de cada columna 
    origen: np.ndarray               # archivo de origen de cada muestra 
    hover: np.ndarray                # texto al pasar el cursor
    escalar: bool = True             # ¿estandarizar antes de los algoritmos? (
    mapa: np.ndarray | None = None   # (lat, lon) por muestra (solo coordenadas)
    centro: tuple | None = None      # (lat0, lon0) de la proyección local
    descripcion: str = ""
    tipo: str = ""                   # "img" | "audio" | "texto" | "geo"
    ref: list | None = None          # qué es cada muestra: (archivo, y, x) · (archivo, t) · texto · (lat, lon)
    meta: list | None = None         # procedencia por muestra: dict(event_id, sha256, fuente…) 
    recursos: dict = field(default_factory=dict)   # nombre -> ruta del archivo original (solo lectura)
    huella: str = ""                 # identifica el contenido + configuración (caché y trazabilidad)
    avisos: list = field(default_factory=list)     # advertencias de carga (p. ej. límites de la proyección)
    proyeccion: str = ""             # "local" | "esfera" (solo coordenadas)
    descartes: list = field(default_factory=list)  # (recurso, posición, motivo) de lo que no pudo usarse: nada se pierde en silencio
    granularidad: str = ""           # "fragmentos" (parches / ventanas) | "archivo" (un vector por imagen o audio)

    # ------------------------------------------------------------------
    def tomar(self, idx) -> "Datos":
        idx = np.asarray(idx)
        sel = lambda v: None if v is None else (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx])
        return Datos(self.X[idx], self.nombres, self.origen[idx], self.hover[idx], self.escalar, sel(self.mapa),
                     self.centro, self.descripcion, self.tipo, sel(self.ref), sel(self.meta), self.recursos, self.huella,
                     self.avisos, self.proyeccion, self.descartes, self.granularidad)

    def con_columnas(self, cols) -> "Datos":
        cols = np.asarray(cols)
        d = self.tomar(np.arange(self.X.shape[0]))
        d.X, d.nombres = self.X[:, cols], [self.nombres[i] for i in cols]
        return d

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def es_dispersa(self) -> bool:
        return hasattr(self.X, "tocsr")

    def etiquetas_externas(self, campo: str) -> np.ndarray | None:
        """Etiquetas conocidas por muestra (p. ej. «clase» Aves/Amphibia o «especie»), tomadas de la procedencia.
        Sirven solo para **evaluar** los grupos encontrados (ARI/NMI), nunca para entrenar."""
        if not self.meta:
            return None
        v = [(m or {}).get(campo) for m in self.meta]
        if sum(x is not None for x in v) < max(4, 0.8 * len(v)) or len({x for x in v if x is not None}) < 2:
            return None
        return np.array(["(sin dato)" if x is None else str(x) for x in v])

    def a_latlon(self, xy: np.ndarray) -> np.ndarray:
        """Inversa de la proyección equirectangular local (válida en áreas pequeñas.)"""
        if self.proyeccion == "esfera":              # xyz (km) sobre la esfera terrestre -> lat/lon
            R = 6371.0
            x, y, z = xy[:, 0], xy[:, 1], xy[:, 2]
            return np.c_[np.degrees(np.arcsin(np.clip(z / np.maximum(np.sqrt(x**2 + y**2 + z**2), 1e-9), -1, 1))), np.degrees(np.arctan2(y, x))]
        lat0, lon0 = self.centro
        return np.c_[lat0 + xy[:, 1] / 110.57, lon0 + xy[:, 0] / (111.32 * np.cos(np.radians(lat0)))]


@dataclass
class Param:
    clave: str
    etiqueta: str
    defecto: object                 # int -> slider · float -> número · str -> lista (con `opciones`)
    min: float | None = None
    max: float | None = None
    paso: float | None = None
    opciones: list | None = None
    si: tuple | None = None         # (clave, valor): visible solo si ese parámetro tiene ese valor
    tipos: tuple | None = None      # visible solo para estos tipos de datos


@dataclass
class Res:
    informe: str                                   # markdown
    Z: np.ndarray | None = None                    # proyección a graficar (n, 2-3+)
    ejes: list = field(default_factory=list)
    etiquetas: np.ndarray | None = None            # grupos (-1 = ruido)
    centros: np.ndarray | None = None              # en el espacio de Z
    figuras: list = field(default_factory=list)    # figuras Plotly; una tupla se muestra en columnas
    tabla: object = None                           # pandas.DataFrame
    kpis: dict = field(default_factory=dict)       # cifras clave
    extra: dict = field(default_factory=dict)      # metadatos libres (p. ej. ayudas de reproducibilidad)


@dataclass
class Archivo:
    """Archivo elegido para una comparativa (imagen o audio concreto): nombre visible + ruta real."""

    name: str
    ruta: object

    def __str__(self):
        return self.name
