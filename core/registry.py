"""Registro central de algoritmos y comparativas.

Cada entrada guarda *metadatos* (parámetros, tipos admitidos, restricciones) y la función como
referencia perezosa «paquete.módulo:función». Así la interfaz puede listar y configurar un
algoritmo sin importar scikit-learn, Plotly ni UMAP hasta que se ejecuta.

Para añadir un método sin tocar el núcleo:

    from core.registry import registrar
    from core.results import Param
    registrar("Mi método", "mi_paquete.mi_modulo:funcion",
              [Param("k", "Grupos", 3, 2, 10)], tipos=("audio", "img"),
              descripcion="…", min_muestras=10)

La función recibe `(datos, X, params)` y devuelve un `core.results.Res`.
Para retirar uno: `retirar("Mi método")`.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable

from .results import Param

TIPOS_TODOS = ("img", "audio", "texto", "geo")


@dataclass
class Entrada:
    nombre: str
    funcion: str | Callable          # "modulo:funcion" o un callable
    params: list
    tipos: tuple = TIPOS_TODOS
    descripcion: str = ""
    clase: str = "algoritmo"         # "algoritmo" | "comparativa"
    min_muestras: int = 5            
    min_variables: int = 1           
    acepta_dispersa: bool = False    # puede recibir TF-IDF sin densificar 
    archivo: tuple = ()              # tipos para los que se pide elegir un archivo concreto
    supervisado: bool = False        #  se marca expresamente
    notas: str = ""

    def resolver(self) -> Callable:
        if callable(self.funcion):
            return self.funcion
        mod, _, fn = self.funcion.partition(":")
        return getattr(importlib.import_module(mod), fn)


_REG: dict[str, Entrada] = {}


def _clave(nombre: str, clase: str) -> str:
    return nombre if clase == "algoritmo" else f"cmp::{nombre}"


def registrar(nombre, funcion, params, tipos=TIPOS_TODOS, descripcion="", clase="algoritmo", **kw) -> Entrada:
    e = Entrada(nombre, funcion, list(params), tuple(tipos), descripcion, clase, **kw)
    _REG[_clave(nombre, clase)] = e
    return e


def retirar(nombre: str, clase: str = "algoritmo") -> None:
    _REG.pop(_clave(nombre, clase), None)


def algoritmos(tipo: str | None = None) -> dict[str, Entrada]:
    _asegurar_catalogo()
    return {e.nombre: e for k, e in _REG.items() if e.clase == "algoritmo" and (tipo is None or tipo in e.tipos)}


def comparativas(tipo: str | None = None) -> dict[str, Entrada]:
    _asegurar_catalogo()
    return {e.nombre: e for k, e in _REG.items() if e.clase == "comparativa" and (tipo is None or tipo in e.tipos)}


def obtener(nombre: str, clase: str = "algoritmo") -> Entrada:
    _asegurar_catalogo()
    return _REG[_clave(nombre, clase)]


def valores_por_defecto(e: Entrada, tipo: str) -> dict:
    """Parámetros visibles con su valor inicial (respeta `si` y `tipos`)."""
    v: dict = {}
    for p in e.params:
        if (p.tipos and tipo not in p.tipos) or (p.si and v.get(p.si[0]) != p.si[1]):
            continue
        v[p.clave] = p.defecto
    return v


def validar_parametros(e: Entrada, tipo: str, vals: dict) -> dict:
    """Completa y acota los parámetros según su definición. La visibilidad (`si`, `tipos`) se
    evalúa con los valores elegidos por el usuario, en el mismo orden en que los muestra la interfaz."""
    out: dict = {}
    for p in e.params:
        if (p.tipos and tipo not in p.tipos) or (p.si and out.get(p.si[0]) != p.si[1]):
            continue
        v = vals.get(p.clave, p.defecto)
        if p.opciones:
            v = v if v in p.opciones else p.defecto
        elif p.min is not None and p.max is not None:
            v = min(max(v, p.min), p.max)
            v = float(v) if isinstance(p.defecto, float) else int(v)
        out[p.clave] = v
    for k, v in vals.items():       # parámetros especiales (archivo seleccionado, etc.)
        out.setdefault(k, v)
    return out


_cargado = False


def _asegurar_catalogo() -> None:
    global _cargado
    if not _cargado:
        _cargado = True
        from ml import catalogo  # noqa: F401  (registra las 12 opciones y las 13 comparativas)
