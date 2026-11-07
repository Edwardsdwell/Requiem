"""Widgets web propios (HTML/CSS/JS en `visualization/web/`) incrustados en Streamlit .

`/*@archivo*/` dentro de un widget se sustituye por el contenido de ese archivo y `__DATOS__` por los datos en JSON.
Streamlit se importa solo al mostrar un widget. Los widgets se pintan únicamente cuando la pestaña
que los contiene se abre (carga visual diferida) y respetan `prefers-reduced-motion`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

WEB = Path(__file__).parent / "web"


def _leer(nombre: str) -> str:
    return (WEB / nombre).read_text(encoding="utf-8")


def html(nombre: str, datos=None) -> str:
    s = re.sub(r"/\*@(\S+?)\*/", lambda m: _leer(m[1]), _leer(nombre))
    return s.replace("__DATOS__", json.dumps(datos or {}, ensure_ascii=False).replace("</", "<\\/"))


def mostrar(nombre: str, datos=None, alto: int = 400) -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    contenido = html(nombre, datos)
    if hasattr(st, "iframe"):
        try:
            st.iframe(contenido, height=max(alto, 1))
            return
        except Exception:  # noqa: BLE001 - firma distinta entre versiones
            pass
    components.html(contenido, height=alto)
