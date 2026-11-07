"""Gestión de errores centralizada."""
from __future__ import annotations

import logging
import traceback
from logging.handlers import RotatingFileHandler


class ErrorApp(Exception):
    """Error esperable con un código y un mensaje para el usuario."""

    def __init__(self, mensaje: str, codigo: str = "E_GENERAL", detalle: str = ""):
        super().__init__(mensaje)
        self.codigo, self.mensaje, self.detalle = codigo, mensaje, detalle


class ErrorDatos(ErrorApp):
    def __init__(self, mensaje, detalle=""):
        super().__init__(mensaje, "E_DATOS", detalle)


class ErrorValidacion(ErrorApp):
    def __init__(self, mensaje, detalle=""):
        super().__init__(mensaje, "E_VALIDACION", detalle)


class ErrorDescarga(ErrorApp):
    def __init__(self, mensaje, detalle=""):
        super().__init__(mensaje, "E_DESCARGA", detalle)


class Cancelado(Exception):
    """Se lanza al cancelar una tarea en un punto de control."""


def logger(cfg=None) -> logging.Logger:
    log = logging.getLogger("nosupervisado")
    if not log.handlers:
        log.setLevel(logging.INFO)
        if cfg is not None:
            try:
                h = RotatingFileHandler(cfg.subdir("results") / "app.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
                h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                log.addHandler(h)
            except OSError:
                pass
        if not log.handlers:
            log.addHandler(logging.NullHandler())
    return log


def describir(e: BaseException) -> dict:
    """Estructura uniforme para mostrar un error: código, mensaje y traza resumida."""
    if isinstance(e, ErrorApp):
        return dict(codigo=e.codigo, mensaje=e.mensaje, detalle=e.detalle)
    tb = "".join(traceback.format_exception_only(type(e), e)).strip()
    return dict(codigo="E_INESPERADO", mensaje=f"{type(e).__name__}: {e}", detalle=tb)
