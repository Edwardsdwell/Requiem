"""Gestor de tareas en segundo plano con progreso y cancelación.

Las operaciones largas (descargas, extracción de características, algoritmos) se ejecutan en
un `ThreadPoolExecutor` con un número limitado de trabajadores, fuera del hilo de la interfaz.
La cancelación es cooperativa: el código llama a `ctx.punto()` entre archivos, lotes o iteraciones.
"""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .errors import Cancelado, describir, logger


def memoria_mb() -> float | None:
    """Memoria residente del proceso (MB), si `psutil` está instalado (opcional)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        try:
            import resource, sys

            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return r / (2**20 if sys.platform == "darwin" else 2**10)
        except Exception:
            return None


class Contexto:
    """Se pasa a las funciones largas para informar del progreso y comprobar la cancelación."""

    def __init__(self):
        self._cancelar = threading.Event()
        self.progreso, self.mensaje = 0.0, ""

    def cancelar(self):
        self._cancelar.set()

    @property
    def cancelado(self) -> bool:
        return self._cancelar.is_set()

    def punto(self, frac: float | None = None, mensaje: str | None = None):
        if frac is not None:
            self.progreso = float(min(max(frac, 0.0), 1.0))
        if mensaje is not None:
            self.mensaje = mensaje
        if self._cancelar.is_set():
            raise Cancelado()


@dataclass
class Tarea:
    id: str
    nombre: str
    ctx: Contexto = field(default_factory=Contexto)
    estado: str = "en cola"          # en cola | ejecutando | terminada | fallida | cancelada
    resultado: object = None
    error: dict | None = None
    t0: float = field(default_factory=time.time)
    t1: float | None = None
    mem0: float | None = None
    mem_pico: float | None = None

    @property
    def duracion(self) -> float:
        return (self.t1 or time.time()) - self.t0

    @property
    def activa(self) -> bool:
        return self.estado in ("en cola", "ejecutando")


class GestorTareas:
    def __init__(self, max_trabajadores: int = 2, cfg=None):
        self.pool = ThreadPoolExecutor(max_workers=max(1, max_trabajadores), thread_name_prefix="ns")
        self.tareas: dict[str, Tarea] = {}
        self.log = logger(cfg)

    def lanzar(self, nombre: str, funcion, *args, **kw) -> Tarea:
        t = Tarea(uuid.uuid4().hex[:10], nombre)
        self.tareas[t.id] = t

        def _run():
            if t.ctx.cancelado:
                t.estado, t.t1 = "cancelada", time.time()
                return
            t.estado, t.mem0 = "ejecutando", memoria_mb()
            try:
                t.resultado = funcion(*args, ctx=t.ctx, **kw)
                t.estado = "terminada"
            except Cancelado:
                t.estado = "cancelada"
            except BaseException as e:  # noqa: BLE001 - se informa por la interfaz
                t.estado, t.error = "fallida", describir(e)
                self.log.exception("Tarea %s falló", nombre)
            finally:
                t.t1 = time.time()
                m = memoria_mb()
                t.mem_pico = None if m is None else max(m, t.mem0 or 0)
                t.ctx.progreso = 1.0 if t.estado == "terminada" else t.ctx.progreso

        self.pool.submit(_run)
        return t

    def ejecutar_sync(self, nombre: str, funcion, *args, **kw) -> Tarea:
        """Misma semántica que `lanzar`, pero espera el resultado (scripts y pruebas)."""
        t = self.lanzar(nombre, funcion, *args, **kw)
        while t.activa:
            time.sleep(0.01)
        return t

    def activas(self) -> list[Tarea]:
        return [t for t in self.tareas.values() if t.activa]

    def cancelar(self, id_: str):
        if id_ in self.tareas:
            self.tareas[id_].ctx.cancelar()

    def purgar(self, max_historial: int = 30):
        viejas = [t for t in self.tareas.values() if not t.activa]
        for t in sorted(viejas, key=lambda x: x.t0)[:-max_historial]:
            self.tareas.pop(t.id, None)
