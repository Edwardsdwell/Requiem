"""Núcleo central: coordina adquisición, carga, preparación, algoritmos y comparativas.

`Motor` no depende de Streamlit: la misma clase sirve a la interfaz web, a scripts y a las pruebas. 
Las operaciones largas se lanzan como tareas en segundo plano con progreso y cancelación.
"""
from __future__ import annotations

import io
import json
import time
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import metadata

import numpy as np
from pathlib import Path

from config import CFG, Config
from data import db, loaders
from data.storage import EXT, Almacen, detectar_tipo, sha256_archivo
from . import registry
from .cache import Cache, hash_obj
from .errors import ErrorDatos, ErrorValidacion, logger
from .results import Archivo, Datos, Res
from .tasks import GestorTareas, memoria_mb

VERSION_MOTOR = "1"


def _versiones() -> dict:
    out = {}
    for p in ("numpy", "scipy", "pandas", "scikit-learn", "plotly", "librosa", "umap-learn", "Pillow", "streamlit"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "no instalado"
    return out


@dataclass
class Salida:
    """Resultado de una ejecución, con lo necesario para mostrarlo, exportarlo y reproducirlo."""

    datos: Datos
    res: Res
    run_id: str
    clave: str
    nombre: str
    clase: str
    tipo: str
    params: dict
    prep: dict
    desde_cache: bool = False
    duracion_s: float = 0.0
    memoria_mb: float | None = None
    avisos: list = field(default_factory=list)


class Motor:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.almacen = Almacen(cfg)
        self.tareas = GestorTareas(cfg.max_trabajadores, cfg)
        self.cache_resultados = Cache(cfg.subdir("cache", "results"))
        self.cache_prep = Cache(cfg.subdir("cache", "prep"))
        self.log = logger(cfg)
        db.init(cfg).close()
        # PostgreSQL es la fuente de la app; Config sin URL mantiene aisladas las pruebas.
        self.registros_fuente = None
        if cfg.database_url:
            con = db.connect(cfg)
            try:
                from data.postgres_source import sincronizar
                self.registros_fuente = sincronizar(cfg, self.almacen, con)
            finally:
                con.close()

    # ------------------------------------------------------------------ base de datos
    @contextmanager
    def bd(self):
        """Conexión propia por operación: segura entre hilos."""
        con = db.connect(self.cfg)
        try:
            yield con
        finally:
            con.close()

    def resumen(self) -> dict:
        with self.bd() as con:
            return db.resumen(con)

    def recursos(self, tipo: str):
        with self.bd() as con:
            return loaders.listar(tipo, con, self.almacen)

    # ------------------------------------------------------------------ adquisición local y remota 
    def _registrar_documento(self, con, tipo, nombre, ruta: Path, sha: str, fuente="local", url=None):
        con.execute("INSERT OR IGNORE INTO documents(doc_id,kind,name,local_path,sha256,size_bytes,source,origin_url,created_utc) VALUES(?,?,?,?,?,?,?,?,?)",
                    ("doc:" + sha[:16], tipo, nombre, self.almacen.relativa(ruta), sha, ruta.stat().st_size, fuente, url, db.now()))

    def ingerir(self, elementos, tipo: str | None = None, ctx=None) -> dict:
        """`elementos`: lista de (nombre, bytes | Path). Detecta el tipo si no se indica, **valida el contenido**
        (no vacío, formato real, legible) y guarda por SHA-256. Cada rechazo lleva su motivo."""
        from data.validacion import validar_archivo

        inf = dict(agregados=[], duplicados=[], rechazados=[], avisos=[])
        if len(elementos) > self.cfg.max_archivos_carga:
            raise ErrorValidacion(f"Demasiados archivos ({len(elementos)}); el límite es {self.cfg.max_archivos_carga}")
        with self.bd() as con:
            for i, (nombre, contenido) in enumerate(elementos):
                if ctx is not None:
                    ctx.punto(i / max(len(elementos), 1), f"Incorporando {nombre}")
                tmp = None
                try:
                    if isinstance(contenido, (bytes, bytearray)):
                        tmp = self.almacen.nuevo_temporal(Path(str(nombre)).suffix.lower() or ".part")
                        tmp.write_bytes(bytes(contenido))
                        origen = tmp
                    else:
                        origen = Path(contenido)
                    t = tipo or (self._tipo_bytes(nombre, tmp) if tmp else detectar_tipo(origen))
                    if t is None:
                        raise ErrorValidacion(f"{nombre}: formato no reconocido")
                    self.almacen.validar_nombre(str(nombre), t)
                    v = validar_archivo(origen, t)
                    if not v.ok:
                        raise ErrorValidacion(v.motivo)
                    if v.motivo:
                        inf["avisos"].append((Path(str(nombre)).name, v.motivo))
                    ruta, sha, dup = self.almacen.ingerir(origen, t, nombre)
                    self._registrar_documento(con, t, Path(str(nombre)).name, ruta, sha)
                    (inf["duplicados"] if dup else inf["agregados"]).append((Path(str(nombre)).name, t))
                except Exception as e:  # noqa: BLE001 - se informa por archivo, el resto sigue
                    inf["rechazados"].append((str(nombre), getattr(e, "mensaje", str(e))))
                finally:
                    if tmp is not None:
                        tmp.unlink(missing_ok=True)
            db.auditar(con, "ingerir", {k: len(v) for k, v in inf.items()}, len(elementos))
            con.commit()
        return inf

    def _tipo_bytes(self, nombre: str, tmp: Path):
        """Detecta el tipo de un archivo subido: usa la extensión del nombre original y, si es ambigua, el contenido."""
        ext = Path(nombre).suffix.lower()
        for t in ("audio", "img"):
            if ext in EXT[t]:
                return t
        return detectar_tipo(tmp)                               # el temporal ya tiene la extensión original

    def ingerir_carpeta(self, carpeta: str | Path, recursivo: bool = True, tipo: str | None = None, ctx=None) -> dict:
        from data.validacion import limpiar_ruta_usuario

        carpeta = limpiar_ruta_usuario(carpeta)                 # admite comillas y rutas de Windows pegadas
        if not carpeta.is_dir():
            raise ErrorValidacion(f"No es una carpeta: {carpeta}")
        patron = carpeta.rglob("*") if recursivo else carpeta.glob("*")
        archivos = [p for p in patron if p.is_file() and not p.name.startswith(".") and not p.is_symlink()]
        archivos = [p for p in archivos if (detectar_tipo(p) is not None) and (tipo is None or detectar_tipo(p) == tipo)]
        return self.ingerir([(p.name, p) for p in sorted(archivos)[: self.cfg.max_archivos_carga]], tipo, ctx)

    def ingerir_url(self, url: str, tipo: str, ctx=None) -> dict:
        """Recurso remoto autorizado (audio, imagen, texto o geografía): validación de URL, límites y hash."""
        from data.download import Descargador

        r = Descargador(self.cfg, self.almacen).con_reintentos(url, tipo, ctx)
        with self.bd() as con:
            self._registrar_documento(con, tipo, r.ruta.name, r.ruta, r.sha256, "remoto", url)
            con.commit()
        return dict(agregados=[] if r.duplicado else [(r.ruta.name, tipo)], duplicados=[(r.ruta.name, tipo)] if r.duplicado else [], rechazados=[])

    def importar_legado(self, carpeta: str | Path, medios: bool = True, ctx=None) -> dict:
        from scripts.importar_legado import importar

        return importar(Path(str(carpeta)), self.cfg, medios, verbose=False, ctx=ctx)

    def verificar_integridad(self, ctx=None) -> dict:
        """Comprueba que cada registro válido tenga su archivo íntegro y legible (ver `data.integridad`)."""
        from data.integridad import verificar

        with self.bd() as con:
            return verificar(con, self.almacen, ctx)

    def lanzar_verificacion(self):
        return self.tareas.lanzar("Verificar integridad", lambda ctx: self.verificar_integridad(ctx))

    def inventario(self):
        """Todos los registros, por modalidad, clasificados en válido / inválido / pendiente con su motivo."""
        from data.inventario import inventario

        with self.bd() as con:
            return inventario(con, self.almacen)

    def informe_calidad(self, ctx=None) -> dict:
        """Informe de calidad de datos por modalidad (ver `data.inventario.informe_calidad`)."""
        from data.inventario import informe_calidad

        with self.bd() as con:
            return informe_calidad(con, self.almacen, self.cfg, ctx)

    def lanzar_informe_calidad(self):
        return self.tareas.lanzar("Informe de calidad", lambda ctx: self.informe_calidad(ctx))

    def auditoria(self, limite: int = 200):
        import pandas as pd

        with self.bd() as con:
            return pd.DataFrame([dict(f) for f in con.execute("SELECT * FROM auditoria ORDER BY utc DESC LIMIT ?", (limite,))])

    # ------------------------------------------------------------------ descarga y recolección 
    def descargar_pendientes(self, tipo=None, limite=None, ctx=None) -> dict:
        from data.download import Descargador

        with self.bd() as con:
            return Descargador(self.cfg, self.almacen).descargar_pendientes(con, tipo, limite, ctx)

    def lanzar_descarga(self, tipo=None, limite=None):
        return self.tareas.lanzar("Descarga de pendientes", lambda ctx: self.descargar_pendientes(tipo, limite, ctx))

    # ------------------------------------------------------------------ carga y calidad
    def cargar(self, tipo: str, recursos, ctx=None, max_muestras=None) -> Datos:
        return loaders.cargar(tipo, list(recursos), self.cfg, ctx, max_muestras)

    def preparar(self, d: Datos, prep=None):
        from ml.preprocessing import preparar

        return preparar(d, prep, self.cache_prep)

    def calidad(self, tipo, recursos, max_muestras=None, ctx=None):
        from ml.preprocessing import calidad

        return calidad(self.cargar(tipo, recursos, ctx, max_muestras))

    def precalcular(self, tipo, recursos, ctx=None):
        """Extrae y guarda las características (sin ejecutar algoritmos): descarga ≠ análisis."""
        d = self.cargar(tipo, recursos, ctx)
        return d.descripcion

    # ------------------------------------------------------------------ ejecución de algoritmos y comparativas
    def _clave(self, d: Datos, nombre, clase, params, prep_info, extra=None) -> str:
        return hash_obj(VERSION_MOTOR, d.huella, d.n, d.X.shape[1], nombre, clase, params, prep_info, self.cfg.semilla,
                        {k: v for k, v in _versiones().items() if k in ("numpy", "scikit-learn", "plotly", "scipy")}, extra)

    def ejecutar(self, tipo: str, recursos: list, nombre: str, params: dict | None = None, prep=None,
                 clase: str = "algoritmo", archivo=None, max_muestras: int | None = None, ctx=None, usar_cache: bool = True) -> Salida:
        """Flujo completo: carga → preparación → validación → algoritmo → registro. Reutiliza resultados idénticos."""
        from ml import preprocessing as P

        entrada = registry.obtener(nombre, clase)
        params = registry.validar_parametros(entrada, tipo, dict(params or {}))
        if ctx is not None:
            ctx.punto(0.0, "Cargando datos")
        d = self.cargar(tipo, recursos, ctx, max_muestras)
        d, X, info = self.preparar(d, prep)
        err, avisos = P.validar(entrada, d, X, params)
        if err:
            raise ErrorValidacion(" ".join(err))
        clave_par = dict(params)
        if archivo is not None:
            rec = archivo if hasattr(archivo, "sha256") else next((r for r in recursos if r.nombre == archivo), None)
            if rec is None:
                raise ErrorValidacion("El archivo elegido no está entre los seleccionados")
            params["archivo"] = Archivo(rec.nombre, rec.ruta)
            clave_par["archivo"] = rec.sha256
        elif entrada.archivo and tipo in entrada.archivo:
            raise ErrorValidacion("Esta comparativa necesita que elija un archivo")
        clave = self._clave(d, nombre, clase, clave_par, info)
        return self._correr(clave, d, X, info, nombre, clase, tipo, params, entrada, avisos, ctx, usar_cache,
                            lambda: entrada.resolver()(d, X if entrada.acepta_dispersa else P.densificar(X, self.cfg.max_denso_mb), params) if clase == "algoritmo"
                            else entrada.resolver()(self._denso(d), X if entrada.acepta_dispersa else P.densificar(X, self.cfg.max_denso_mb), params, d.recursos),
                            preparacion=info)

    def _denso(self, d: Datos) -> Datos:
        from ml.preprocessing import densificar

        if not d.es_dispersa:
            return d
        d2 = d.tomar(range(d.n))
        d2.X = densificar(d.X, self.cfg.max_denso_mb)
        return d2

    def _correr(self, clave, d, X, info, nombre, clase, tipo, params, entrada, avisos, ctx, usar_cache, fn, preparacion=None) -> Salida:
        from ml import preprocessing as P

        guardado = self.cache_resultados.leer_pkl(clave) if usar_cache else None
        if guardado is not None:
            d_c, res_c = guardado
            run = self._registrar_run(tipo, clase, nombre, params, d_c, 0.0, memoria_mb(), "cache", clave, "Resultado reutilizado de la caché", res_c.kpis)
            return Salida(d_c, res_c, run, clave, nombre, clase, tipo, _limpio(params), preparacion or info, True, 0.0, memoria_mb(), avisos)
        t0, tk, tm = time.time(), P._SEMILLA.set(self.cfg.semilla), P._CTX.set(ctx)
        try:
            if ctx is not None:
                ctx.punto(0.15, f"Ejecutando «{nombre}»")
            res = fn()
        except BaseException as e:
            self._registrar_run(tipo, clase, nombre, params, d, time.time() - t0, memoria_mb(), "cancelada" if type(e).__name__ == "Cancelado" else "fallida", None, str(e)[:300])
            raise
        finally:
            P._SEMILLA.reset(tk)
            P._CTX.reset(tm)
        dur = time.time() - t0
        d_g = d.tomar(range(d.n)) if d.es_dispersa is False else d
        try:
            self.cache_resultados.guardar_pkl(clave, (d_g, res))
        except Exception:  # noqa: BLE001 - la caché es opcional
            self.log.warning("No se pudo guardar el resultado en caché", exc_info=True)
        run = self._registrar_run(tipo, clase, nombre, params, d, dur, memoria_mb(), "correcta", clave, "", res.kpis)
        return Salida(d, res, run, clave, nombre, clase, tipo, _limpio(params), preparacion or info, False, dur, memoria_mb(), avisos)

    def ejecutar_comparacion(self, tipo: str, recursos: list, modo: str, prep=None, algoritmos=None, params_por_alg=None,
                             algoritmo=None, parametro=None, valores=None, base=None, max_muestras=None, ctx=None,
                             estabilidad: bool = True, repeticiones: int = 3) -> Salida:
        """Varios algoritmos sobre el mismo conjunto (ranking multicriterio), un algoritmo con varios valores de un
        parámetro (sensibilidad) o la estabilidad de un algoritmo ante submuestreo."""
        from ml import comparisons as Cm

        d = self.cargar(tipo, recursos, ctx, max_muestras)
        d, X, info = self.preparar(d, prep)
        if modo == "algoritmos":
            if not algoritmos or len(algoritmos) < 2:
                raise ErrorValidacion("Elija al menos dos algoritmos")
            nombre, par = "Comparativa de algoritmos", dict(algoritmos=sorted(algoritmos), por=params_por_alg or {}, estabilidad=estabilidad, reps=repeticiones)
            fn = lambda: Cm.comparar_algoritmos(d, X, list(algoritmos), params_por_alg, ctx, estabilidad, repeticiones)
        elif modo == "estabilidad":
            if not algoritmo:
                raise ErrorValidacion("Indique el algoritmo")
            nombre, par = "Estabilidad", dict(algoritmo=algoritmo, base=base or {}, reps=repeticiones)
            fn = lambda: Cm.estabilidad(d, X, algoritmo, base, max(repeticiones, 3), ctx=ctx)
        else:
            if not (algoritmo and parametro and valores):
                raise ErrorValidacion("Indique el algoritmo, el parámetro y los valores")
            nombre, par = "Sensibilidad de parámetros", dict(algoritmo=algoritmo, parametro=parametro, valores=list(valores), base=base or {},
                                                              estabilidad=estabilidad, reps=repeticiones)
            fn = lambda: Cm.comparar_parametros(d, X, algoritmo, parametro, list(valores), base, ctx, estabilidad, repeticiones)
        entrada = registry.Entrada(nombre, fn, [], tuple(registry.TIPOS_TODOS), clase="comparativa")
        clave = self._clave(d, nombre, "multi", par, info)
        return self._correr(clave, d, X, info, nombre, "comparativa", tipo, par, entrada, list(d.avisos or []), ctx, True, fn, info)

    # ------------------------------------------------------------------ recomendaciones y estimaciones (sec. 3 y 5)
    def recomendar(self, tipo, recursos, nombre, params=None, prep=None, max_muestras=None) -> dict:
        from ml.recomendaciones import recomendar

        d, X, _ = self.preparar(self.cargar(tipo, recursos, None, max_muestras), prep)
        return recomendar(nombre, d, X, params)

    def estimar(self, tipo: str, nombre: str, n: int, p: int) -> dict:
        """Tiempo y memoria aproximados antes de ejecutar: tiempo por muestra de ejecuciones anteriores del mismo
        algoritmo (si las hay) y memoria de las matrices que se construirán."""
        with self.bd() as con:
            filas = con.execute("SELECT duracion_s, n_muestras FROM runs WHERE nombre=? AND tipo=? AND estado='correcta' AND n_muestras>0 "
                                "ORDER BY created_utc DESC LIMIT 10", (nombre, tipo)).fetchall()
        t = None
        if filas:
            por = np.median([f["duracion_s"] / f["n_muestras"] for f in filas])
            exp = 2.0 if "jerárquica" in nombre or "t-SNE" in nombre else 1.0          # coste ~ n² para algunos métodos
            ref = np.median([f["n_muestras"] for f in filas])
            t = float(por * ref * (n / max(ref, 1)) ** exp)
        mem = n * p * 8 / 2**20 * 3                                                     # datos + copia escalada + proyección
        if "jerárquica" in nombre:
            mem += min(n * (n - 1) / 2 * 8 / 2**20, self.cfg.max_distancias_mb)
        if "DBSCAN" in nombre or "OPTICS" in nombre or "t-SNE" in nombre or "UMAP" in nombre:
            mem += n * 50 * 16 / 2**20                                                   # grafos de vecinos
        return dict(segundos=t, memoria_mb=mem, basado_en=len(filas))

    def lanzar_ejecucion(self, tipo, recursos, nombre, params=None, prep=None, clase="algoritmo", archivo=None, max_muestras=None):
        """Ejecuta en segundo plano; la interfaz consulta el progreso y puede cancelar."""
        return self.tareas.lanzar(nombre, lambda ctx: self.ejecutar(tipo, recursos, nombre, params, prep, clase, archivo, max_muestras, ctx))

    def lanzar_comparacion(self, tipo, recursos, **kw):
        return self.tareas.lanzar("Comparativa múltiple", lambda ctx: self.ejecutar_comparacion(tipo, recursos, ctx=ctx, **kw))

    def lanzar_precarga(self, tipo, recursos):
        return self.tareas.lanzar(f"Extraer características ({tipo})", lambda ctx: self.precalcular(tipo, recursos, ctx))

    # ------------------------------------------------------------------ trazabilidad 
    def _registrar_run(self, tipo, clase, nombre, params, d, dur, mem, estado, clave, mensaje, metricas=None) -> str:
        rid = "run-" + uuid.uuid4().hex[:10]
        try:
            with self.bd() as con:
                con.execute("INSERT INTO runs(run_id,created_utc,tipo,clase,nombre,params,huella,n_muestras,n_variables,semilla,duracion_s,"
                            "memoria_mb,estado,clave_resultado,mensaje,versiones,metricas) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (rid, db.now(), tipo, clase, nombre, json.dumps(_limpio(params), default=str, ensure_ascii=False), d.huella, d.n,
                             d.X.shape[1], self.cfg.semilla, dur, mem, estado, clave, mensaje, json.dumps(_versiones()),
                             json.dumps({str(k): str(v) for k, v in (metricas or {}).items()}, ensure_ascii=False)))
                con.commit()
        except Exception:  # noqa: BLE001
            self.log.warning("No se pudo registrar la ejecución", exc_info=True)
        return rid

    def historial(self, limite: int = 50):
        import pandas as pd

        with self.bd() as con:
            filas = con.execute("SELECT run_id, created_utc, tipo, clase, nombre, params, n_muestras, n_variables, semilla, duracion_s, memoria_mb, estado, "
                                "clave_resultado, mensaje, metricas, versiones, huella "
                                "FROM runs ORDER BY created_utc DESC LIMIT ?", (limite,)).fetchall()
        return pd.DataFrame([dict(f) for f in filas])

    def recuperar(self, clave: str):
        """Reabre un resultado guardado (también tras reiniciar la aplicación)."""
        return self.cache_resultados.leer_pkl(clave)

    def exportar(self, s: Salida) -> bytes:
        """ZIP con configuración, metadatos, tablas, proyección e informe de una ejecución."""
        import pandas as pd
        from visualization.charts import a_html

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("configuracion.json", json.dumps(dict(
                run_id=s.run_id, nombre=s.nombre, clase=s.clase, tipo=s.tipo, parametros=s.params, preparacion=s.prep,
                semilla=self.cfg.semilla, huella_datos=s.datos.huella, muestras=s.datos.n, variables=s.datos.X.shape[1],
                versiones=_versiones(), clave_resultado=s.clave, exportado_utc=db.now()), indent=2, ensure_ascii=False, default=str))
            z.writestr("informe.md", s.res.informe)
            z.writestr("kpis.json", json.dumps({k: str(v) for k, v in s.res.kpis.items()}, indent=2, ensure_ascii=False))
            if s.res.tabla is not None:
                z.writestr("tabla.csv", s.res.tabla.to_csv(index=False))
            if s.res.Z is not None:
                df = pd.DataFrame(s.res.Z[:, :3], columns=s.res.ejes[:3])
                df.insert(0, "muestra", s.datos.hover)
                df.insert(0, "archivo", s.datos.origen)
                if s.res.etiquetas is not None:
                    df["grupo"] = s.res.etiquetas
                z.writestr("proyeccion.csv", df.to_csv(index=False))
            if s.res.figuras:
                z.writestr("figuras.html", a_html(s.res.figuras))
        return buf.getvalue()

    # ------------------------------------------------------------------ mantenimiento y monitoreo
    def estado_sistema(self) -> dict:
        return dict(memoria_mb=memoria_mb(), tareas_activas=len(self.tareas.activas()), disco=self.almacen.uso_disco(),
                    cache_entradas=Cache(self.cfg.subdir("cache")).n_entradas())

    def limpiar(self, que: str) -> int:
        if que == "temporales":
            return self.almacen.limpiar_temporales(forzar=True)
        if que == "derivados":
            return self.almacen.limpiar_derivados()
        if que == "cache":
            return Cache(self.cfg.subdir("cache")).limpiar()
        if que == "resultados":
            return self.cache_resultados.limpiar()
        raise ErrorValidacion(f"No se puede limpiar «{que}»")


def _limpio(params: dict) -> dict:
    return {k: (str(v) if isinstance(v, Archivo) else v) for k, v in params.items()}
