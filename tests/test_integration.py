"""Flujo completo: adquisición -> carga -> algoritmo -> exportación -> reutilización."""
import io
import json
import time
import zipfile

import numpy as np
import pytest

from core.engine import Motor
from core.errors import ErrorValidacion


def preparar_datos(motor, archivos):
    inf = motor.ingerir_carpeta(archivos)
    assert not inf["rechazados"] and len(inf["agregados"]) == 8
    return {t: motor.recursos(t) for t in ("audio", "img", "texto", "geo")}


def test_cuatro_tipos_desde_una_sola_interfaz(motor, archivos):
    rs = preparar_datos(motor, archivos)
    esperado = {"audio": ("Agrupación · KMeans", {"k": 3}), "img": ("Reducción de dimensionalidad · PCA", {}),
                "texto": ("Clusterización jerárquica", {"k": 3}), "geo": ("DBSCAN · HDBSCAN", {"min_samples": 5})}
    for tipo, (nombre, params) in esperado.items():
        s = motor.ejecutar(tipo, rs[tipo], nombre, params)
        assert s.res.informe and s.datos.tipo == tipo and not s.desde_cache


def test_resultados_reutilizables_tras_reiniciar(cfg, archivos):
    m1 = Motor(cfg)
    rs = preparar_datos(m1, archivos)["audio"]
    a = m1.ejecutar("audio", rs, "Agrupación · KMeans", {"k": 3})
    m2 = Motor(cfg)                                                      # «reinicio» de la aplicación
    b = m2.ejecutar("audio", m2.recursos("audio"), "Agrupación · KMeans", {"k": 3})
    assert b.desde_cache and b.clave == a.clave and np.array_equal(a.res.etiquetas, b.res.etiquetas)
    assert m2.recuperar(a.clave) is not None
    h = m2.historial()
    assert list(h.estado) == ["cache", "correcta"] and h.semilla.eq(0).all()


def test_reproducible_y_la_semilla_cambia_la_clave(cfg, archivos):
    m = Motor(cfg)
    rs = preparar_datos(m, archivos)["img"]
    a = m.ejecutar("img", rs, "Agrupación · KMeans", {"k": 3}, usar_cache=False)
    b = m.ejecutar("img", rs, "Agrupación · KMeans", {"k": 3}, usar_cache=False)
    assert np.array_equal(a.res.etiquetas, b.res.etiquetas) and a.clave == b.clave
    cfg.semilla = 7
    assert m.ejecutar("img", rs, "Agrupación · KMeans", {"k": 3}, usar_cache=False).clave != a.clave


def test_exportacion_con_configuracion_y_metadatos(motor, archivos):
    rs = preparar_datos(motor, archivos)["geo"]
    s = motor.ejecutar("geo", rs, "Agrupación · KMeans", {"k": 3})
    z = zipfile.ZipFile(io.BytesIO(motor.exportar(s)))
    assert {"configuracion.json", "informe.md", "kpis.json", "tabla.csv", "proyeccion.csv", "figuras.html"} <= set(z.namelist())
    conf = json.loads(z.read("configuracion.json"))
    assert conf["parametros"]["k"] == 3 and conf["semilla"] == 0 and "scikit-learn" in conf["versiones"] and conf["huella_datos"]


def test_comparativas_originales_con_archivo_seleccionado(motor, archivos):
    rs = preparar_datos(motor, archivos)
    s = motor.ejecutar("img", rs["img"], "Compresión de color con K-means", {"k": 4}, clase="comparativa", archivo=rs["img"][0])
    assert s.res.kpis["Colores K"] == 4
    with pytest.raises(ErrorValidacion):
        motor.ejecutar("img", rs["img"], "Escala de grises y negativo", {}, clase="comparativa")      # falta el archivo


def test_comparar_varios_algoritmos_y_varios_valores(motor, archivos):
    rs = preparar_datos(motor, archivos)["geo"]
    a = motor.ejecutar_comparacion("geo", rs, "algoritmos", algoritmos=["Agrupación · KMeans", "Clusterización jerárquica", "Modelo mixto gaussiano"])
    assert len(a.res.tabla) == 3 and "silueta ↑" in a.res.tabla
    b = motor.ejecutar_comparacion("geo", rs, "parametros", algoritmo="Agrupación · KMeans", parametro="k", valores=[2, 3, 4, 5])
    assert list(b.res.tabla["k"]) == [2, 3, 4, 5]
    assert b.res.tabla.set_index("k")["silueta ↑"].idxmax() == 3          # tres grupos reales


def test_validacion_antes_de_ejecutar(motor, archivos):
    rs = preparar_datos(motor, archivos)["geo"]
    with pytest.raises(ErrorValidacion):
        motor.ejecutar("geo", rs, "Reducción de dimensionalidad · PCA", {})                              # PCA no admite coordenadas


def test_tarea_en_segundo_plano_y_cancelacion(motor, archivos):
    rs = preparar_datos(motor, archivos)["audio"]
    t = motor.lanzar_ejecucion("audio", rs, "Métricas para agrupación", {"k_max": 25})
    time.sleep(0.05)
    motor.tareas.cancelar(t.id)
    while t.activa:
        time.sleep(0.05)
    assert t.estado in ("cancelada", "terminada")                         # cooperativa: puede terminar antes del siguiente punto de control
    t2 = motor.lanzar_ejecucion("audio", rs, "Agrupación · KMeans", {"k": 3})
    while t2.activa:
        time.sleep(0.05)
    assert t2.estado == "terminada" and t2.resultado.res.kpis["Grupos"] == 3
    assert not list(motor.almacen.tmp.glob("*")) and motor.resumen()["ejecuciones"] >= 1


def test_tarea_fallida_se_informa_sin_romper(motor):
    t = motor.tareas.ejecutar_sync("falla", lambda ctx: 1 / 0)
    assert t.estado == "fallida" and t.error["codigo"] == "E_INESPERADO"


def test_originales_nunca_se_modifican(motor, archivos):
    rs = preparar_datos(motor, archivos)
    antes = {r.ruta: r.ruta.stat().st_mtime_ns for t in rs.values() for r in t if r.ruta}
    for t, nombre in (("audio", "Agrupación · KMeans"), ("img", "Reducción de dimensionalidad · PCA")):
        motor.ejecutar(t, rs[t], nombre, {})
    assert antes == {p: p.stat().st_mtime_ns for p in antes}


def test_limpieza_regenerable(motor, archivos):
    rs = preparar_datos(motor, archivos)["img"]
    a = motor.ejecutar("img", rs, "Reducción de dimensionalidad · PCA", {})
    assert motor.limpiar("cache") > 0
    b = motor.ejecutar("img", rs, "Reducción de dimensionalidad · PCA", {})
    assert not b.desde_cache and np.allclose(a.res.Z, b.res.Z)           # se regenera idéntico
