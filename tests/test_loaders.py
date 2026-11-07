"""Procesadores de audio, imágenes, texto y geografía."""
import json

import numpy as np
import pytest
import scipy.sparse as sp

from core.errors import ErrorDatos
from data import loaders
from data.loaders import Recurso
from data.loaders import geo as geomod


def rec(motor, archivos, nombre, tipo):
    inf = motor.ingerir([(nombre, archivos / nombre)], tipo)
    assert inf["rechazados"] == []
    return next(r for r in motor.recursos(tipo) if r.nombre == nombre)


def test_audio_caracteristicas_y_cache(motor, archivos, cfg):
    rs = [rec(motor, archivos, f"tono{i}.wav", "audio") for i in range(3)]
    d = motor.cargar("audio", rs)
    assert d.X.shape[1] == 43 and d.nombres[0] == "mfcc0_media" and d.nombres[-3:] == ["centroide", "zcr", "rms"]
    assert d.n == 3 * 7 and set(d.origen) == {r.nombre for r in rs}      # 4 s, ventana 1 s, salto 0.5 s -> 7 ventanas
    d2 = motor.cargar("audio", rs)
    assert d2.aciertos_cache == 3 and np.allclose(d.X, d2.X) and d.huella == d2.huella


def test_audio_ventana_configurable_invalida_cache(motor, archivos, cfg):
    r = rec(motor, archivos, "tono0.wav", "audio")
    a = motor.cargar("audio", [r])
    cfg.audio_salto_s = 1.0
    b = motor.cargar("audio", [r])
    assert b.n < a.n and b.aciertos_cache == 0 and a.huella != b.huella


def test_imagenes_parches(motor, archivos):
    r = rec(motor, archivos, "img0.png", "img")
    d = motor.cargar("img", [r])
    assert d.X.shape == (64, 192) and d.X.min() >= 0 and d.X.max() <= 1 and d.ref[0] == (r.nombre, 0, 0)


def test_texto_es_disperso_y_no_se_estandariza(motor, archivos):
    r = rec(motor, archivos, "texto.txt", "texto")
    d = motor.cargar("texto", [r])
    assert sp.issparse(d.X) and d.escalar is False and d.X.shape[0] == 36
    d2, X, info = motor.preparar(d)
    assert sp.issparse(X) and info["escalado"] == "ninguno"              


def test_texto_sin_contenido_da_error_claro(motor, tmp_path):
    p = tmp_path / "vacio.txt"
    p.write_text("hola")
    motor.ingerir([("vacio.txt", p)], "texto")
    with pytest.raises(ErrorDatos):
        motor.cargar("texto", [r for r in motor.recursos("texto") if r.nombre == "vacio.txt"])


def test_geo_columnas_en_espanol_y_proyeccion_local(motor, archivos):
    r = rec(motor, archivos, "puntos.csv", "geo")
    d = motor.cargar("geo", [r])
    assert d.n == 75 and d.proyeccion == "local" and d.nombres == ["x (km)", "y (km)"] and not d.avisos
    assert abs(d.X[:, 0].mean()) < 1e-6 and np.allclose(d.a_latlon(d.X), d.mapa, atol=1e-6)


def test_geojson_orden_lon_lat(tmp_path):
    p = tmp_path / "a.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-68.1, -16.5]}, "properties": {"n": 1}}]}))
    assert geomod.leer_geo(p)[0][:2] == (-16.5, -68.1)                    # [lon, lat] -> (lat, lon)
    p2 = tmp_path / "b.json"
    p2.write_text(json.dumps({"coordinates": [[-68.1, -16.5], [-68.2, -16.6]]}))
    assert geomod.leer_geo(p2)[1][:2] == (-16.6, -68.2)


def test_geo_descarta_coordenadas_invalidas(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("lat,lon\n95,10\n10,200\n10,10\nabc,2\n")
    assert len(geomod.leer_geo(p)) == 1


def test_geo_extension_grande_usa_esfera_con_aviso(cfg):
    ll = np.array([[0, 0], [10, 170], [-30, -100], [50, 20], [-60, 100.0]])
    X, nombres, centro, modo, avisos = geomod.proyectar(ll, cfg)
    assert modo == "esfera" and X.shape == (5, 3) and avisos and np.allclose(np.linalg.norm(X, axis=1), 6371)


def test_audio_ilegible_se_rechaza_al_ingerir_y_se_omite_al_cargar(motor, archivos, tmp_path):
    malo = tmp_path / "malo.wav"
    malo.write_bytes(b"RIFF" + b"\0" * 100)
    inf = motor.ingerir([("malo.wav", malo), ("tono0.wav", archivos / "tono0.wav")], "audio")
    assert [n for n, _ in inf["rechazados"]] == ["malo.wav"] and inf["rechazados"][0][1]       # rechazado con motivo
    # si un archivo dañado llegara igualmente al almacén, la carga lo omite con aviso y lo deja en `descartes`
    ruta, sha, _ = motor.almacen.ingerir(malo, "audio", "malo.wav")
    with motor.bd() as con:
        motor._registrar_documento(con, "audio", "malo.wav", ruta, sha)
        con.commit()
    d = motor.cargar("audio", motor.recursos("audio"))
    assert d.avisos and set(d.origen) == {"tono0.wav"} and d.descartes[0][0] == "malo.wav"


def test_submuestreo_reproducible(motor, archivos):
    r = rec(motor, archivos, "img0.png", "img")
    a, b = motor.cargar("img", [r], max_muestras=20), motor.cargar("img", [r], max_muestras=20)
    assert a.n == 20 and a.hover.tolist() == b.hover.tolist()
