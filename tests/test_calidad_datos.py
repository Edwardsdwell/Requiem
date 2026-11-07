"""Secciones 1 y 2 de las notas: importación robusta, rutas de Windows, archivos dañados, integridad y calidad de datos."""
import json
import sqlite3
import wave

import numpy as np
import pytest

from data import db
from data.loaders import geo as geomod
from data.validacion import (limpiar_ruta_usuario, nombre_de_ruta, numero, partes_de_ruta, validar_archivo,
                             validar_coordenada)


# ------------------------------------------------------------------ coordenadas
@pytest.mark.parametrize("lat,lon,ok", [
    (-16.5, -68.1, True), ("-16,5", "-68,1", True), ("16.5 S", "68.1 O", True), ("abc", 1, False), (95, 10, False),
    (10, 200, False), (0, 0, False), (None, 3, False), (float("nan"), 1, False), ("", "", False), (True, 1, False)])
def test_validar_coordenada(lat, lon, ok):
    a, b, motivo = validar_coordenada(lat, lon)
    assert (motivo == "") is ok
    if ok:
        assert a == pytest.approx(-16.5) and b == pytest.approx(-68.1)


def test_numero_hemisferios():
    assert numero("16.5 S") == -16.5 and numero("68,1 W") == -68.1 and numero("12 N") == 12.0 and numero("x") is None


def test_lector_geo_no_se_interrumpe_y_explica_descartes(tmp_path):
    p = tmp_path / "mixto.csv"
    p.write_text("decimalLatitude;decimalLongitude;nombre\n-16,5;-68,1;a\nabc;2;b\n95;10;c\n0;0;d\n-17,2;-66,1;e\n", encoding="utf-8")
    pts, desc = geomod.leer_geo_informe(p)
    assert [x[:2] for x in pts] == [(-16.5, -68.1), (-17.2, -66.1)]                  # columnas GBIF y coma decimal
    assert len(desc) == 3 and all(m for _, m in desc)


def test_geojson_con_features_malas(tmp_path):
    p = tmp_path / "a.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-68.1, -16.5]}, "properties": {}},
        {"type": "Feature", "geometry": None, "properties": {}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": ["x", "y"]}, "properties": {}},
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[0, 1], [2, 3]]}, "properties": {}}]}))
    pts, desc = geomod.leer_geo_informe(p)
    assert len(pts) == 1 and len(desc) == 3


def test_json_mal_formado_no_lanza(tmp_path):
    p = tmp_path / "roto.json"
    p.write_text('{"lat": 1, "lon": ')
    pts, desc = geomod.leer_geo_informe(p)
    assert pts == [] and "JSON mal formado" in desc[0][1]


# ------------------------------------------------------------------ rutas
@pytest.mark.parametrize("ruta,nombre", [
    ("data\\raw\\audio\\abc.wav", "abc.wav"), ("C:\\Users\\HP VICTUS\\fauna\\data\\raw\\images\\x.jpg", "x.jpg"),
    ("data/raw/audio/abc.m4a", "abc.m4a"), ("", ""), (None, "")])
def test_rutas_windows_y_posix(ruta, nombre):
    assert nombre_de_ruta(ruta) == nombre


def test_partes_sin_unidad_ni_retroceso():
    assert partes_de_ruta("C:\\a\\..\\raw\\audio\\x.wav") == ["a", "raw", "audio", "x.wav"]


def test_ruta_pegada_con_comillas(tmp_path):
    d = tmp_path / "con espacio"
    d.mkdir()
    assert limpiar_ruta_usuario(f'  "{d}"  ') == d and limpiar_ruta_usuario(f"'{d}'") == d


def test_almacen_resuelve_ruta_windows_absoluta(motor, archivos):
    motor.ingerir([("img0.png", archivos / "img0.png")], "img")
    r = motor.recursos("img")[0]
    win = "C:\\Users\\otro\\proyecto\\data_store\\raw\\images\\" + r.ruta.name
    assert motor.almacen.resolver(win, "img") == r.ruta
    assert motor.almacen.resolver("..\\..\\etc\\passwd", "img") is None


# ------------------------------------------------------------------ archivos
def test_validar_archivos(tmp_path, archivos):
    vacio = tmp_path / "v.wav"
    vacio.write_bytes(b"")
    falso = tmp_path / "f.jpg"
    falso.write_bytes(b"%PDF-1.4 no soy una imagen")
    trunc = tmp_path / "t.png"
    trunc.write_bytes((archivos / "img0.png").read_bytes()[:300])
    assert validar_archivo(vacio, "audio").motivo.startswith("archivo vacío")
    assert "PDF" in validar_archivo(falso, "img").motivo
    assert not validar_archivo(trunc, "img").ok
    ok = validar_archivo(archivos / "tono0.wav", "audio")
    assert ok.ok and ok.info["duracion_s"] == pytest.approx(4.0, abs=0.01)
    assert validar_archivo(archivos / "img0.png", "img").info["ancho"] == 128
    assert not validar_archivo(tmp_path / "noexiste.wav", "audio").ok


def test_ingesta_rechaza_vacios_y_corruptos_con_motivo(motor, archivos, tmp_path):
    (tmp_path / "vacio.png").write_bytes(b"")
    (tmp_path / "malo.txt").write_bytes(b"\xff\xfe" * 500)
    inf = motor.ingerir([("vacio.png", tmp_path / "vacio.png"), ("img0.png", archivos / "img0.png"), ("malo.txt", tmp_path / "malo.txt")])
    rech = dict(inf["rechazados"])
    assert set(rech) == {"vacio.png", "malo.txt"} and all(rech.values()) and inf["agregados"] == [("img0.png", "img")]


# ------------------------------------------------------------------ importación del proyecto alternativo (estructura fauna_sonora)
def _legado(tmp_path, archivos):
    """Mini fauna_sonora: db/fauna.db + data/raw/{audio,images} con casos problemáticos."""
    from PIL import Image

    raiz = tmp_path / "Mis Descargas" / "fauna_sonora"
    (raiz / "db").mkdir(parents=True)
    (raiz / "data" / "raw" / "audio").mkdir(parents=True)
    (raiz / "data" / "raw" / "images").mkdir(parents=True)
    con = sqlite3.connect(raiz / "db" / "fauna.db")
    con.executescript("""
    CREATE TABLE events(event_id TEXT PRIMARY KEY, source TEXT, source_id TEXT, observed_raw TEXT, species TEXT, class TEXT, license TEXT, extra_col TEXT);
    CREATE TABLE geo(event_id TEXT PRIMARY KEY, lat REAL, lon REAL, crs TEXT, uncertainty_m REAL, country TEXT, locality TEXT, geo_status TEXT);
    CREATE TABLE texts(event_id TEXT, field TEXT, original TEXT, normalized TEXT, lang TEXT, PRIMARY KEY(event_id, field));
    CREATE TABLE audio(audio_id TEXT PRIMARY KEY, event_id TEXT, url TEXT, license TEXT, creator TEXT, local_path TEXT, sha256 TEXT, download_status TEXT);
    CREATE TABLE images(image_id TEXT PRIMARY KEY, event_id TEXT, url TEXT, license TEXT, creator TEXT, local_path TEXT, sha256 TEXT, download_status TEXT);
    """)
    from data.storage import sha256_archivo

    filas_a, filas_i = [], []
    for i in range(4):
        e = f"gbif:{i}"
        con.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?)", (e, "gbif", str(i), "2025-01-01", f"Especie {i}", "Aves" if i % 2 else "Amphibia", "cc-by", "x"))
        lat = [-16.5, "abc", 999, 0][i]
        con.execute("INSERT INTO geo VALUES(?,?,?,?,?,?,?,?)", (e, lat, -68.1 if i != 3 else 0, "EPSG:4326", 10, "BO", "La Paz", "present"))
        con.execute("INSERT INTO texts VALUES(?,?,?,?,?)", (e, "locality", f"Localidad {i}", None, None))
        src = archivos / f"tono{i % 3}.wav"
        dst = raiz / "data" / "raw" / "audio" / f"a{i}.wav"
        dst.write_bytes(src.read_bytes())
        filas_a.append((f"{e}:a0", e, "http://x/a.wav", "cc-by", "yo", f"data\\raw\\audio\\a{i}.wav", sha256_archivo(dst), "ok"))
        im = raiz / "data" / "raw" / "images" / f"i{i}.png"
        Image.fromarray(np.full((40, 40, 3), 60 * i, np.uint8)).save(im)
        filas_i.append((f"{e}:i0", e, "http://x/i.png", "cc-by", "yo", f"C:\\Users\\otro\\fauna_sonora\\data\\raw\\images\\i{i}.png", sha256_archivo(im), "ok"))
    (raiz / "data" / "raw" / "audio" / "a1.wav").write_bytes(b"")                                # vacío
    (raiz / "data" / "raw" / "images" / "i2.png").unlink()                                       # faltante
    (raiz / "data" / "raw" / "images" / "i3.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 50)   # corrupto
    con.executemany("INSERT INTO audio VALUES(?,?,?,?,?,?,?,?)", filas_a)
    con.executemany("INSERT INTO images VALUES(?,?,?,?,?,?,?,?)", filas_i)
    con.execute("INSERT INTO texts VALUES('gbif:NOEXISTE','locality','huérfano',NULL,NULL)")
    con.commit()
    con.close()
    return raiz


def test_importar_fauna_sonora_con_problemas(motor, archivos, tmp_path):
    raiz = _legado(tmp_path, archivos)
    rep = motor.importar_legado(f'"{raiz.parent}"')                     # carpeta padre, con espacios y comillas
    assert rep["estados"]["audio"] == {"correcto": 3, "invalido": 1}
    assert rep["estados"]["images"] == {"correcto": 2, "invalido": 1, "pendiente": 1}
    assert rep["faltantes"] == ["gbif:2:i0"] and rep["copiados"] >= 4
    assert {e for e, _ in rep["geo_invalidos"]} == {"gbif:1", "gbif:2", "gbif:3"}               # texto, fuera de rango, (0,0)
    assert rep["filas_con_error"] and rep["filas_con_error"][0][0] == "texts"                     # huérfano informado, no silenciado
    assert rep["columnas_ignoradas"] == {"events": ["extra_col"]}
    assert rep["verificacion"]["faltantes"] == 0 and rep["verificacion"]["danados"] == 0
    with motor.bd() as con:
        assert con.execute("SELECT COUNT(*) FROM geo").fetchone()[0] == 4                          # nada se pierde
        e = con.execute("SELECT error FROM audio WHERE audio_id='gbif:1:a0'").fetchone()[0]
        assert e.startswith("archivo vacío")
        assert con.execute("SELECT duration_s FROM audio WHERE audio_id='gbif:0:a0'").fetchone()[0] == pytest.approx(4.0, abs=.01)
    from core.errors import ErrorDatos

    with pytest.raises(ErrorDatos) as e:                                                          # 1 válida de 4: error que explica por qué
        motor.cargar("geo", motor.recursos("geo"))
    assert "Solo hay 1" in e.value.mensaje and "3 descartadas" in e.value.detalle
    # idempotente: reimportar no duplica
    rep2 = motor.importar_legado(raiz)
    assert rep2["ya_existian"]["events"] == 4


def test_inventario_e_informe_de_calidad(motor, archivos, tmp_path):
    from data.inventario import informe_markdown, informe_zip

    motor.importar_legado(_legado(tmp_path, archivos))
    inv = motor.inventario()
    c = inv.groupby(["modalidad", "estado"]).size().to_dict()
    assert c[("audio", "válido")] == 3 and c[("audio", "inválido")] == 1
    assert c[("img", "pendiente")] == 1 and c[("geo", "inválido")] == 3
    assert inv.id_origen.notna().sum() >= 12                                                        # se conserva el id de origen
    rep = motor.informe_calidad()
    r = rep["resumen"].set_index("modalidad")
    assert r.loc["Audio", "válidos"] == 3 and r.loc["Coordenadas", "inválidos"] == 3
    assert rep["audio"]["duracion_s"]["n"] == 3 and rep["eventos"]["eventos"] == 4 and rep["recomendaciones"]
    md = informe_markdown(rep)
    assert "## Resumen por modalidad" in md and "archivo vacío" in md
    import io
    import zipfile

    assert {"informe_calidad.md", "inventario.csv", "resumen.csv"} <= set(zipfile.ZipFile(io.BytesIO(informe_zip(rep))).namelist())
    assert len(motor.auditoria()) >= 2


def test_verificacion_detecta_archivo_borrado_y_modificado(motor, archivos, tmp_path):
    motor.importar_legado(_legado(tmp_path, archivos))
    with motor.bd() as con:
        rutas = [motor.almacen.resolver(r[0], "audio") for r in con.execute("SELECT local_path FROM audio WHERE download_status='correcto' ORDER BY audio_id")]
    rutas[0].unlink()
    rutas[1].write_bytes(rutas[1].read_bytes()[:-10] + b"0123456789")
    rep = motor.verificar_integridad()
    assert len(rep["faltantes"]) >= 1 and len(rep["danados"]) >= 1
    inv = motor.inventario()
    assert (inv[inv.modalidad == "audio"].estado == "válido").sum() <= 1


# ------------------------------------------------------------------ granularidad «archivo» y validación externa
def test_granularidad_archivo(motor, archivos, cfg):
    motor.ingerir([(f"img{i}.png", archivos / f"img{i}.png") for i in range(3)] + [(f"tono{i}.wav", archivos / f"tono{i}.wav") for i in range(3)])
    cfg.img_granularidad = cfg.audio_granularidad = "archivo"
    di, da = motor.cargar("img", motor.recursos("img")), motor.cargar("audio", motor.recursos("audio"))
    assert di.X.shape == (3, 79) and di.granularidad == "archivo" and di.ref[0][1] == -1
    assert da.X.shape == (3, 87) and da.nombres[-1] == "duracion_s"
    cfg.img_granularidad = "fragmentos"
    assert motor.cargar("img", motor.recursos("img")).X.shape == (192, 192)                         # caché distinta por granularidad


def test_migracion_de_base_antigua(cfg):
    ruta = cfg.db_path
    ruta.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(ruta)
    con.execute("CREATE TABLE runs(run_id TEXT PRIMARY KEY, created_utc TEXT, tipo TEXT, clase TEXT, nombre TEXT, params TEXT, huella TEXT, "
                "n_muestras INTEGER, n_variables INTEGER, semilla INTEGER, duracion_s REAL, memoria_mb REAL, estado TEXT, clave_resultado TEXT, mensaje TEXT)")
    con.execute("INSERT INTO runs(run_id) VALUES('viejo')")
    con.commit()
    con.close()
    c = db.init(cfg)
    cols = {r[1] for r in c.execute("PRAGMA table_info(runs)")}
    assert {"versiones", "metricas"} <= cols and c.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


