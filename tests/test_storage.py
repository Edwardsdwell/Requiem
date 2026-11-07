"""Almacenamiento, base de datos y descargas."""
import http.server
import threading

import pytest

from core.errors import ErrorDescarga, ErrorValidacion
from core.tasks import Contexto
from data import db
from data.download import Descargador, validar_url
from data.storage import Almacen, detectar_tipo, sha256_archivo


def test_contenido_unico_por_hash(cfg, archivos):
    al = Almacen(cfg)
    r1, sha1, dup1 = al.ingerir(archivos / "tono0.wav", "audio")
    r2, sha2, dup2 = al.ingerir(archivos / "tono0.wav", "audio", "otro_nombre.wav")
    assert (dup1, dup2) == (False, True) and r1 == r2 and sha1 == sha2 == sha256_archivo(r1)
    assert r1.name == f"{sha1}.wav" and len(list(al.dir_tipo("audio").glob("*"))) == 1
    assert not list(al.tmp.glob("*"))                                   # sin temporales abandonados


def test_nombres_peligrosos_y_formatos(cfg):
    al = Almacen(cfg)
    for n in ("virus.exe", "../../etc/passwd", ".oculto.wav", "pagina.html"):
        with pytest.raises(ErrorValidacion):
            al.validar_nombre(n, "audio")
    with pytest.raises(ErrorValidacion):
        al.validar_nombre("foto.png", "audio")                           # formato incompatible con el procesador


def test_resolver_no_sale_del_almacen_y_acepta_rutas_heredadas(cfg, archivos):
    al = Almacen(cfg)
    ruta, sha, _ = al.ingerir(archivos / "tono1.wav", "audio")
    assert al.resolver(f"data\\raw\\audio\\{sha}.wav") == ruta            # ruta de Windows del proyecto alternativo
    assert al.resolver("raw/audio/../../../../etc/passwd") is None
    assert al.resolver("/etc/passwd", "audio") is None


def test_limpieza_de_temporales(cfg):
    al = Almacen(cfg)
    t = al.nuevo_temporal()
    t.write_bytes(b"x")
    assert al.limpiar_temporales(forzar=True) == 1 and not t.exists()


def test_detectar_tipo(archivos):
    assert [detectar_tipo(archivos / n) for n in ("tono0.wav", "img0.png", "texto.txt", "puntos.csv")] == ["audio", "img", "texto", "geo"]


def test_bd_integridad_e_indices(cfg):
    con = db.init(cfg)
    assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(Exception):                                          # audio sin evento: viola la clave foránea
        con.execute("INSERT INTO audio(audio_id,event_id) VALUES('a','no-existe')")
    idx = {r[1] for r in con.execute("PRAGMA index_list(audio)")}
    assert {"ix_audio_status", "ix_audio_sha", "ix_audio_event"} <= idx


def test_snapshot_con_manifiesto(cfg):
    con = db.init(cfg)
    con.execute("INSERT INTO events(event_id) VALUES('e1')")
    con.execute("INSERT INTO audio(audio_id,event_id,sha256) VALUES('a1','e1','abc')")
    sid = db.crear_snapshot(con, nota="prueba")
    assert db.verificar_snapshot(con, sid)
    con.execute("UPDATE audio SET sha256='zzz'")
    assert not db.verificar_snapshot(con, sid)


def test_licencias():
    assert db.lic_ok("http://creativecommons.org/licenses/by-nc/4.0/") and db.lic_ok("CC0_1_0")
    assert not db.lic_ok("http://creativecommons.org/licenses/by-nd/4.0/") and not db.lic_ok(None)


# ------------------------------------------------------------------ descargas con un servidor local
@pytest.fixture
def servidor(archivos):
    wav = (archivos / "tono0.wav").read_bytes()
    estado = {"flaky": 0}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/ok.wav":
                cuerpo, ct, code = wav, "audio/wav", 200
            elif self.path == "/grande.wav":
                cuerpo, ct, code = b"0" * (3 * 2**20), "audio/wav", 200
            elif self.path == "/flaky.wav":
                estado["flaky"] += 1
                cuerpo, ct, code = (wav, "audio/wav", 200) if estado["flaky"] > 2 else (b"", "text/plain", 503)
            elif self.path == "/html":
                cuerpo, ct, code = b"<html></html>", "text/html", 200
            else:
                cuerpo, ct, code = b"", "text/plain", 404
            self.send_response(code)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", estado
    srv.shutdown()


def test_descarga_streaming_hash_y_duplicado(cfg, servidor):
    base, _ = servidor
    dl = Descargador(cfg)
    r1, r2 = dl.descargar(base + "/ok.wav", "audio"), dl.descargar(base + "/ok.wav", "audio")
    assert r1.estado == "correcto" and r1.sha256 == sha256_archivo(r1.ruta) and r2.duplicado and r1.ruta == r2.ruta
    assert not list(dl.almacen.tmp.glob("*"))


def test_limites_mime_y_http(cfg, servidor):
    base, _ = servidor
    cfg.max_audio_mb = 1
    dl = Descargador(cfg)
    for ruta in ("/grande.wav", "/html", "/nada"):
        with pytest.raises(ErrorDescarga):
            dl.descargar(base + ruta, "audio")
    assert not list(dl.almacen.tmp.glob("*")) and not list(dl.almacen.dir_tipo("audio").glob("*"))   # nada a medias


def test_reintentos_con_espera_incremental(cfg, servidor):
    base, estado = servidor
    r = Descargador(cfg).con_reintentos(base + "/flaky.wav", "audio")
    assert r.estado == "correcto" and estado["flaky"] == 3


def test_cancelacion_no_deja_residuos(cfg, servidor):
    base, _ = servidor
    ctx = Contexto()
    ctx.cancelar()
    from core.errors import Cancelado

    with pytest.raises(Cancelado):
        Descargador(cfg).descargar(base + "/ok.wav", "audio", ctx)
    assert not list(Almacen(cfg).tmp.glob("*"))


def test_lote_descarga_sin_filtrar_licencias(cfg, servidor):
    base, _ = servidor
    con = db.init(cfg)
    con.execute("INSERT INTO events(event_id) VALUES('e')")
    filas = [("a1", base + "/ok.wav", "CC0_1_0"), ("a2", base + "/ok.wav", "http://creativecommons.org/licenses/by-nd/4.0/"),
             ("a3", None, "CC0_1_0"), ("a4", base + "/nada", "CC0_1_0")]
    con.executemany("INSERT INTO audio(audio_id,event_id,url,license,download_status) VALUES(?, 'e', ?, ?, 'pendiente')", filas)
    con.commit()
    res = Descargador(cfg).descargar_pendientes(con)
    est = dict(con.execute("SELECT audio_id, download_status FROM audio").fetchall())
    # La aplicación conserva y descarga medios sin decidir su elegibilidad jurídica.
    assert est == {"a1": "correcto", "a2": "correcto", "a3": "omitido", "a4": "fallido"}
    assert res["correctos"] == 2
    again = Descargador(cfg).pendientes(con, reintentar_fallidos=False)
    assert again == []                                                     # lo ya resuelto no se vuelve a pedir


@pytest.mark.parametrize("url", ["ftp://x.com/a.wav", "file:///etc/passwd", "http://user:pw@example.com/a", "http://127.0.0.1/a",
                                 "http://localhost/a", "http://169.254.169.254/latest", "http://10.0.0.5/a", "http://[::1]/a"])
def test_ssrf_bloqueado(url):
    with pytest.raises(ErrorDescarga):
        validar_url(url, permitir_privadas=False)
