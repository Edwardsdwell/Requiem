"""Rendimiento y memoria: comprueban que la caché y las matrices dispersas ayudan de verdad."""
import time

import numpy as np
import scipy.sparse as sp

from core.engine import Motor


def test_cache_de_caracteristicas_acelera(cfg, archivos):
    m = Motor(cfg)
    m.ingerir_carpeta(archivos)
    rs = m.recursos("audio")
    t0 = time.perf_counter()
    m.cargar("audio", rs)
    frio = time.perf_counter() - t0
    t0 = time.perf_counter()
    m.cargar("audio", rs)
    caliente = time.perf_counter() - t0
    print(f"audio frío {frio:.2f}s · caliente {caliente:.3f}s")
    assert caliente < frio / 3


def test_cache_de_resultados_acelera(cfg, archivos):
    m = Motor(cfg)
    m.ingerir_carpeta(archivos)
    rs = m.recursos("img")
    a = m.ejecutar("img", rs, "Modelo mixto gaussiano", {"k": 3})
    b = m.ejecutar("img", rs, "Modelo mixto gaussiano", {"k": 3})
    assert b.desde_cache and b.duracion_s == 0.0 and a.duracion_s > 0


def test_tfidf_disperso_ocupa_menos_que_denso(cfg, tmp_path):
    rng = np.random.default_rng(0)
    vocab = [f"palabra{i}x" for i in range(1500)]
    parrafos = [" ".join(rng.choice(vocab, 25)) + " texto de relleno para superar la longitud minima" for _ in range(400)]
    f = tmp_path / "corpus.txt"
    f.write_text("\n\n".join(parrafos), encoding="utf-8")
    m = Motor(cfg)
    m.ingerir([("corpus.txt", f)], "texto")
    d = m.cargar("texto", m.recursos("texto"))
    disperso = d.X.data.nbytes + d.X.indices.nbytes + d.X.indptr.nbytes
    print(f"disperso {disperso / 1024:.0f} KB frente a denso {d.X.toarray().nbytes / 1024:.0f} KB")
    assert sp.issparse(d.X) and disperso < d.X.toarray().nbytes / 3


def test_tamano_de_cache_y_derivados_acotados(cfg, archivos):
    m = Motor(cfg)
    m.ingerir_carpeta(archivos)
    m.cargar("img", m.recursos("img"))
    uso = m.almacen.uso_disco()
    originales = sum(p.stat().st_size for p in archivos.glob("*"))
    assert uso["originales"] <= originales * 1.01                         # el almacén no duplica contenido
    assert uso["caché"] < 5 * 2**20


def test_kmeans_sobre_disperso_sin_densificar_memoria_estable():
    from sklearn.cluster import KMeans

    X = sp.random(20000, 800, 0.01, format="csr", random_state=0)
    t0 = time.perf_counter()
    KMeans(5, n_init=1, random_state=0).fit(X)
    assert time.perf_counter() - t0 < 60 and X.data.nbytes < 20000 * 800 * 8 / 10


def test_respuesta_grafica_con_muchos_puntos():
    from core.results import Datos, Res
    from visualization.charts import dispersion

    n = 8000
    Z = np.random.default_rng(0).normal(size=(n, 3))
    d = Datos(Z, list("abc"), np.array(["f"] * n), np.array(["m"] * n), tipo="audio")
    t0 = time.perf_counter()
    fig = dispersion(d, Res("x", Z, ["a", "b", "c"], np.zeros(n, int)), "2D", "Grupo", 4, .8)
    assert time.perf_counter() - t0 < 5 and fig.data[0].type == "scattergl"
