"""Algoritmos, registro extensible y preparación."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

from core import registry
from core.errors import ErrorValidacion
from core.results import Datos, Param, Res
from ml import preprocessing as P

RAIZ = Path(__file__).resolve().parent.parent


def blobs(n=60, p=4, k=3, sep=6.0, semilla=0):
    rng = np.random.default_rng(semilla)
    y = np.repeat(np.arange(k), n // k)
    X = rng.normal(size=(len(y), p)) + sep * np.eye(k, p)[y] * 1.0 + y[:, None]
    d = Datos(X, [f"v{i}" for i in range(p)], np.array([f"f{i}" for i in y]), np.array([f"m{i}" for i in range(len(y))]), tipo="audio")
    return d, y


def correr(nombre, d, X, **kw):
    e = registry.obtener(nombre)
    return e.resolver()(d, X, registry.validar_parametros(e, d.tipo, kw))


def test_catalogo_completo_y_perezoso():
    assert len(registry.algoritmos()) == 12 and len(registry.comparativas()) == 13
    codigo = "import core.registry as r; r.algoritmos(); import sys; print(any(m in sys.modules for m in ('sklearn','plotly','umap','librosa')))"
    assert subprocess.run([sys.executable, "-c", codigo], cwd=RAIZ, capture_output=True, text=True).stdout.strip() == "False"


@pytest.mark.parametrize("nombre", sorted(registry.algoritmos()))
def test_los_12_algoritmos_devuelven_resultados_estandar(nombre):
    d, _ = blobs()
    e = registry.obtener(nombre)
    X = P.preparar(d)[1]
    res = correr(nombre, d, X)
    assert isinstance(res, Res) and res.informe and (res.Z is not None or res.figuras)
    assert e.descripcion and e.params


def test_kmeans_recupera_los_grupos_y_dbscan_marca_ruido():
    d, y = blobs()
    X = P.preparar(d)[1]
    assert adjusted_rand_score(y, correr("Agrupación · KMeans", d, X, k=3).etiquetas) > 0.9
    assert correr("KMeans (k óptimo)", d, X).kpis["k óptimo"] in (2, 3)
    Xr = np.vstack([X, [[40, 40, 40, 40]]])
    d2 = Datos(Xr, d.nombres, np.append(d.origen, "x"), np.append(d.hover, "r"), tipo="audio")
    r = correr("DBSCAN · HDBSCAN", d2, Xr, min_samples=4)
    assert r.etiquetas[-1] == -1                                         # el atípico es ruido (-1)


@pytest.mark.parametrize("cov", ["full", "tied", "diag", "spherical"])
def test_gmm_covarianzas(cov):
    d, _ = blobs()
    assert correr("Modelo mixto gaussiano", d, P.preparar(d)[1], covarianza=cov).etiquetas is not None


@pytest.mark.parametrize("enlace", ["ward", "average", "complete", "single"])
def test_jerarquico_enlaces(enlace):
    d, _ = blobs()
    assert len(set(correr("Clusterización jerárquica", d, P.preparar(d)[1], enlace=enlace).etiquetas)) == 3


def test_reduccion_y_lda_marcado_como_supervisado():
    d, y = blobs()
    X = P.preparar(d)[1]
    assert correr("Reducción de dimensionalidad · PCA", d, X).Z.shape[1] == 3
    assert correr("Descomposición en valores singulares", d, X, rango=2).kpis["Energía retenida"]
    assert correr("Aproximación iterativa · PPCA", d, X).Z.shape[1] == 2
    r = correr("Autoencoder · LDA", d, X, metodo="LDA")
    assert "supervisado" in r.informe.lower()                            
    assert registry.obtener("Autoencoder · LDA").notas


def test_tsne_y_umap():
    d, _ = blobs(n=45)
    X = P.preparar(d)[1]
    assert correr("t-SNE · UMAP", d, X, metodo="t-SNE", iteraciones=250).Z.shape == (45, 2)
    assert correr("t-SNE · UMAP", d, X, metodo="UMAP", dims=3).Z.shape == (45, 3)


def test_texto_disperso_con_kmeans_y_lda_tematico(motor, archivos):
    motor.ingerir([("texto.txt", archivos / "texto.txt")], "texto")
    rs = motor.recursos("texto")
    s = motor.ejecutar("texto", [r for r in rs if r.nombre == "texto.txt"], "Agrupación · KMeans", {"k": 3})
    assert sp.issparse(s.datos.X) and s.res.kpis["Grupos"] == 3
    t = motor.ejecutar("texto", [r for r in rs if r.nombre == "texto.txt"], "Autoencoder · LDA", {"metodo": "LDA", "temas": 3})
    assert t.res.tabla is not None and "Dirichlet" in t.res.informe


# ------------------------------------------------------------------ registro extensible 
def test_registrar_y_retirar_sin_tocar_el_nucleo():
    def mi(d, X, p):
        return Res("hola", Z=X[:, :2], ejes=["a", "b"])

    registry.registrar("Prueba", mi, [Param("k", "k", 2, 1, 5)], ("audio",), "x", min_muestras=3)
    assert "Prueba" in registry.algoritmos("audio") and "Prueba" not in registry.algoritmos("geo")
    d, _ = blobs()
    assert registry.obtener("Prueba").resolver()(d, d.X, {}).informe == "hola"
    registry.retirar("Prueba")
    assert "Prueba" not in registry.algoritmos()


def test_parametros_se_acotan_y_respetan_dependencias():
    e = registry.obtener("DBSCAN · HDBSCAN")
    v = registry.validar_parametros(e, "audio", {"min_samples": 999, "metodo": "HDBSCAN"})
    assert v["min_samples"] == 50 and "eps" not in v and v["tam_min"] == 10


# ------------------------------------------------------------------ preparación y calidad
def test_calidad_detecta_nan_constantes_duplicados():
    X = np.random.default_rng(0).normal(size=(20, 4))
    X[:, 2] = 5.0
    X[3, 1] = np.nan
    X[10] = X[11]
    d = Datos(X, list("abcd"), np.array(["f"] * 20), np.array(["m"] * 20), tipo="audio")
    q = P.calidad(d)
    assert q["nan"] == 1 and list(q["constantes"]) == [2] and q["duplicados"] == 1 and len(q["advertencias"]) == 3


@pytest.mark.parametrize("modo,n", [("excluir", 19), ("imputar", 20), ("señalar", 20)])
def test_tratamiento_de_ausentes(modo, n):
    X = np.random.default_rng(1).normal(size=(20, 4))
    X[0, 0] = np.nan
    d = Datos(X, list("abcd"), np.array(["f"] * 20), np.array(["m"] * 20), tipo="audio", meta=[{} for _ in range(20)])
    d2, X2, _ = P.preparar(d, P.Prep(ausentes=modo))
    assert d2.n == n and np.isfinite(X2).all()
    assert (modo != "señalar") or d2.meta[0].get("incompleta")


def test_escalados_y_original_intacto():
    d, _ = blobs()
    copia = d.X.copy()
    for esc in ("standard", "minmax", "l2", "ninguno", "auto"):
        d2, X, _ = P.preparar(d, P.Prep(escalado=esc))
        assert X.shape == d.X.shape
    assert np.array_equal(d.X, copia)                                   
    _, Xs, _ = P.preparar(d, P.Prep(escalado="standard"))
    assert np.allclose(Xs.mean(0), 0, atol=1e-6) and np.allclose(Xs.std(0), 1, atol=1e-6)
    _, Xm, _ = P.preparar(d, P.Prep(escalado="minmax"))
    assert Xm.min() >= 0 and Xm.max() <= 1


def test_variables_constantes_se_excluyen():
    d, _ = blobs()
    d.X[:, 1] = 3.0
    d2, X, info = P.preparar(d)
    assert X.shape[1] == 3 and info["constantes_quitadas"] == 1 and "v1" not in d2.nombres


def test_validacion_previa_bloquea_con_mensajes():
    d, _ = blobs(n=6)
    e = registry.obtener("t-SNE · UMAP")
    err, _ = P.validar(e, d, d.X, {})
    assert any("muestras" in m for m in err)
    err2, _ = P.validar(registry.obtener("Agrupación · KMeans"), d, d.X, {"k": 50})
    assert err2
    err3, _ = P.validar(registry.obtener("Reducción de dimensionalidad · PCA"), Datos(d.X, d.nombres, d.origen, d.hover, tipo="geo"), d.X, {})
    assert any("no es aplicable" in m for m in err3)                      


def test_densificar_respeta_limite_de_memoria():
    X = sp.random(5000, 5000, 0.001, format="csr")
    with pytest.raises(ErrorValidacion):
        P.densificar(X, max_mb=10)
    assert P.densificar(X, max_mb=512).shape == (5000, 5000)
