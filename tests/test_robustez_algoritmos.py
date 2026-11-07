"""Secciones 3 y 4 de las notas: los 12 algoritmos ante casos límite, diagnósticos, recomendaciones,
estabilidad, sensibilidad y ranking multicriterio con detección de comparaciones no válidas."""
import numpy as np
import pytest
import scipy.sparse as sp

from core import registry
from core.errors import ErrorValidacion
from core.results import Datos
from ml import comparisons as C
from ml import preprocessing as P
from ml.algorithms import diagnostico, jerarquico


def datos(X, tipo="audio", clases=None):
    n = X.shape[0]
    meta = [dict(clase=clases[i]) if clases is not None else {} for i in range(n)]
    return Datos(X, [f"v{i}" for i in range(X.shape[1])], np.array(["a"] * n), np.array([str(i) for i in range(n)]), True,
                 tipo=tipo, ref=list(range(n)), meta=meta)


def correr(nombre, d, X, extra=None):
    e = registry.obtener(nombre)
    p = registry.validar_parametros(e, d.tipo, dict(extra or {}))
    err, _ = P.validar(e, d, X, p)
    if err:
        raise ErrorValidacion(" ".join(err))
    return e.resolver()(d, X if e.acepta_dispersa else P.densificar(X), p)


rng = np.random.default_rng(0)
CASOS = {
    "pocas_muestras": rng.normal(size=(13, 6)),
    "duplicados": np.repeat(rng.normal(size=(2, 6)), 20, axis=0),
    "casi_constante": np.ones((40, 6)) + 1e-9 * rng.normal(size=(40, 6)),
    "una_variable_util": np.c_[rng.normal(size=40), np.zeros(40), np.zeros(40)],
    "escala_enorme": rng.normal(size=(40, 6)) * 1e6,
}


@pytest.mark.parametrize("caso", list(CASOS))
@pytest.mark.parametrize("tipo", ["audio", "texto"])
def test_doce_algoritmos_no_rompen_en_casos_limite(caso, tipo):
    """Cada algoritmo devuelve un resultado o un error de validación comprensible, nunca una excepción inesperada."""
    X = CASOS[caso]
    if tipo == "texto":
        X = sp.csr_matrix(np.abs(X))
    d = datos(X, tipo)
    for nombre in registry.algoritmos(tipo):
        try:
            r = correr(nombre, d, X)
        except ErrorValidacion as e:
            assert e.mensaje
            continue
        assert r.informe


def test_datos_identicos_se_rechazan_con_mensaje():
    X = np.ones((30, 5))
    with pytest.raises(ErrorValidacion, match="variabilidad"):
        correr("Agrupación · KMeans", datos(X), X)


def test_k_mayor_que_muestras_distintas():
    X = np.repeat(rng.normal(size=(2, 4)), 15, axis=0)
    with pytest.raises(ErrorValidacion, match="distintas"):
        correr("Agrupación · KMeans", datos(X), X, {"k": 5})


def test_min_samples_mayor_que_n():
    X = rng.normal(size=(9, 3))
    with pytest.raises(ErrorValidacion, match="min_samples"):
        correr("DBSCAN · HDBSCAN", datos(X), X, {"min_samples": 20})


def test_todo_ruido_y_un_solo_grupo_se_diagnostican():
    X = rng.normal(size=(60, 4))
    r = correr("DBSCAN · HDBSCAN", datos(X), X, {"eps": 1e-6, "min_samples": 5})
    assert (r.etiquetas == -1).all() and "ruido" in r.informe and "Sugerencia" in r.informe
    assert any("ruido" in x for x in r.extra["diagnostico"])
    assert "único grupo" in diagnostico(np.zeros(20, int))[0]
    assert any("concentra" in x for x in diagnostico(np.r_[np.zeros(95, int), np.ones(5, int)]))


def test_validacion_externa_con_clases():
    X = np.vstack([rng.normal(0, .3, (30, 3)), rng.normal(5, .3, (30, 3))])
    d = datos(X, clases=["Aves"] * 30 + ["Amphibia"] * 30)
    r = correr("Agrupación · KMeans", d, X, {"k": 2})
    assert float(r.kpis["ARI vs clase"]) == pytest.approx(1.0) and "Validación externa" in r.informe


def test_terminos_caracteristicos_en_texto():
    docs = np.zeros((20, 4))
    docs[:10, 0], docs[10:, 3] = 1, 1
    d = datos(sp.csr_matrix(docs + 0.01), "texto")
    d.nombres = ["rana", "x", "y", "pájaro"]
    r = correr("Agrupación · KMeans", d, d.X, {"k": 2})
    assert set(r.tabla["términos característicos"].str.split(", ").str[0]) == {"rana", "pájaro"}


def test_jerarquico_respeta_limite_de_memoria():
    X = rng.normal(size=(600, 3))
    r = jerarquico(datos(X), X, {"k": 3, "enlace": "ward", "hojas": 20}, max_mb=0.5)
    assert len(r.etiquetas) == 600 and "submuestra" in r.informe or "muestras (límite" in r.informe


def test_recomendaciones_por_algoritmo():
    from ml.recomendaciones import recomendar

    X = np.vstack([rng.normal(c, .2, (40, 3)) for c in (0, 4, 8)])
    d = datos(X)
    assert recomendar("Agrupación · KMeans", d, X)["k"][0] == 3
    r = recomendar("DBSCAN · HDBSCAN", d, X)
    assert r["eps"][0] > 0 and 4 <= r["min_samples"][0] <= 12 and r["min_samples"][1]
    assert recomendar("t-SNE · UMAP", d, X)["perplexity"][0] <= (len(X) - 1) / 3
    assert 2 <= recomendar("Reducción de dimensionalidad · PCA", d, X)["n_comp"][0] <= 3


def test_ranking_multicriterio_y_comparaciones_no_validas():
    X = np.vstack([rng.normal(c, .3, (40, 2)) for c in ((0, 0), (5, 5), (0, 5))])
    d = datos(X, "geo")
    d.nombres = ["x", "y"]
    res = C.comparar_algoritmos(d, X, ["Agrupación · KMeans", "Clusterización jerárquica", "DBSCAN · HDBSCAN"],
                                {"DBSCAN · HDBSCAN": {"eps": 1e-6}}, estabilidad=True, reps=2)
    t = res.tabla
    assert t.loc[t["configuración"] == "DBSCAN · HDBSCAN", "comparable"].iloc[0].startswith("no:")
    assert t["posición"].notna().sum() == 2 and t.iloc[0]["comparable"] == "sí"
    assert len(t.attrs["criterios"]) >= 4 and "estabilidad (ARI) ↑" in t
    assert "no válidas" in res.informe


def test_sensibilidad_y_estabilidad():
    X = np.vstack([rng.normal(c, .3, (40, 2)) for c in ((0, 0), (6, 6))])
    d = datos(X, "geo")
    s = C.comparar_parametros(d, X, "Agrupación · KMeans", "k", [2, 3, 4], estabilidad=False)
    assert "ARI vs anterior" in s.tabla and s.kpis["Mejor (ranking)"] == "k=2"
    e = C.estabilidad(d, X, "Agrupación · KMeans", {"k": 2}, reps=4)
    assert float(e.kpis["ARI medio"]) > 0.9 and e.kpis["Estabilidad"] == "alta"
