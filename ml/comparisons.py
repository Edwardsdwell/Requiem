"""Comparativas «original vs procesado» para los cuatro tipos de datos.

Cada comparativa recibe `(datos, X preparada, parámetros, recursos)` y devuelve un `Res`. En `Res.figuras`,
un elemento suelto ocupa todo el ancho y una tupla se muestra en columnas (izquierda = original,
derecha = procesado). Los metadatos viven en `ml/catalogo.py`. Los resultados intermedios (PCA, barridos
de k) se reutilizan entre figuras de una misma comparativa.
"""
from __future__ import annotations

import base64
import io
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from core.results import Res
from visualization.charts import ESCALA_CONTINUA, PALETA, categorias, estilo
from visualization.maps import centro_de, zoom_para

from . import algorithms as A
from .preprocessing import rs

GRIS, ROJO, LUM = "#6B6A67", "#7A1F1F", np.array([0.299, 0.587, 0.114])


# ------------------------------------------------------------------ piezas comunes
def _por_origen(d) -> bool:
    """¿Sirve el archivo/categoría de origen como color? (entre 2 y ~30 valores distintos)."""
    return 1 < len(np.unique(d.origen)) <= max(2, min(30, d.n // 2))


def _orig(d):
    """(etiquetas, leyenda) del color «original»: el origen de las muestras o, si no sirve, la clase conocida
    (en la base de datos, taxa.family / taxa.order); sin etiquetas útiles devuelve (None, None)."""
    y = d.origen if _por_origen(d) else d.etiquetas_externas("clase")
    if y is not None:
        u, inv = np.unique(y, return_inverse=True)
        if len(u) > 1:
            return inv, {f"G{i}": str(n) for i, n in enumerate(u)}
    return None, None


def _p2(d, X, C=None):
    if d.proyeccion == "esfera":                       # xyz sobre la esfera: se aplana con PCA para el plano 2D
        pc = PCA(2, random_state=rs()).fit(X)
        return pc.transform(X), [f"PC{i + 1} ({v:.0%})" for i, v in enumerate(pc.explained_variance_ratio_)], None if C is None else pc.transform(C)
    Z, ejes, C = A._proy(d, X, C)
    return Z[:, :2], list(ejes[:2]), None if C is None else C[:, :2]


def _disp(Z, etq, titulo, ejes, C=None, leyenda=None, hover=None, h=430):
    """Dispersión 2D: grupos de colores, ruido como X blanca y centroides como X roja."""
    fig = go.Figure()
    ht = dict(hovertext=hover, hoverinfo="text") if hover is not None else dict(hoverinfo="skip")
    if etq is None:
        fig.add_scatter(x=Z[:, 0], y=Z[:, 1], mode="markers", name="datos", marker=dict(size=5, color=GRIS, opacity=.7), **ht)
    else:
        nom, orden, cmap = categorias(etq)
        nom = np.asarray(nom)
        for g in orden:
            m = nom == g
            kh = dict(hovertext=hover[m], hoverinfo="text") if hover is not None else dict(hoverinfo="skip")
            if g == "ruido":
                fig.add_scatter(x=Z[m, 0], y=Z[m, 1], mode="markers", name="ruido (-1)",
                                marker=dict(size=7, symbol="x", color="#1B1B1A", line=dict(width=1.5, color="#1B1B1A")), **kh)
            else:
                fig.add_scatter(x=Z[m, 0], y=Z[m, 1], mode="markers", name=(leyenda or {}).get(g, g),
                                marker=dict(size=5, color=cmap[g], opacity=.75), **kh)
    if C is not None:
        fig.add_scatter(x=C[:, 0], y=C[:, 1], mode="markers", name="centroides",
                        marker=dict(size=13, color=ROJO, symbol="x", line=dict(width=1, color="white")))
    estilo(fig, titulo, x=ejes[0], y=ejes[1], h=h)
    fig.update_layout(legend_title_text="")
    return fig


def _mapa(d, etq, titulo, leyenda=None, C=None, h=430):
    """Los mismos puntos sobre un mapa real (solo coordenadas)."""
    lat, lon = d.mapa[:, 0], d.mapa[:, 1]
    zoom = zoom_para(lat, lon)
    df = pd.DataFrame(dict(lat=lat, lon=lon, hover=d.hover))
    kw = dict(color_discrete_sequence=[GRIS])
    if etq is not None:
        df["g"], orden, cmap = categorias(etq)
        df["g"] = df["g"].map(lambda g: (leyenda or {}).get(g, g))
        cmap = {(leyenda or {}).get(g, g): c for g, c in cmap.items()}
        kw = dict(color="g", color_discrete_map=cmap, category_orders={"g": [(leyenda or {}).get(g, g) for g in orden]})
    fig = px.scatter_map(df, lat="lat", lon="lon", hover_name="hover", zoom=zoom, opacity=.8,
                         center=centro_de(lat, lon), map_style="open-street-map", **kw)
    fig.update_traces(marker_size=7)
    if C is not None:
        ll = d.a_latlon(C if d.proyeccion == "esfera" else C[:, :2])
        fig.add_scattermap(lat=ll[:, 0], lon=ll[:, 1], mode="markers", name="centroides", marker=dict(size=14, color=ROJO))
    fig.update_layout(title=dict(text=titulo, x=0.01, font_size=15), height=h, margin=dict(l=0, r=0, t=45, b=0), legend_title_text="")
    return fig


def _par_res(d, res, titulo, h=430):
    """(original, procesado) a partir del `Res` de un algoritmo de agrupación; en mapa si hay coordenadas."""
    etq, ley = _orig(d)
    if d.mapa is not None:
        return (_mapa(d, etq, "Original · sin agrupar", ley, h=h), _mapa(d, res.etiquetas, titulo, None, res.centros, h=h))
    Z, e = res.Z[:, :2], res.ejes[:2]
    C = None if res.centros is None else res.centros[:, :2]
    return (_disp(Z, etq, "Original · sin agrupar", e, leyenda=ley, hover=d.hover, h=h),
            _disp(Z, res.etiquetas, titulo, e, C, hover=d.hover, h=h))


def _calor(M, titulo, esc="RdBu_r", zmin=None, zmax=None, y="muestras", invertir=True, h=380):
    fig = go.Figure(go.Heatmap(z=M, colorscale=esc, zmin=zmin, zmax=zmax))
    fig.update_yaxes(autorange="reversed" if invertir else True, title_text=y)
    return estilo(fig, titulo, x="variables" if invertir else "tiempo", h=h)


def _rejilla(mats, titulos, cols=3, esc="Magma", zmin=None, zmax=None, h=520):
    filas = -(-len(mats) // cols)
    fig = make_subplots(rows=filas, cols=cols, subplot_titles=titulos, horizontal_spacing=.03, vertical_spacing=.13)
    for i, M in enumerate(mats):
        fig.add_trace(go.Heatmap(z=M, colorscale=esc, zmin=zmin, zmax=zmax, showscale=False), row=i // cols + 1, col=i % cols + 1)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return estilo(fig, "Comparativa progresiva de rangos", h=h)


def _energia(s, r):
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    r90, r95 = int(np.searchsorted(e, .90)) + 1, int(np.searchsorted(e, .95)) + 1
    fig = go.Figure(go.Scatter(x=np.arange(1, len(e) + 1), y=e, mode="lines", name="energía acumulada"))
    for v, c in ((.90, "#6B6A67"), (.95, "#1B1B1A")):
        fig.add_hline(y=v, line_dash="dot", line_color=c, annotation_text=f"{v:.0%}")
    fig.add_vline(x=r, line_dash="dash", line_color="crimson", annotation_text=f"k={r}")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
    return estilo(fig, "Energía acumulada de los valores singulares", x="número de valores singulares"), e, r90, r95


def _tam(r, n, m):
    """Tamaño de la aproximación de rango r respecto a la matriz n×m."""
    v = r * (n + m + 1) / (n * m)
    return f"{v:.0%}" if v < 1 else "sin ahorro"


def _sub(M, filas=60, cols=40, var=None):
    """Submatriz representativa para un mapa de calor: filas espaciadas y columnas de mayor varianza."""
    i = np.linspace(0, len(M) - 1, min(filas, len(M))).astype(int)
    j = np.sort(np.argsort((M if var is None else var).var(0))[::-1][:cols])
    return i, j


def _etiquetas_color(d, X, k):
    """Colores para comparar: el archivo de origen o, con un solo archivo, grupos de KMeans."""
    etq, ley = _orig(d)
    return (etq, ley) if etq is not None else (KMeans(k, n_init=5, random_state=rs()).fit_predict(X), None)


def _fuente(r):
    """Ruta, `Archivo` (con `.ruta`), bytes o `media_id` de la base de datos → algo que PIL y librosa abren."""
    r = getattr(r, "ruta", r)
    if isinstance(r, (int, np.integer)):
        from . import db
        r = db.contenido(int(r))
    return io.BytesIO(r) if isinstance(r, (bytes, bytearray, memoryview)) else r


def _nombre(r) -> str:
    r = getattr(r, "ruta", r)
    if isinstance(r, (str, pathlib.Path)):
        return pathlib.Path(r).name
    return f"medio {r}" if isinstance(r, (int, np.integer)) else "medio de la base de datos"


def _cargar_img(r, lado=400):
    from PIL import Image

    im = Image.open(_fuente(r)).convert("RGB")
    im.thumbnail((lado, lado))
    return np.asarray(im, dtype=np.float32) / 255


def _imagen(a, titulo):
    fig = px.imshow((np.clip(a, 0, 1) * 255).astype(np.uint8))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title=dict(text=titulo, x=0.01, font_size=15), margin=dict(l=0, r=0, t=45, b=0))
    return fig


def _rejilla_img(imgs, titulos, cols=3):
    u8 = [(np.clip(a, 0, 1) * 255).astype(np.uint8) for a in imgs]
    try:
        fig = px.imshow(np.stack(u8), facet_col=0, facet_col_wrap=cols)
        for an, t in zip(fig.layout.annotations, titulos):
            an.text = t
    except Exception:  # respaldo: subgráficos con go.Image
        fig = make_subplots(rows=-(-len(u8) // cols), cols=cols, subplot_titles=titulos)
        for i, a in enumerate(u8):
            fig.add_trace(go.Image(z=a), row=i // cols + 1, col=i % cols + 1)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


# ------------------------------------------------------------------ 1 · KMeans paso a paso
def kmeans_pasos(d, X, p, rutas):
    Z, ejes, _ = _p2(d, X)
    k = p["k"]
    C = (kmeans_plusplus(Z, k, random_state=rs())[0] if p["init"] == "k-means++"
         else Z[np.random.default_rng(rs() + 3).choice(len(Z), k, replace=False)])
    pasos = []
    for _ in range(p["iteraciones"]):
        D = ((Z[:, None, :] - C[None]) ** 2).sum(2)
        lab = D.argmin(1)
        pasos.append((lab, C.copy(), float(D[np.arange(len(Z)), lab].sum())))
        Cn = np.array([Z[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(Cn, C):
            break
        C = Cn
    n = len(pasos)

    mx, my = np.ptp(Z[:, 0]) * .06, np.ptp(Z[:, 1]) * .06
    gx, gy = np.linspace(Z[:, 0].min() - mx, Z[:, 0].max() + mx, 90), np.linspace(Z[:, 1].min() - my, Z[:, 1].max() + my, 90)
    GX, GY = np.meshgrid(gx, gy)
    P = np.c_[GX.ravel(), GY.ravel()]
    cols = [PALETA[j % len(PALETA)] for j in range(k)]
    rgba = lambda h, a: f"rgba({int(h[1:3], 16)},{int(h[3:5], 16)},{int(h[5:7], 16)},{a})"
    esc = [t for j, c in enumerate(cols) for t in ([j / k, rgba(c, .28)], [(j + 1) / k, rgba(c, .28)])]
    vor = lambda C_: ((P[:, None, :] - C_[None]) ** 2).sum(2).argmin(1).reshape(GX.shape) + .5   # celdas de Voronoi

    lab0, C0, _ = pasos[0]
    fig = go.Figure([
        go.Heatmap(z=vor(C0), x=gx, y=gy, colorscale=esc, zmin=0, zmax=k, showscale=False, hoverinfo="skip"),
        go.Scatter(x=Z[:, 0], y=Z[:, 1], mode="markers", name="datos", showlegend=False, hoverinfo="skip",
                   marker=dict(size=5, color=[cols[i] for i in lab0])),
        go.Scatter(x=C0[:, 0], y=C0[:, 1], mode="markers", name="centroides",
                   marker=dict(size=14, color=ROJO, symbol="x", line=dict(width=1, color="white")))])
    fig.frames = [go.Frame(name=str(i), traces=[0, 1, 2], data=[
        go.Heatmap(z=vor(Ci)), go.Scatter(marker=dict(color=[cols[j] for j in li])), go.Scatter(x=Ci[:, 0], y=Ci[:, 1])])
        for i, (li, Ci, _) in enumerate(pasos)]
    anim = dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))
    fig.update_layout(
        sliders=[dict(active=0, pad=dict(t=45), currentvalue=dict(prefix="Iteración: "),
                      steps=[dict(method="animate", label=str(i), args=[[str(i)], anim]) for i in range(n)])],
        updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=-0.1, xanchor="left", yanchor="top", buttons=[
            dict(label="▶ Reproducir", method="animate", args=[None, dict(frame=dict(duration=600, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
            dict(label="Pausa", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])])
    estilo(fig, "Procesado · regiones de Voronoi y centroides por iteración", x=ejes[0], y=ejes[1], h=560)
    fig.update_xaxes(range=[gx[0], gx[-1]])
    fig.update_yaxes(range=[gy[0], gy[-1]])
    fig.update_layout(margin=dict(l=50, r=20, t=50, b=110))

    etq, ley = _orig(d)
    orig = _disp(Z, etq, "Original · datos de entrada", ejes, leyenda=ley, hover=d.hover, h=560)
    ine = [x[2] for x in pasos]
    curva = px.line(x=list(range(n)), y=ine, markers=True, labels={"x": "iteración", "y": "inercia (WCSS)"})
    estilo(curva, "Suma de cuadrados intra-clúster por iteración")
    tabla = pd.DataFrame({"iteración": range(n), "inercia": np.round(ine, 3),
                          "cambio %": [None] + [round(100 * (ine[i] - ine[i - 1]) / ine[i - 1], 2) for i in range(1, n)]})
    conv = "convergió" if n < p["iteraciones"] else "alcanzó el máximo de iteraciones"
    return Res(f"**KMeans paso a paso** (k={k}, inicialización `{p['init']}`) sobre la proyección 2D. El algoritmo {conv} en {n} iteraciones. "
               "Izquierda: datos sin procesar. Derecha: celdas de Voronoi coloreadas, puntos asignados y centroides (✕ rojas); "
               "mueve el deslizador para ver cómo cambian.",
               figuras=[(orig, fig), curva], tabla=tabla,
               kpis={"Iteraciones": n, "Inercia inicial": f"{ine[0]:.0f}", "Inercia final": f"{ine[-1]:.0f}",
                     "Reducción": f"{1 - ine[-1] / ine[0]:.0%}"})


# ------------------------------------------------------------------ 2 · selección de k
def _codo(t):
    v = t.inercia.values
    x, y = np.linspace(0, 1, len(v)), (v - v[0]) / (v[-1] - v[0] + 1e-12)
    return int(t.k.iloc[int(np.argmax(y - x))])


def seleccion_k(d, X, p, rutas):
    t = A._barrido(X, p["k_max"])
    norm = lambda v: (v - v.min()) / (np.ptp(v) + 1e-12)
    t["inercia invertida"], t["silueta (norm.)"], t["CH (norm.)"] = norm(-t.inercia), norm(t.silueta), norm(t.CH)
    t["score combinado"] = t[["inercia invertida", "silueta (norm.)", "CH (norm.)"]].mean(1)
    k_sc, k_sil, k_ch, k_codo = (int(t.k[t["score combinado"].idxmax()]), int(t.k[t.silueta.idxmax()]),
                                 int(t.k[t.CH.idxmax()]), _codo(t))
    f1 = make_subplots(rows=1, cols=3, subplot_titles=("Codo (inercia)", "Silhouette ↑", "Calinski-Harabasz ↑"))
    for i, (c, kk) in enumerate([("inercia", k_codo), ("silueta", k_sil), ("CH", k_ch)], 1):
        f1.add_scatter(x=t.k, y=t[c], mode="lines+markers", showlegend=False, row=1, col=i)
        f1.add_scatter(x=[kk], y=[t[c][t.k == kk].iloc[0]], mode="markers", showlegend=False, row=1, col=i,
                       marker=dict(symbol="star", size=15, color="crimson"))
    estilo(f1, "Métricas según k (⭐ = mejor valor / codo)", x="k")
    f2 = go.Figure()
    for c, dash in (("inercia invertida", "solid"), ("silueta (norm.)", "solid"), ("CH (norm.)", "solid"), ("score combinado", "dot")):
        f2.add_scatter(x=t.k, y=t[c], mode="lines+markers", name=c, line=dict(dash=dash, width=4 if dash == "dot" else 2))
    f2.add_vline(x=k_sc, line_dash="dash", line_color="crimson", annotation_text=f"k={k_sc}")
    estilo(f2, "Métricas normalizadas y score combinado", x="k", y="valor normalizado (0–1)")
    res = A.kmeans_fijo(d, X, {"k": k_sc, "init": "k-means++"})
    return Res(f"**Selección de k**: la inercia invertida, la silueta y Calinski-Harabasz se normalizan a 0–1 y se promedian (línea punteada). "
               f"El mejor k combinado es **{k_sc}**. Abajo, los datos originales frente al agrupamiento con ese k. "
               "Aviso: la inercia invertida siempre favorece k altos, por eso conviene contrastarla con la silueta y el codo.",
               figuras=[f1, f2, _par_res(d, res, f"Procesado · KMeans con k={k_sc}")], tabla=t.round(3),
               kpis={"k combinado": k_sc, "k silueta": k_sil, "k Calinski-Harabasz": k_ch, "k codo": k_codo, **{a: b for a, b in res.kpis.items() if a in ("Silueta", "Inercia")}})


# ------------------------------------------------------------------ 3 · datos sin estandarizar vs estandarizados
def kmeans_escala(d, X, p, rutas):
    k, pares, kp, sil = p["k"], [], {}, {}
    X0 = A._densa(d.X)
    for nom, M in (("sin estandarizar", X0), ("estandarizados", StandardScaler().fit_transform(X0))):
        km = KMeans(k, n_init=10, random_state=rs()).fit(M)
        Z, e, C = _p2(d, M, km.cluster_centers_)
        sil[nom] = silhouette_score(M, km.labels_, sample_size=min(len(M), 2000), random_state=rs())
        pares.append(_disp(Z, km.labels_, f"{'Original' if nom[0] == 's' else 'Procesado'} · datos {nom} (silueta {sil[nom]:.2f})", e, C, hover=d.hover))
        kp[f"Silueta {nom}"] = f"{sil[nom]:.2f}"
    nota = (" Nota: el TF-IDF normalmente no se estandariza; aquí se hace solo para comparar." if d.tipo == "texto" else "")
    return Res(f"**KMeans con k={k}** sobre los datos tal cual (izquierda) y estandarizados a media 0 y desviación 1 (derecha). "
               "Si unas variables tienen escalas mucho mayores, dominan la distancia y los centroides quedan descentrados." + nota,
               figuras=[tuple(pares)], kpis=kp)


# ------------------------------------------------------------------ 4 · DBSCAN
def dbscan_comp(d, X, p, rutas):
    res = A.dbscan_hdbscan(d, X, p)
    figs = [_par_res(d, res, f"Procesado · {p['metodo']} (el ruido se marca con ✕)")]
    if d.mapa is not None:  # además, el plano en km con los puntos de ruido marcados
        etq, ley = _orig(d)
        figs.append((_disp(res.Z, etq, "Original · plano (km)", res.ejes, leyenda=ley, hover=d.hover),
                     _disp(res.Z, res.etiquetas, "Procesado · plano (km) con ruido", res.ejes, hover=d.hover)))
    figs += res.figuras
    return Res(f"**{p['metodo']}** sobre los datos originales. Los grupos son zonas densas; los puntos aislados quedan como **ruido** (✕ blanca, etiqueta -1)."
               + (" En coordenadas, el mapa permite inspeccionar sobre el territorio dónde se concentran los puntos." if d.mapa is not None else ""),
               figuras=figs, tabla=res.tabla, kpis=res.kpis)


# ------------------------------------------------------------------ 5 · dendrograma
def dendro(d, X, p, rutas):
    idx = np.sort(np.random.default_rng(rs()).choice(X.shape[0], min(X.shape[0], p["muestra"]), replace=False))
    res = A.jerarquico(d.tomar(idx), X[idx], p)
    return Res(f"**Clustering jerárquico** (`{p['enlace']}`) sobre {len(idx)} muestras. El eje vertical del dendrograma es la distancia a la que se fusionan "
               f"los grupos; la línea roja marca el corte que deja {p['k']} grupos.",
               figuras=[_par_res(d.tomar(idx), res, f"Procesado · {p['k']} grupos jerárquicos"), res.figuras[0]],
               tabla=res.tabla, kpis=res.kpis)


# ------------------------------------------------------------------ 6 · GMM con elipses
def _elipse(mu, cov, chi2=5.991):
    w, v = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, 120)
    return mu + ((v * np.sqrt(np.clip(w, 0, None) * chi2)) @ np.array([np.cos(t), np.sin(t)])).T


def gmm_elipses(d, X, p, rutas):
    Z, ejes, _ = _p2(d, X)
    k = p["k"]
    g = GaussianMixture(k, covariance_type="full", reg_covar=1e-4, n_init=3, random_state=rs()).fit(Z)
    lg = g.predict(Z)
    km = KMeans(k, n_init=10, random_state=rs()).fit(Z)
    etq, ley = _orig(d)
    f_gmm = _disp(Z, lg, "Procesado · GMM con elipses de covarianza (95 %)", ejes, g.means_, hover=d.hover, h=420)
    for j in range(k):
        e = _elipse(g.means_[j], g.covariances_[j])
        f_gmm.add_scatter(x=e[:, 0], y=e[:, 1], mode="lines", showlegend=False, hoverinfo="skip", line=dict(color="#1B1B1A", width=2))
    ks = list(range(1, min(p["k_max"], len(Z) - 1) + 1))
    aj = [GaussianMixture(i, covariance_type="full", reg_covar=1e-4, random_state=rs()).fit(Z) for i in ks]
    crit = go.Figure([go.Scatter(x=ks, y=[m.bic(Z) for m in aj], mode="lines+markers", name="BIC"),
                      go.Scatter(x=ks, y=[m.aic(Z) for m in aj], mode="lines+markers", name="AIC")])
    crit.add_vline(x=k, line_dash="dash", line_color="crimson")
    estilo(crit, "Selección de componentes (menor = mejor)", x="componentes")
    ari = adjusted_rand_score(km.labels_, lg)
    return Res(f"**Modelo mixto gaussiano** con {k} componentes sobre la proyección 2D. Las elipses envuelven el 95 % de cada gaussiana y permiten grupos alargados "
               "o inclinados; KMeans solo produce fronteras rectas, equidistantes entre centroides.",
               figuras=[(_disp(Z, etq, "Original · datos", ejes, leyenda=ley, hover=d.hover, h=420),
                         _disp(Z, km.labels_, "Procesado · KMeans (fronteras rectas)", ejes, km.cluster_centers_, hover=d.hover, h=420), f_gmm), crit],
               kpis={"Acuerdo KMeans–GMM (ARI)": f"{ari:.2f}", "BIC": f"{g.bic(Z):.0f}"})


# ------------------------------------------------------------------ 7 · PCA
def pca_comp(d, X, p, rutas):
    X = A._densa(X)
    nom = A._nombres(d, X.shape[1])
    q = min(max(2, p["n_comp"]), *X.shape)
    pc = PCA(q, random_state=rs()).fit(X)
    Z, v = pc.transform(X)[:, :2], pc.explained_variance_ratio_
    etq, ley = _orig(d)
    orig = _disp(X[:, :2], etq, "Original · primeras dos variables", nom[:2], leyenda=ley, hover=d.hover)
    proc = _disp(Z, etq, f"Procesado · PC1 y PC2 ({v[:2].sum():.0%} de la varianza)", [f"PC1 ({v[0]:.0%})", f"PC2 ({v[1]:.0%})"], leyenda=ley, hover=d.hover)
    L = pc.components_[:2]
    top = np.argsort(np.abs(L).max(0))[::-1][:15]
    carga = px.imshow(L[:, top], x=[nom[j] for j in top], y=["PC1", "PC2"], aspect="auto", color_continuous_scale="RdBu_r", color_continuous_midpoint=0)
    estilo(carga, "Cargas (loadings) de las variables más influyentes")
    scree = A._scree(v)
    estilo(scree, "Varianza explicada", h=360)
    return Res(f"**PCA**: izquierda, los datos en dos de sus variables originales; derecha, proyectados sobre las dos primeras componentes principales. "
               "El mapa de cargas indica qué variables pesan más en cada componente (rojo/azul = contribución positiva/negativa).",
               figuras=[(orig, proc), (scree, carga)],
               kpis={"PC1": f"{v[0]:.1%}", "PC2": f"{v[1]:.1%}", "Acumulada": f"{v.sum():.1%}"})


# ------------------------------------------------------------------ 8 · SVD (imagen · espectrograma · matriz)
def _svd_img(ruta, r):
    a = _cargar_img(ruta)
    h, w, _ = a.shape
    sv = [np.linalg.svd(a[:, :, c], full_matrices=False) for c in range(3)]
    n = len(sv[0][1])
    rec = lambda k: np.clip(np.stack([(U[:, :k] * s[:k]) @ Vt[:k] for U, s, Vt in sv], 2), 0, 1)
    r = min(r, n)
    R = rec(r)
    err = np.abs(a @ LUM - R @ LUM)
    fe, e, r90, r95 = _energia(np.linalg.svd(a @ LUM, compute_uv=False), r)
    mapa = px.imshow(err, color_continuous_scale="Reds")
    mapa.update_xaxes(visible=False)
    mapa.update_yaxes(visible=False)
    mapa.update_layout(title=dict(text="Mapa de error absoluto (rojo = más pérdida)", x=0.01, font_size=15), margin=dict(l=0, r=0, t=45, b=0))
    ks = [k for k in (5, 20, 50, 100, 200, 300) if k <= n] or [n]
    mse = float(((a - R) ** 2).mean())
    return Res(f"**SVD sobre la imagen** `{_nombre(ruta)}` ({h}×{w}px), cada canal RGB por separado. Con k={r} valores singulares se conserva la estructura principal; "
               "el mapa de error muestra dónde se pierde más detalle (bordes y texturas).",
               figuras=[(_imagen(a, "Original"), _imagen(R, f"Procesada · reconstrucción con k={r}")), (mapa, fe),
                        _rejilla_img([rec(k) for k in ks], [f"k = {k}" for k in ks])],
               kpis={"Energía retenida": f"{e[r - 1]:.1%}", "PSNR": f"{10 * np.log10(1 / max(mse, 1e-12)):.1f} dB",
                     "Tamaño comprimido": _tam(r, h, w), "k para 90 %": r90, "k para 95 %": r95})


def _svd_audio(ruta, r):
    import librosa

    sr = 22050
    y, _ = librosa.load(_fuente(ruta), sr=sr, mono=True, duration=30)
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    r = min(r, len(s))
    rec = lambda k: (U[:, :k] * s[:k]) @ Vt[:k]
    R = rec(r)
    lo, hi = float(np.percentile(S, 1)), float(S.max())
    fe, e, r90, r95 = _energia(s, r)
    err = _calor(np.abs(S - R), "Mapa de error absoluto (dB)", "Reds", y="mel (frecuencia)", invertir=False)
    ks = [k for k in (1, 2, 5, 10, 20, 50) if k <= len(s)] or [len(s)]
    return Res(f"**SVD sobre el espectrograma mel** de `{_nombre(ruta)}` (primeros 30 s). Cada valor singular añade una «capa» de estructura tiempo-frecuencia; "
               f"con k={r} se reconstruye la forma general y se pierde el detalle fino (rojo en el mapa de error).",
               figuras=[(_calor(S, "Original · espectrograma (dB)", "Magma", lo, hi, "mel (frecuencia)", False),
                         _calor(R, f"Procesado · reconstrucción con k={r}", "Magma", lo, hi, "mel (frecuencia)", False)), (err, fe),
                        _rejilla([rec(k) for k in ks], [f"k = {k}" for k in ks], 3, "Magma", lo, hi)],
               kpis={"Energía retenida": f"{e[r - 1]:.1%}", "Error medio": f"{np.abs(S - R).mean():.1f} dB",
                     "Tamaño comprimido": _tam(r, *S.shape), "k para 90 %": r90, "k para 95 %": r95})


def _svd_tabla(d, X, r):
    X = A._densa(X)
    n, m = X.shape
    mu = X.mean(0)
    U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    r = min(r, len(s))
    R = (U[:, :r] * s[:r]) @ Vt[:r]
    Mc = X - mu
    i, j = _sub(Mc)
    zm = float(np.percentile(np.abs(Mc[np.ix_(i, j)]), 98)) or 1.0
    fe, e, r90, r95 = _energia(s, r)
    err = _calor(np.abs(Mc - R)[np.ix_(i, j)], "Mapa de error absoluto", "Reds", 0, zm)
    return Res(f"**SVD sobre la matriz de datos** ({n} muestras × {m} características, centrada): mapa de calor de {len(i)} muestras y las {len(j)} variables de mayor "
               f"varianza, originales frente a la reconstrucción de rango {r}. El mapa de error indica qué celdas pierden más información.",
               figuras=[(_calor(Mc[np.ix_(i, j)], "Original · datos centrados", zmin=-zm, zmax=zm),
                         _calor(R[np.ix_(i, j)], f"Procesado · reconstrucción de rango {r}", zmin=-zm, zmax=zm)), (err, fe)],
               kpis={"Energía retenida": f"{e[r - 1]:.1%}", "Error relativo": f"{np.linalg.norm(Mc - R) / np.linalg.norm(Mc):.1%}",
                     "Tamaño comprimido": _tam(r, n, m), "k para 90 %": r90, "k para 95 %": r95})


def svd_comp(d, X, p, rutas):
    if d.tipo == "img":
        return _svd_img(p["archivo"], p["rango"])
    if d.tipo == "audio":
        return _svd_audio(p["archivo"], p["rango"])
    return _svd_tabla(d, X, p["rango"])


# ------------------------------------------------------------------ 9 · reconstrucción lineal: PCA vs FA vs PPCA
def _mosaico(filas, lado, k=6):
    fs = [np.concatenate([np.pad(np.clip(v.reshape(lado, lado, 3), 0, 1), ((0, 1), (0, 1), (0, 0)), constant_values=1) for v in M], 1) for M in filas]
    return (np.kron(np.concatenate(fs, 0), np.ones((k, k, 1))) * 255).astype(np.uint8)


def reconstruccion(d, X, p, rutas):
    X = A._densa(X)
    q = max(1, min(p["q"], X.shape[1] - 1))
    mu = X.mean(0)
    pc = PCA(q, random_state=rs()).fit(X)
    fa = FactorAnalysis(q, random_state=rs()).fit(X)
    Ez, W, _, _ = A._ppca(X - mu, q, p["iter"])
    R = {"PCA": pc.inverse_transform(pc.transform(X)), "Factor Analysis": fa.transform(X) @ fa.components_ + fa.mean_,
         "PPCA (EM)": Ez @ W.T + mu}
    sst = float(((X - mu) ** 2).sum())
    mse = {k: float(((v - X) ** 2).mean()) for k, v in R.items()}
    r2 = {k: 1 - float(((v - X) ** 2).sum()) / sst for k, v in R.items()}
    tabla = pd.DataFrame({
        "método": list(R), "MSE": [round(mse[k], 4) for k in R], "R² (varianza explicada)": [f"{r2[k]:.1%}" for k in R],
        "probabilístico": ["no", "sí", "sí"], "reconstrucción": ["lineal"] * len(R),
        "implementación": ["scikit-learn", "scikit-learn", "EM propio"]})
    barras = px.bar(x=list(mse), y=list(mse.values()), color=list(mse), color_discrete_sequence=PALETA, labels={"x": "", "y": "error cuadrático medio"})
    barras.update_layout(showlegend=False)
    estilo(barras, f"Error de reconstrucción con {q} variables latentes (menor = mejor)")
    nombres = ["Original", *R]
    if d.tipo == "img" and d.granularidad != "archivo":
        sc = StandardScaler().fit(d.X)
        inv = (lambda M: M * sc.scale_ + sc.mean_) if d.escalar else (lambda M: M)
        sel = np.linspace(0, len(X) - 1, 8).astype(int)
        lado = int(round((X.shape[1] / 3) ** .5))
        fig = px.imshow(_mosaico([inv(M[sel]) for M in (X, *R.values())], lado))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        for i, nm in enumerate(nombres):
            fig.add_annotation(x=-0.01, y=1 - (i + .5) / len(nombres), xref="paper", yref="paper", text=f"<b>{nm}</b>", showarrow=False, xanchor="right")
        fig.update_layout(title=dict(text="Parches originales (arriba) y reconstruidos por cada método", x=0.01, font_size=15), margin=dict(l=130, r=0, t=45, b=0))
    else:
        i, j = _sub(X)
        zm = float(np.percentile(np.abs(X[np.ix_(i, j)]), 98)) or 1.0
        fig = make_subplots(rows=1, cols=len(nombres), subplot_titles=nombres, horizontal_spacing=.02)
        for c, M in enumerate((X, *R.values()), 1):
            fig.add_trace(go.Heatmap(z=M[np.ix_(i, j)], coloraxis="coloraxis"), row=1, col=c)
        fig.update_yaxes(autorange="reversed", visible=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=True, row=1, col=1)
        fig.update_layout(coloraxis=dict(colorscale="RdBu_r", cmin=-zm, cmax=zm))
        estilo(fig, "Matriz original y reconstruida (muestras × variables)", h=420)
    mejor = min(mse, key=mse.get)
    return Res(f"**Reconstrucción con {q} variables latentes**: se comprime cada muestra a {q} números y se vuelve a expandir. La primera fila/panel es el original; "
               f"el resto, lo que recupera cada método. El de menor error aquí es **{mejor}**. PCA, FA y PPCA son métodos lineales; "
               "FA y PPCA incluyen un modelo probabilístico del ruido.",
               figuras=[fig, barras], tabla=tabla, kpis={f"MSE {k}": f"{v:.3f}" for k, v in mse.items()})


# ------------------------------------------------------------------ 10 · t-SNE / UMAP
def _con_parches(d, Z, rutas, n=45):
    from visualization import media as medios

    por = dict(rutas)
    idx = np.random.default_rng(rs() + 1).choice(len(Z), min(n, len(Z)), replace=False)
    fig = go.Figure(go.Scatter(x=Z[:, 0], y=Z[:, 1], mode="markers", marker=dict(size=3, color=GRIS, opacity=.35), hoverinfo="skip"))
    lado = max(np.ptp(Z[:, 0]), np.ptp(Z[:, 1])) * .05
    for i in idx:
        nom, y, x = d.ref[i]
        b = io.BytesIO()
        medios.parche(por[nom], y, x).save(b, "PNG")
        fig.add_layout_image(source="data:image/png;base64," + base64.b64encode(b.getvalue()).decode(), x=Z[i, 0], y=Z[i, 1], xref="x", yref="y",
                             sizex=lado, sizey=lado, xanchor="center", yanchor="middle")
    fig.update_yaxes(scaleanchor="x")
    return estilo(fig, "Parches reales superpuestos sobre el mapa", h=620)


def no_lineal(d, X, p, rutas):
    m = p["metodo"]
    Z0, e0, _ = _p2(d, X)
    res = A.tsne_umap(d, X, {"metodo": m, "dims": 2, "metrica": "euclidean", "perplexity": p.get("perplexity", 30.0),
                              "iteraciones": p.get("iteraciones", 1000), "vecinos": p.get("vecinos", 15), "min_dist": p.get("min_dist", 0.1)})
    etq, ley = _etiquetas_color(d, X, p["k_color"])
    tw0 = trustworthiness(X, Z0, n_neighbors=min(10, X.shape[0] // 2 - 1))
    figs = [(_disp(Z0, etq, f"Original · PCA 2D (trustworthiness {tw0:.2f})", e0, leyenda=ley, hover=d.hover, h=520),
             _disp(res.Z, etq, f"Procesado · {m} 2D (trustworthiness {res.kpis['Trustworthiness']})", res.ejes, leyenda=ley, hover=d.hover, h=520))]
    if d.tipo == "img" and d.ref:
        figs.append(_con_parches(d, res.Z, rutas))
    color = ("archivo de origen" if _por_origen(d) else "clase conocida") if _orig(d)[0] is not None else f"KMeans k={p['k_color']}"
    return Res(f"**{m}** frente a una proyección lineal (PCA) de los mismos datos, con los mismos colores "
               f"({color}). Los métodos no lineales conservan los vecindarios y suelen separar "
               "mejor los grupos; la *trustworthiness* (0–1) mide cuántos vecinos del mapa lo son también en el espacio original.",
               figuras=figs, kpis={"Trustworthiness PCA": f"{tw0:.3f}", f"Trustworthiness {m}": res.kpis["Trustworthiness"]})


# ------------------------------------------------------------------ 11 · LDA supervisado
def lda_comp(d, X, p, rutas):
    X = A._densa(X)
    cls, ley = _orig(d)
    fuente = "archivo de origen" if _por_origen(d) else "clase conocida"
    if cls is None:
        cls, ley, fuente = KMeans(p["k"], n_init=10, random_state=rs()).fit_predict(X), None, f"pseudoclases KMeans (k={p['k']})"
    nc = max(1, min(2, len(np.unique(cls)) - 1, X.shape[1]))
    lda = LinearDiscriminantAnalysis(n_components=nc).fit(X, cls)
    Z = lda.transform(X)
    Z = np.c_[Z, np.zeros(len(Z))] if nc == 1 else Z
    Z0, e0, _ = _p2(d, X)
    s0, s1 = (silhouette_score(M, cls, sample_size=min(len(M), 2000), random_state=rs()) for M in (Z0, Z))
    return Res(f"**LDA (análisis discriminante lineal)**, método supervisado, con clases = {fuente}. Busca los ejes que mejor separan las clases, "
               f"por lo que la silueta por clase pasa de {s0:.2f} (proyección PCA, sin usar las clases) a {s1:.2f}. "
               "Ojo: la exactitud se mide sobre los mismos datos y es optimista.",
               figuras=[(_disp(Z0, cls, "Original · PCA 2D (coloreado por clase)", e0, leyenda=ley, hover=d.hover),
                         _disp(Z, cls, "Procesado · ejes discriminantes LD1–LD2", ["LD1", "LD2"], leyenda=ley, hover=d.hover))],
               kpis={"Silueta PCA": f"{s0:.2f}", "Silueta LDA": f"{s1:.2f}", "Exactitud": f"{lda.score(X, cls):.0%}"})


# ------------------------------------------------------------------ 12 · imagen: cuantización de color con K-means
def cuantizacion(d, X, p, rutas):
    a = _cargar_img(p["archivo"], 320)
    h, w, _ = a.shape
    P = a.reshape(-1, 3)
    k = p["k"]
    ajustar = lambda kk: KMeans(kk, n_init=3, random_state=rs()).fit(P[np.random.default_rng(rs()).choice(len(P), min(len(P), 20000), replace=False)])
    km = ajustar(k)
    lab = km.predict(P)
    Q = km.cluster_centers_[lab].reshape(h, w, 3)
    mse = float(((a - Q) ** 2).mean())
    pct = np.bincount(lab, minlength=k) / len(lab)
    hexa = ["#%02x%02x%02x" % tuple((np.clip(c, 0, 1) * 255).astype(int)) for c in km.cluster_centers_]
    pal = go.Figure(go.Bar(x=[f"C{j}" for j in range(k)], y=pct, marker_color=hexa, marker_line_color="#555", marker_line_width=1))
    pal.update_yaxes(tickformat=".0%")
    estilo(pal, "Paleta de colores representativos y su peso")
    err = px.imshow(np.abs(a - Q).mean(2), color_continuous_scale="Reds")
    err.update_xaxes(visible=False)
    err.update_yaxes(visible=False)
    err.update_layout(title=dict(text="Mapa de error por píxel", x=0.01, font_size=15), margin=dict(l=0, r=0, t=45, b=0))
    ks = [2, 4, 8, 16, 32]
    def con(kk):
        m = ajustar(kk)
        return m.cluster_centers_[m.predict(P)].reshape(h, w, 3)

    prog = _rejilla_img([a] + [con(kk) for kk in ks], ["Original"] + [f"K = {kk}" for kk in ks])
    n_orig = len(np.unique((P * 255).astype(np.uint8), axis=0))
    return Res(f"**Cuantización de color con K-means** de `{_nombre(p['archivo'])}`: cada píxel pasa a ser el color medio de su clúster, así que la imagen usa solo {k} colores "
               "(compresión y segmentación visual). Abajo se ve cómo mejora al subir K.",
               figuras=[(_imagen(a, f"Original · {n_orig:,} colores distintos"), _imagen(Q, f"Procesada · K-means con {k} colores")), (pal, err), prog],
               kpis={"Colores originales": f"{n_orig:,}", "Colores K": k, "PSNR": f"{10 * np.log10(1 / max(mse, 1e-12)):.1f} dB",
                     "Tamaño aprox.": f"{(h * w * int(np.ceil(np.log2(k))) + k * 24) / (h * w * 24):.0%}"})


# ------------------------------------------------------------------ 13 · imagen: escala de grises y negativo
def gris_negativo(d, X, p, rutas):
    a = _cargar_img(p["archivo"])
    g = a @ LUM
    g3 = np.repeat(g[:, :, None], 3, 2)
    hist = go.Figure([go.Histogram(x=g.ravel(), nbinsx=64, name="gris original", opacity=.7),
                      go.Histogram(x=(1 - g).ravel(), nbinsx=64, name="negativo", opacity=.7)])
    hist.update_layout(barmode="overlay")
    estilo(hist, "Histograma de intensidades: original frente a negativo", x="intensidad (0 = negro, 1 = blanco)", y="píxeles")
    return Res(f"**Operaciones básicas** sobre `{_nombre(p['archivo'])}`: escala de grises (luminancia = 0,299 R + 0,587 G + 0,114 B) y negativo (1 − intensidad). "
               "El negativo refleja el histograma respecto al centro.",
               figuras=[_rejilla_img([a, g3, 1 - a, 1 - g3], ["Original", "Escala de grises", "Negativo (color)", "Negativo (grises)"], 4), hist],
               kpis={"Brillo original": f"{g.mean():.2f}", "Brillo negativo": f"{1 - g.mean():.2f}", "Contraste (σ)": f"{g.std():.2f}"})




# ------------------------------------------------------------------ comparativas de varias ejecuciones 
def _ejecutar(nombre, d, X, params, **extra):
    """Ejecuta un algoritmo del registro con parámetros validados (`extra` se añade ya validado)."""
    from core import registry
    from .preprocessing import densificar

    e = registry.obtener(nombre)
    vals = registry.validar_parametros(e, d.tipo, params)
    Xe = X if e.acepta_dispersa else densificar(X)
    return e.resolver()(d, Xe, {**vals, **extra}), vals


CRITERIOS = {"silueta ↑": True, "Davies-Bouldin ↓": False, "Calinski-Harabasz ↑": True, "ruido % ↓": False,
             "equilibrio ↑": True, "estabilidad (ARI) ↑": True}


def _equilibrio(lab) -> float:
    """Entropía normalizada de los tamaños de grupo (1 = grupos del mismo tamaño, 0 = uno lo concentra todo)."""
    lab = np.asarray(lab)
    c = np.bincount(lab[lab != -1] - lab[lab != -1].min()) if (lab != -1).any() else np.array([])
    c = c[c > 0]
    if len(c) < 2:
        return 0.0
    q = c / c.sum()
    return float(-(q * np.log(q)).sum() / np.log(len(c)))


def validez(m: dict) -> str:
    """Motivo por el que una configuración no es comparable con las demás ('' = comparable)."""
    if m["grupos"] == 0:
        return "todas las muestras son ruido"
    if m["grupos"] == 1:
        return "un único grupo: las métricas internas no están definidas"
    if m["ruido"] > 0.8:
        return f"{m['ruido']:.0%} de ruido: las métricas describen solo una minoría"
    if not np.isfinite(m.get("silueta", np.nan)):
        return "métricas no calculables"
    return ""


def _fila(nombre, res, X, extra=None, d=None):
    from .metrics import evaluar

    m = evaluar(X, res.etiquetas)
    motivo = validez(m)
    f = {"configuración": nombre, **(extra or {}), "grupos": m["grupos"], "ruido % ↓": round(100 * m["ruido"], 1),
         "silueta ↑": round(m["silueta"], 3) if np.isfinite(m["silueta"]) else None,
         "Davies-Bouldin ↓": round(m["davies_bouldin"], 3) if np.isfinite(m["davies_bouldin"]) else None,
         "Calinski-Harabasz ↑": round(m["calinski_harabasz"], 1) if np.isfinite(m["calinski_harabasz"]) else None,
         "equilibrio ↑": round(_equilibrio(res.etiquetas), 3)}
    if d is not None:
        for campo, v in A.validacion_externa(d, res.etiquetas).items():
            f[f"ARI vs {campo}"] = round(v["ARI"], 3)
    f["comparable"] = "sí" if not motivo else f"no: {motivo}"
    return f


def ranking(t: pd.DataFrame) -> pd.DataFrame:
    """Ranking multicriterio (Borda): promedio del puesto en cada criterio disponible, sin depender de una sola métrica.
    Las configuraciones no comparables quedan fuera del ranking (con su motivo)."""
    t = t.copy()
    ok = t["comparable"] == "sí"
    cols = [c for c in CRITERIOS if c in t and t.loc[ok, c].notna().sum() >= 2]
    puestos = pd.DataFrame(index=t.index)
    for c in cols:
        puestos[c] = t.loc[ok, c].rank(ascending=not CRITERIOS[c], method="min")
    t["puesto medio"] = puestos.mean(axis=1).round(2).where(ok)
    t["posición"] = t["puesto medio"].rank(method="min")
    t.attrs["criterios"] = cols
    return t.sort_values(["posición", "configuración"], na_position="last").reset_index(drop=True)


def _mapa_criterios(t: pd.DataFrame):
    """Mapa de calor de criterios normalizados a 0-1 (1 = mejor) para ver de un vistazo en qué destaca cada configuración."""
    cols = t.attrs.get("criterios") or [c for c in CRITERIOS if c in t]
    ok = t[t["comparable"] == "sí"]
    if ok.empty or not cols:
        return None
    M = []
    for c in cols:
        v = ok[c].astype(float)
        rng = v.max() - v.min()
        z = (v - v.min()) / rng if rng > 0 else v * 0 + 1
        M.append(z if CRITERIOS[c] else 1 - z)
    M = np.array(M).T
    fig = px.imshow(M, x=cols, y=ok["configuración"].astype(str).tolist(), color_continuous_scale=ESCALA_CONTINUA, zmin=0, zmax=1,
                    text_auto=".2f", aspect="auto", labels={"color": "normalizado (1 = mejor)"})
    return estilo(fig, "Criterios normalizados (1 = mejor en ese criterio)", h=200 + 40 * len(ok))


def _estabilidad_runs(nombre, d, X, params, base, reps=4, frac=0.8, ctx=None, prefijo=""):
    """ARI entre el agrupamiento con todos los datos y los obtenidos con submuestras del {frac:.0%} (bootstrap sin reemplazo)."""
    from .preprocessing import punto

    n = X.shape[0]
    aris, consist, cuenta = [], np.zeros(n), np.zeros(n)
    for r in range(reps):
        punto(None, f"{prefijo}estabilidad {r + 1}/{reps}")
        idx = np.sort(np.random.default_rng(rs() + 101 + r).choice(n, max(int(frac * n), 5), replace=False))
        res, _ = _ejecutar(nombre, d.tomar(idx), X[idx], params, _rapido=True)
        lr, lb = np.asarray(res.etiquetas), np.asarray(base)[idx]
        aris.append(float(adjusted_rand_score(lb, lr)))
        mapa = {c: pd.Series(lb[lr == c]).mode().iloc[0] for c in np.unique(lr)}      # cada grupo nuevo -> grupo base mayoritario
        consist[idx] += np.array([mapa[c] == b for c, b in zip(lr, lb)], dtype=float)
        cuenta[idx] += 1
    return np.array(aris), np.divide(consist, cuenta, out=np.full(n, np.nan), where=cuenta > 0)


def _rejilla_dispersion(Z, ejes, resultados, d, cols=3):
    n = len(resultados)
    filas = -(-n // cols)
    fig = make_subplots(rows=filas, cols=cols, subplot_titles=[t for t, _ in resultados], horizontal_spacing=.04, vertical_spacing=.12)
    for i, (t, lab) in enumerate(resultados):
        nom, orden, cmap = categorias(lab)
        nom = np.asarray(nom)
        for g in orden:
            m = nom == g
            fig.add_trace(go.Scattergl(x=Z[m, 0], y=Z[m, 1], mode="markers", showlegend=False, hoverinfo="skip",
                                       marker=dict(size=4 if g != "ruido" else 6, color=cmap[g], opacity=.75, symbol="x" if g == "ruido" else "circle")),
                          row=i // cols + 1, col=i % cols + 1)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return estilo(fig, "Misma proyección, distintas agrupaciones", h=330 * filas)


def comparar_algoritmos(d, X, nombres: list, params_por_alg: dict | None = None, ctx=None, estabilidad: bool = True, reps: int = 3) -> Res:
    """Varios algoritmos de agrupación sobre el mismo conjunto y la misma proyección, con ranking multicriterio."""
    params_por_alg = params_por_alg or {}
    Z, ejes, _ = A._proy(d, X)                         # la proyección se calcula una sola vez y se reutiliza
    resultados, filas = [], []
    for i, nm in enumerate(nombres):
        if ctx is not None:
            ctx.punto(i / len(nombres), f"Comparando: {nm}")
        try:
            res, vals = _ejecutar(nm, d, X, params_por_alg.get(nm, {}))
        except Exception as e:  # noqa: BLE001 - un algoritmo que falla no invalida la comparación de los demás
            filas.append({"configuración": nm, "comparable": f"no: falló ({str(e)[:80]})"})
            continue
        if res.etiquetas is None:
            continue
        resultados.append((nm, res.etiquetas))
        f = _fila(nm, res, X, d=d)
        if estabilidad and f["comparable"] == "sí":
            aris, _ = _estabilidad_runs(nm, d, X, vals, res.etiquetas, reps, ctx=ctx, prefijo=f"{nm}: ")
            f["estabilidad (ARI) ↑"] = round(float(aris.mean()), 3)
        filas.append(f)
    if not resultados:
        raise ValueError("Ninguno de los algoritmos elegidos produjo grupos")
    t = ranking(pd.DataFrame(filas))
    validos = t[t["comparable"] == "sí"]
    mejor = validos.iloc[0]["configuración"] if len(validos) else "—"
    excl = t[t["comparable"] != "sí"]
    figs = [_rejilla_dispersion(Z, ejes, resultados, d)]
    mapa = _mapa_criterios(t)
    if mapa is not None:
        figs.append(mapa)
    crit = ", ".join(t.attrs.get("criterios", []))
    informe = (f"**Comparativa de {len(resultados)} algoritmos** sobre los mismos {d.n:,} datos y la misma proyección.\n\n"
               f"**Ranking multicriterio** (puesto medio en: {crit or '—'}): mejor configuración **{mejor}**. "
               "No se decide por una sola métrica: cada criterio aporta su puesto y se promedia.")
    if len(excl):
        informe += "\n\n**Comparaciones no válidas** (fuera del ranking): " + "; ".join(f"{r['configuración']} → {r['comparable'][4:]}" for _, r in excl.iterrows())
    if estabilidad:
        informe += f"\n\n*Estabilidad*: ARI medio entre el resultado completo y {reps} submuestras del 80 % (1 = siempre los mismos grupos)."
    return Res(informe, figuras=figs, tabla=t, kpis={"Algoritmos": len(resultados), "Comparables": len(validos), "Mejor (ranking)": mejor})


def comparar_parametros(d, X, nombre: str, parametro: str, valores: list, base: dict | None = None, ctx=None, estabilidad: bool = True, reps: int = 3) -> Res:
    """Análisis de sensibilidad: un mismo algoritmo con distintos valores de un parámetro."""
    Z, ejes, _ = A._proy(d, X)
    resultados, filas, etqs = [], [], []
    for i, v in enumerate(valores):
        if ctx is not None:
            ctx.punto(i / len(valores), f"{nombre}: {parametro}={v}")
        try:
            res, vals = _ejecutar(nombre, d, X, {**(base or {}), parametro: v})
        except Exception as e:  # noqa: BLE001
            filas.append({"configuración": f"{parametro}={v}", parametro: v, "comparable": f"no: falló ({str(e)[:80]})"})
            continue
        if res.etiquetas is None:
            continue
        resultados.append((f"{parametro}={v}", res.etiquetas))
        etqs.append(res.etiquetas)
        f = _fila(f"{parametro}={v}", res, X, {parametro: v}, d=d)
        f["ARI vs anterior"] = round(float(adjusted_rand_score(etqs[-2], etqs[-1])), 3) if len(etqs) > 1 else None
        if estabilidad and f["comparable"] == "sí":
            aris, _ = _estabilidad_runs(nombre, d, X, vals, res.etiquetas, reps, ctx=ctx, prefijo=f"{parametro}={v}: ")
            f["estabilidad (ARI) ↑"] = round(float(aris.mean()), 3)
        filas.append(f)
    if not resultados:
        raise ValueError("El algoritmo elegido no produce grupos")
    t = ranking(pd.DataFrame(filas))
    t = t.sort_values(parametro).reset_index(drop=True) if parametro in t else t
    lin = make_subplots(specs=[[{"secondary_y": True}]])
    for c in ("silueta ↑", "Davies-Bouldin ↓", "estabilidad (ARI) ↑", "ARI vs anterior"):
        if c in t and t[c].notna().any():
            lin.add_scatter(x=t[parametro], y=t[c], mode="lines+markers", name=c, secondary_y=False)
    lin.add_bar(x=t[parametro], y=t["grupos"], name="grupos", opacity=.25, secondary_y=True)
    estilo(lin, f"Sensibilidad a «{parametro}»", x=parametro)
    sil = pd.to_numeric(t.get("silueta ↑"), errors="coerce").dropna()
    cv = float(sil.std() / abs(sil.mean())) if len(sil) > 1 and sil.mean() else np.nan
    cambios = pd.to_numeric(t.get("ARI vs anterior"), errors="coerce").dropna()
    veredicto = ("**muy sensible**: pequeños cambios del parámetro reorganizan los grupos" if len(cambios) and cambios.min() < 0.5
                 else "**robusto**: los grupos cambian poco entre valores vecinos" if len(cambios) else "—")
    validos = t[t["comparable"] == "sí"]
    mejor = validos.sort_values("posición").iloc[0]["configuración"] if len(validos) else "—"
    informe = (f"**Análisis de sensibilidad de {nombre}** a `{parametro}` ({len(valores)} valores, mismos datos). El resultado es {veredicto} "
               f"(ARI mínimo entre valores consecutivos: {cambios.min():.2f}; variación relativa de la silueta: {cv:.0%}). "
               f"Mejor configuración por ranking multicriterio: **{mejor}**." if len(cambios) else
               f"**Análisis de sensibilidad de {nombre}** a `{parametro}`. Mejor configuración: **{mejor}**.")
    excl = t[t["comparable"] != "sí"]
    if len(excl):
        informe += "\n\n**No comparables**: " + "; ".join(f"{r['configuración']} → {r['comparable'][4:]}" for _, r in excl.iterrows())
    figs = [_rejilla_dispersion(Z, ejes, resultados, d), lin]
    mapa = _mapa_criterios(t)
    if mapa is not None:
        figs.append(mapa)
    return Res(informe, figuras=figs, tabla=t, kpis={"Configuraciones": len(resultados), "Mejor (ranking)": mejor,
                                                      "Sensibilidad": "alta" if len(cambios) and cambios.min() < 0.5 else "baja"})


def estabilidad(d, X, nombre: str, params: dict | None = None, reps: int = 8, frac: float = 0.8, ctx=None) -> Res:
    """¿Se obtienen los mismos grupos con otros subconjuntos de los datos? (submuestreo repetido + ARI)."""
    res, vals = _ejecutar(nombre, d, X, params or {})
    if res.etiquetas is None:
        raise ValueError("El algoritmo no produce grupos")
    aris, cons = _estabilidad_runs(nombre, d, X, vals, res.etiquetas, reps, frac, ctx)
    Z, ejes, _ = A._proy(d, X)
    hist = px.histogram(x=aris, nbins=max(5, reps), range_x=[-0.1, 1.0], labels={"x": "ARI frente al resultado completo"})
    estilo(hist, f"ARI en {reps} submuestras del {frac:.0%}", y="repeticiones")
    df = pd.DataFrame({"x": Z[:, 0], "y": Z[:, 1], "consistencia": cons, "hover": d.hover})
    sc = px.scatter(df, x="x", y="y", color="consistencia", color_continuous_scale=ESCALA_CONTINUA, range_color=[0, 1], hover_name="hover",
                    labels={"x": ejes[0], "y": ejes[1]})
    estilo(sc, "Consistencia por muestra (verde = siempre en el mismo grupo)", h=480)
    nom, orden, _ = categorias(res.etiquetas)
    por_grupo = pd.DataFrame({"grupo": nom, "consistencia": cons}).groupby("grupo")["consistencia"].agg(["mean", "count"]).reindex(orden)
    tabla = por_grupo.reset_index().rename(columns={"mean": "consistencia media", "count": "muestras"}).round(3)
    m = float(aris.mean())
    nivel = "alta" if m >= 0.8 else "media" if m >= 0.5 else "baja"
    return Res(f"**Estabilidad de {nombre}**: ARI medio {m:.2f} ± {aris.std():.2f} en {reps} submuestras del {frac:.0%} → estabilidad **{nivel}**. "
               "Los grupos con consistencia baja dependen de qué muestras se incluyan y no deberían interpretarse como estructura real.",
               figuras=[sc, hist], tabla=tabla, kpis={"ARI medio": f"{m:.2f}", "ARI mínimo": f"{aris.min():.2f}", "Estabilidad": nivel})
