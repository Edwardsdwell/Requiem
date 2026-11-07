"""Gráficos Plotly compartidos: estilo común y dispersión interactiva 2D/3D/mapa.

Plotly se importa al cargar este módulo, que a su vez solo se importa cuando se dibuja o se ejecuta
un algoritmo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.cluster import KMeans

from .maps import centro_de, zoom_para

PALETA = ["#C62828", "#008A52", "#00AFC1", "#F5A000", "#3D5AFE", "#D81B60", "#7E57C2", "#F4511E", "#00695C", "#8E9AAF", "#7CB342", "#EC407A", "#00838F", "#6D4C41", "#3949AB", "#9E9D24"]
ESCALA_CONTINUA = [[0, "#D9D7D2"], [.5, "#00A6A6"], [1, "#B21E35"]]
_EJES = dict(gridcolor="rgba(27,27,26,.10)", zerolinecolor="rgba(27,27,26,.24)", linecolor="rgba(27,27,26,.35)")
_TEMA = pio.templates["plotly_white"].layout
_TEMA.update(
    colorway=PALETA, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#1B1B1A", family="Georgia, serif"),
    xaxis=_EJES, yaxis=_EJES,
    scene=dict(xaxis={**_EJES, "backgroundcolor": "rgba(0,0,0,0)"},
               yaxis={**_EJES, "backgroundcolor": "rgba(0,0,0,0)"},
               zaxis={**_EJES, "backgroundcolor": "rgba(0,0,0,0)"}),
    hoverlabel=dict(bgcolor="#FBFAF7", font_color="#1B1B1A", bordercolor="#7A1F1F",
                   font=dict(family="monospace")))
pio.templates.default = "plotly_white"
GRIS_RUIDO = "#6B6A67"


def estilo(fig, titulo, x=None, y=None, h=360):
    fig.update_layout(title=dict(text=titulo, x=0.01, font=dict(family="Georgia, serif", size=19)),
                      height=h, margin=dict(l=50, r=20, t=55, b=45),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    if x:
        fig.update_xaxes(title_text=x)
    if y:
        fig.update_yaxes(title_text=y)
    return fig


def categorias(lab):
    """Etiquetas -> (nombres 'G0'/'ruido', orden, mapa de colores). -1 es ruido, en gris."""
    lab = np.asarray(lab)
    ids = sorted({int(l) for l in lab if l != -1})
    orden = [f"G{i}" for i in ids] + (["ruido"] if (lab == -1).any() else [])
    cmap = {f"G{i}": PALETA[j % len(PALETA)] for j, i in enumerate(ids)} | {"ruido": GRIS_RUIDO}
    return ["ruido" if l == -1 else f"G{l}" for l in lab], orden, cmap


def histograma(x, titulo, etiqueta="valor", bins=40, y="muestras"):
    """Histograma de una característica, de distancias o de errores de reconstrucción."""
    fig = px.histogram(x=np.asarray(x), nbins=bins, labels={"x": etiqueta})
    return estilo(fig, titulo, y=y)


def dispersion(d, res, vista, color, tam, opac, variable=None, k=3):
    """vista: '2D' | '3D' | 'Mapa'. color: 'Grupo' | 'KMeans' | 'Archivo' | 'Variable' | 'Ninguno'."""
    Z = res.Z
    n_ej = min(3, Z.shape[1])
    df = pd.DataFrame({"hover": d.hover, **{f"e{i}": Z[:, i] for i in range(n_ej)}})
    etq = {f"e{i}": res.ejes[i] for i in range(n_ej)}
    kw = {}
    if color in ("Grupo", "KMeans"):
        lab = res.etiquetas if color == "Grupo" else KMeans(k, n_init=5, random_state=0).fit_predict(Z)
        df["color"], orden, cmap = categorias(lab)
        kw = dict(color="color", color_discrete_map=cmap, category_orders={"color": orden})
        if "ruido" in orden and vista != "Mapa":                     # el ruido se distingue también por la forma (✕)
            kw.update(symbol="color", symbol_map={g: ("x" if g == "ruido" else "circle") for g in orden})
    elif color == "Archivo":
        df["color"] = d.origen
        kw = dict(color="color", color_discrete_sequence=PALETA)
    elif color == "Variable":
        col = d.X[:, d.nombres.index(variable)]
        df["color"] = np.asarray(col.todense()).ravel() if hasattr(col, "todense") else col
        kw = dict(color="color", color_continuous_scale=ESCALA_CONTINUA)
        etq["color"] = variable
    hd = {f"e{i}": ":.3f" for i in range(n_ej)} if len(df) <= 5000 else {}
    base = dict(hover_name="hover", opacity=opac, labels=etq, **kw)
    gl = len(df) > 1500                                          # renderizado WebGL para conjuntos grandes

    if vista == "Mapa":
        lat, lon = d.mapa[:, 0], d.mapa[:, 1]
        fig = px.scatter_map(df.assign(lat=lat, lon=lon), lat="lat", lon="lon", zoom=zoom_para(lat, lon), center=centro_de(lat, lon),
                             map_style="open-street-map", **base)
    elif vista == "3D":
        fig = px.scatter_3d(df, x="e0", y="e1", z="e2", hover_data=hd, **base)
    else:
        fig = px.scatter(df, x="e0", y="e1", hover_data=hd, render_mode="webgl" if gl else "auto", **base)
    if vista == "Mapa":
        fig.update_traces(marker_size=max(tam, 11), marker_opacity=max(opac, .9), selector=dict(type="scattermap"))
    else:
        fig.update_traces(marker_size=max(tam, 7), marker_opacity=max(opac, .85),
                          marker_line_width=.7, marker_line_color="#0E0E0D")

    C = res.centros
    if C is not None and C.shape[1] >= (3 if vista == "3D" else 2):                                  # centroides 
        if vista == "Mapa":
            ll = d.a_latlon(C)
            fig.add_scattermap(lat=ll[:, 0], lon=ll[:, 1], mode="markers", name="centros", marker=dict(size=19, color="#0E0E0D", symbol="star"))
        elif vista == "3D":
            fig.add_scatter3d(x=C[:, 0], y=C[:, 1], z=C[:, 2], mode="markers", name="centros", marker=dict(size=7, color="white", symbol="x"))
        else:
            fig.add_scatter(x=C[:, 0], y=C[:, 1], mode="markers", name="centros",
                            marker=dict(size=13, color="white", symbol="x", line=dict(width=1, color="black")))
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
    if vista == "3D":                                           # botón para girar la cámara
        ang = np.linspace(0, 2 * np.pi, 120)
        fig.frames = [go.Frame(layout=dict(scene_camera=dict(eye=dict(x=1.9 * np.cos(a), y=1.9 * np.sin(a), z=0.9)))) for a in ang]
        play = dict(frame=dict(duration=40, redraw=True), fromcurrent=True, transition=dict(duration=0), mode="immediate")
        stop = dict(frame=dict(duration=0, redraw=False), mode="immediate")
        fig.update_layout(updatemenus=[dict(type="buttons", showactive=False, x=0.01, y=0.02, bgcolor="#FBFAF7", font_color="#1B1B1A", buttons=[
            dict(label="↻ Girar", method="animate", args=[None, play]), dict(label="Pausa", method="animate", args=[[None], stop])])])
    return fig


def _en_poligono(P, V):
    """Qué puntos P (n,2) caen dentro del polígono V (m,2): trazado de rayos."""
    x, y, dentro, j = P[:, 0], P[:, 1], np.zeros(len(P), bool), len(V) - 1
    for i in range(len(V)):
        (xi, yi), (xj, yj) = V[i], V[j]
        dentro ^= ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi)
        j = i
    return dentro


def indices_seleccion(Z, sel):
    """Índices de las muestras dentro de las cajas y lazos (polígonos) que devuelve `st.plotly_chart(on_select=...)`."""
    m = np.zeros(len(Z), bool)
    for b in sel.get("box", []):
        (x0, x1), (y0, y1) = sorted(b["x"]), sorted(b["y"])
        m |= (Z[:, 0] >= x0) & (Z[:, 0] <= x1) & (Z[:, 1] >= y0) & (Z[:, 1] <= y1)
    for l in sel.get("lasso", []):
        m |= _en_poligono(Z[:, :2], np.c_[l["x"], l["y"]])
    return np.flatnonzero(m)


# ------------------------------------------------------------------ exportación
def a_html(figuras) -> bytes:
    """Todas las figuras en un único HTML interactivo (Plotly desde CDN: el archivo pesa poco)."""
    fl = [f for g in figuras for f in (g if isinstance(g, tuple) else (g,))]
    cuerpo = "".join(f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False) for i, f in enumerate(fl))
    return f"<meta charset='utf-8'><body style='background:#F4F2EE;color:#1B1B1A'>{cuerpo}".encode()


def a_png(fig) -> bytes | None:
    """PNG de una figura si `kaleido` está instalado (dependencia opcional); si no, None."""
    try:
        return pio.to_image(fig, format="png", scale=2)
    except Exception:
        return None
