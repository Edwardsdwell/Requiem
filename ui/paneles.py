"""Paneles añadidos a la interfaz: calidad e inventario de datos, exploración por grupo, sugerencia de parámetros,
comparación entre ejecuciones y auditoría. Solo presentan: toda la lógica vive en `core.engine.Motor`."""
from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import streamlit as st

CFG_PLOT = {"displaylogo": False}


def tarjetas(cifras: dict) -> None:
    st.markdown('<div class="kpis">' + "".join(f"<div class='kpi'><b>{escape(str(v))}</b><span>{escape(str(k))}</span></div>"
                                               for k, v in cifras.items()) + "</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ importación con informe completo
def informe_importacion(rep: dict) -> None:
    est = rep.get("estados", {})
    tarjetas({"Copiados": rep["copiados"], "Repetidos (mismo contenido)": rep["duplicados"], "Faltantes": rep["faltan"],
              "Inválidos": rep["invalidos"], "Coordenadas inválidas": len(rep["geo_invalidos"]), "Filas con error": len(rep["filas_con_error"])})
    st.caption(f"Origen: `{rep['origen']}` · archivos en `{rep['carpeta_raw']}`")
    filas = [dict(tabla=t, en_origen=rep["filas"].get(t), insertadas=rep["insertadas"].get(t), ya_existian=rep["ya_existian"].get(t, 0),
                  **{f"estado: {k}": v for k, v in est.get(t, {}).items()}) for t in rep["filas"]]
    st.dataframe(pd.DataFrame(filas).fillna(""), hide_index=True)
    if rep["motivos_invalidos"]:
        st.warning("Archivos inválidos (conservados con su motivo, no se analizan): " +
                   "; ".join(f"{m} ({n})" for m, n in rep["motivos_invalidos"].items()))
    if rep["geo_invalidos"]:
        with st.expander(f"{len(rep['geo_invalidos'])} coordenadas inválidas conservadas"):
            st.dataframe(pd.DataFrame(rep["geo_invalidos"], columns=["event_id", "motivo"]), hide_index=True)
    if rep["faltantes"]:
        with st.expander(f"{len(rep['faltantes'])} registros sin archivo (quedan pendientes)"):
            st.write(", ".join(map(str, rep["faltantes"][:200])))
    if rep["filas_con_error"]:
        with st.expander(f"{len(rep['filas_con_error'])} filas que no se pudieron insertar"):
            st.dataframe(pd.DataFrame(rep["filas_con_error"], columns=["tabla", "registro", "error"]), hide_index=True)
    if rep["columnas_ignoradas"]:
        st.caption("Columnas del original que no existen en el esquema nuevo: " +
                   "; ".join(f"{t}: {', '.join(c)}" for t, c in rep["columnas_ignoradas"].items()))
    v = rep.get("verificacion", {})
    if v:
        st.success(f"Verificación posterior: {v.get('validos', 0)} de {v.get('revisados', 0)} archivos válidos presentes y con SHA-256 correcto; "
                   f"faltantes {v.get('faltantes', 0)}, dañados {v.get('danados', 0)}.")


# ------------------------------------------------------------------ calidad e inventario
def _graf_calidad(rep: dict):
    import plotly.express as px

    from visualization.charts import estilo

    figs = []
    a = rep.get("audio", {}).get("tabla")
    if isinstance(a, pd.DataFrame) and a.duration_s.notna().any():
        f = px.histogram(a, x="duration_s", color="formato", nbins=30, labels={"duration_s": "duración (s)"})
        figs.append(estilo(f, "Audio: duración por formato", y="archivos"))
    im = rep.get("img", {}).get("tabla")
    if isinstance(im, pd.DataFrame) and im.width.notna().any():
        f = px.scatter(im, x="width", y="height", color="formato", hover_name="image_id", log_x=True, log_y=True,
                       labels={"width": "ancho (px, escala log)", "height": "alto (px, escala log)"})
        figs.append(estilo(f, "Imágenes: resolución"))
    t = rep.get("texto", {}).get("tabla")
    if isinstance(t, pd.DataFrame) and len(t):
        f = px.histogram(x=t.original.str.split().str.len(), color=t.field.str.replace(r"_\d+$", "", regex=True), nbins=30,
                         labels={"x": "palabras por texto", "color": "campo"})
        figs.append(estilo(f, "Texto: longitud por campo", y="textos"))
    g = rep.get("geo", {})
    if g.get("paises"):
        f = px.bar(x=list(g["paises"]), y=list(g["paises"].values()), labels={"x": "país", "y": "puntos válidos"})
        figs.append(estilo(f, "Coordenadas: puntos por país"))
    e = rep.get("eventos", {})
    if e.get("cobertura"):
        f = px.bar(x=list(e["cobertura"]), y=list(e["cobertura"].values()), labels={"x": "modalidad", "y": "eventos con dato válido"})
        f.add_hline(y=e["eventos"], line_dash="dash", annotation_text=f"{e['eventos']} eventos")
        figs.append(estilo(f, "Completitud: eventos con cada modalidad"))
    return figs


def pagina_calidad(m) -> None:
    from core.errors import describir
    from data.inventario import informe_markdown, informe_zip
    st.markdown("""
    ### Evaluación de calidad y detección de anomalías
    Esta sección permite identificar registros que presentan
    comportamientos atípicos o posibles inconsistencias.

    Se estudian:

    - Coordenadas geográficas fuera de rango o sospechosas.
    - Archivos multimedia corruptos o inválidos.
    - Registros duplicados.
    - Características acústicas y visuales atípicas.
    - Observaciones que se alejan de los agrupamientos principales.

    Los resultados representan indicadores para revisión,
    no confirmaciones automáticas de errores biológicos.
    """)
    
    
    c = st.columns(3)
    if c[0].button("Generar informe de calidad", type="primary", key="bt_calidad"):
        st.session_state["t_calidad"] = m.lanzar_informe_calidad()
    if c[1].button("Verificar integridad del almacén", key="bt_verif"):
        st.session_state["t_verif"] = m.lanzar_verificacion()
    if c[2].button("Ver inventario completo", key="bt_inv"):
        st.session_state["inventario"] = m.inventario()
    from ui.pages import seguimiento

    t = seguimiento(m, "t_verif")
    if t is not None:
        r = t.resultado
        (st.success if not (r["faltantes"] or r["danados"] or r["invalidos"]) else st.warning)(
            f"Revisados {r['revisados']}: válidos {r['validos']}, faltantes {len(r['faltantes'])}, dañados {len(r['danados'])}, "
            f"ilegibles {len(r['invalidos'])}. Los cambios de estado quedaron registrados con su motivo.")
    t = seguimiento(m, "t_calidad")
    if t is not None:
        st.session_state["calidad_rep"] = t.resultado
    rep = st.session_state.get("calidad_rep")
    if rep:
        st.subheader("Resumen por modalidad")
        st.dataframe(rep["resumen"], hide_index=True)
        for r in rep["recomendaciones"]:
            st.info(r)
        if len(rep["motivos"]):
            st.subheader("Registros no válidos y motivos")
            st.dataframe(rep["motivos"], hide_index=True)
        figs = _graf_calidad(rep)
        for i in range(0, len(figs), 2):
            for col, f in zip(st.columns(2), figs[i:i + 2]):
                col.plotly_chart(f, theme=None, config=CFG_PLOT, key=f"cal_f{i}_{id(f)}")
        with st.expander("Detalle por modalidad"):
            for k, titulo in (("audio", "Audio"), ("img", "Imágenes"), ("texto", "Texto"), ("geo", "Coordenadas"), ("eventos", "Eventos")):
                if k in rep:
                    st.markdown(f"**{titulo}**")
                    st.json({a: b for a, b in rep[k].items() if not isinstance(b, pd.DataFrame)}, expanded=False)
        d = st.columns(2)
        d[0].download_button("Descargar informe (ZIP: Markdown + CSV)", informe_zip(rep), "informe_calidad.zip", "application/zip", key="dl_cal_zip")
        d[1].download_button("Descargar informe (Markdown)", informe_markdown(rep).encode("utf-8"), "informe_calidad.md", "text/markdown", key="dl_cal_md")
    inv = st.session_state.get("inventario")
    if inv is not None:
        st.subheader("Inventario de registros")
        f = st.columns(3)
        mods = f[0].multiselect("Modalidad", sorted(inv.modalidad.unique()), key="inv_mod")
        ests = f[1].multiselect("Estado", ["válido", "inválido", "pendiente"], key="inv_est")
        txt = f[2].text_input("Buscar (especie, id, motivo)", key="inv_txt")
        v = inv
        if mods:
            v = v[v.modalidad.isin(mods)]
        if ests:
            v = v[v.estado.isin(ests)]
        if txt:
            v = v[v.astype(str).apply(lambda r: r.str.contains(txt, case=False, regex=False)).any(axis=1)]
        st.caption(f"{len(v)} de {len(inv)} registros")
        st.dataframe(v, hide_index=True, height=380)
        st.download_button("Descargar inventario (CSV)", v.to_csv(index=False).encode("utf-8"), "inventario.csv", "text/csv", key="dl_inv")
    st.caption("Las descripciones y los nombres comunes se leen de las tablas de texto de aves_db.")
    with st.expander("Auditoría de operaciones sobre los datos"):
        try:
            st.dataframe(m.auditoria(200), hide_index=True)
        except Exception as e:  # noqa: BLE001
            st.caption(describir(e)["mensaje"])


# ------------------------------------------------------------------ exploración de los elementos de cada grupo
def _representantes(Z, lab, g, n):
    idx = np.flatnonzero(lab == g)
    if g == -1 or len(idx) <= n:
        return idx[:n]
    c = Z[idx].mean(0)
    return idx[np.argsort(((Z[idx] - c) ** 2).sum(1))[:n]]


def explorar_grupos(d, res, k: str) -> None:
    """Galería por grupo: imágenes (o parches), audios con espectrograma, fragmentos de texto o puntos en el mapa."""
    from visualization import media
    from visualization.charts import categorias

    if res.etiquetas is None or res.Z is None:
        return
    lab = np.asarray(res.etiquetas)
    nom, orden, cmap = categorias(lab)
    with st.expander("Explorar los elementos de cada grupo", expanded=False):
        c = st.columns([1.2, 1, 2])
        g_nom = c[0].selectbox("Grupo", orden, key=k + "grp", format_func=lambda x: f"{x} ({(np.asarray(nom) == x).sum()} muestras)")
        n = c[1].slider("Cuántos", 4, 48, 16, 4, key=k + "grpn")
        modo = c[2].radio("Mostrar", ["Más representativos (cerca del centro)", "Al azar"], horizontal=True, key=k + "grpm")
        g = -1 if g_nom == "ruido" else int(g_nom[1:])
        if modo.startswith("Más") and g != -1:
            idx = _representantes(res.Z[:, :2], lab, g, n)
        else:
            todos = np.flatnonzero(lab == g)
            idx = np.sort(np.random.default_rng(0).choice(todos, min(n, len(todos)), replace=False))
        st.markdown(f"<span style='color:{cmap[g_nom]}'>■</span> **{g_nom}** · {len(idx)} de {(lab == g).sum()} muestras",
                    unsafe_allow_html=True)
        if d.meta:
            meta = pd.DataFrame([{kk: v for kk, v in (d.meta[i] or {}).items() if kk in ("archivo", "especie", "clase", "event_id", "id_origen", "pais", "campo")}
                                 | {"muestra": d.hover[i]} for i in idx])
            if "especie" in meta and meta.especie.notna().any():
                st.caption("Especies en la muestra mostrada: " + ", ".join(f"{e} ({c_})" for e, c_ in meta.especie.value_counts().head(8).items()))
        if d.tipo == "img":
            try:
                st.image([media.parche(d.recursos[d.ref[i][0]], d.ref[i][1], d.ref[i][2]) for i in idx], width=96 if d.granularidad == "archivo" else 64,
                         caption=[str(d.hover[i])[:30] for i in idx])
            except Exception as e:  # noqa: BLE001
                st.caption(f"No se pudieron mostrar las imágenes: {e}")
        elif d.tipo == "audio":
            for i in idx[:4]:
                nm, t0 = d.ref[i]
                cc = st.columns([1, 2])
                cc[0].caption(f"{d.hover[i]}")
                try:
                    cc[0].audio(media.fragmento_audio_bytes(d.recursos[nm], t0, 3), format="audio/wav")
                    cc[1].plotly_chart(media.espectrograma_fig(d.recursos[nm], t0, 3), theme=None, config=CFG_PLOT, key=f"{k}esp{i}")
                except Exception as e:  # noqa: BLE001
                    cc[1].caption(f"Sin espectrograma: {e}")
        elif d.tipo == "texto":
            for i in idx[:12]:
                st.markdown(f"> {str(d.ref[i])[:320]}{'…' if len(str(d.ref[i])) > 320 else ''}")
        elif d.tipo == "geo" and d.mapa is not None:
            st.map(pd.DataFrame({"lat": d.mapa[idx, 0], "lon": d.mapa[idx, 1]}), size=40000 if d.proyeccion == "esfera" else 200)
        if d.meta:
            st.dataframe(meta, hide_index=True)


# ------------------------------------------------------------------ sugerencia de parámetros
def sugerir(m, tipo, sel, nombre, entrada, prep, max_m, key_base, vals) -> None:
    from ui.pages import mostrar_error

    if st.button("Sugerir parámetros para estos datos", key=key_base + "_sug"):
        try:
            with st.spinner("Analizando los datos…"):
                st.session_state[key_base + "_sug_r"] = m.recomendar(tipo, sel, nombre, vals, prep, max_m)
        except Exception as e:  # noqa: BLE001
            mostrar_error(e)
    rec = st.session_state.get(key_base + "_sug_r")
    if rec:
        st.dataframe(pd.DataFrame([dict(parámetro=p, sugerido=v, motivo=r) for p, (v, r) in rec.items()]), hide_index=True)
        st.button("Aplicar sugeridos", key=key_base + "_sug_ap", on_click=_aplicar_sugeridos, args=(entrada, rec, key_base))


def _aplicar_sugeridos(entrada, rec, key_base) -> None:
    """Callback: se ejecuta antes de dibujar los controles, así sus valores se pueden cambiar."""
    por = {p.clave: p for p in entrada.params}
    for clave, (v, _) in rec.items():
        q = por.get(clave)
        if q is None:
            continue
        if q.opciones:
            v = v if v in q.opciones else q.defecto
        elif q.min is not None:
            v = min(max(v, q.min), q.max)
            v = float(v) if isinstance(q.defecto, float) else int(v)
        k = f"{key_base}_{clave}"
        st.session_state.pop(k, None)                       # el control se vuelve a crear con el valor sugerido como inicial
        st.session_state.setdefault("_pendientes", {})[k] = v
    st.session_state.pop(key_base + "_sug_r", None)


def estimacion(m, tipo, sel, nombre, max_m) -> None:
    """Tiempo y memoria aproximados antes de ejecutar."""
    gran = m.cfg.img_granularidad if tipo == "img" else m.cfg.audio_granularidad if tipo == "audio" else "fragmentos"
    por_rec = {"img": 64 if gran != "archivo" else 1, "audio": 30 if gran != "archivo" else 1, "texto": 60, "geo": 70}[tipo]
    n = int(min(max_m, max(1, len(sel)) * por_rec))
    p = {"img": 192 if gran != "archivo" else 79, "audio": 43 if gran != "archivo" else 87, "texto": 400, "geo": 3}[tipo]
    e = m.estimar(tipo, nombre, n, p)
    t = f"~{e['segundos']:.0f} s" if e["segundos"] is not None else "sin ejecuciones previas para estimar el tiempo"
    st.caption(f"Estimación para ~{n:,} muestras: {t} · ~{e['memoria_mb']:.0f} MB de memoria")


# ------------------------------------------------------------------ comparación entre ejecuciones anteriores
def comparar_ejecuciones(m, h: pd.DataFrame) -> None:
    import json

    ok = h[h.clave_resultado.notna()]
    if len(ok) < 2:
        return
    with st.expander("Comparar dos ejecuciones anteriores"):
        fmt = lambda i: f"{i} · {ok[ok.run_id == i].nombre.iloc[0]} · {ok[ok.run_id == i].tipo.iloc[0]} · {ok[ok.run_id == i].created_utc.iloc[0][:19]}"
        c = st.columns(2)
        a = c[0].selectbox("Ejecución A", ok.run_id, format_func=fmt, key="cmp_a")
        b = c[1].selectbox("Ejecución B", ok.run_id, index=1, format_func=fmt, key="cmp_b")
        fa, fb = ok[ok.run_id == a].iloc[0], ok[ok.run_id == b].iloc[0]
        filas = []
        for campo in ("nombre", "tipo", "n_muestras", "n_variables", "semilla", "duracion_s", "huella"):
            filas.append(dict(campo=campo, A=str(fa[campo]), B=str(fb[campo]), igual="✓" if str(fa[campo]) == str(fb[campo]) else "≠"))
        for origen, col in (("parámetro", "params"), ("métrica", "metricas")):
            pa = json.loads(fa[col] or "{}") if isinstance(fa[col], str) else {}
            pb = json.loads(fb[col] or "{}") if isinstance(fb[col], str) else {}
            for kk in sorted(set(pa) | set(pb)):
                filas.append(dict(campo=f"{origen}: {kk}", A=str(pa.get(kk, "—")), B=str(pb.get(kk, "—")), igual="✓" if pa.get(kk) == pb.get(kk) else "≠"))
        st.dataframe(pd.DataFrame(filas), hide_index=True)
        ga, gb = m.recuperar(fa.clave_resultado), m.recuperar(fb.clave_resultado)
        if ga and gb and ga[1].etiquetas is not None and gb[1].etiquetas is not None and fa.huella == fb.huella and len(ga[1].etiquetas) == len(gb[1].etiquetas):
            from sklearn.metrics import adjusted_rand_score

            st.metric("Coincidencia de los grupos (ARI)", f"{adjusted_rand_score(ga[1].etiquetas, gb[1].etiquetas):.3f}",
                      help="1 = las dos ejecuciones forman exactamente los mismos grupos; 0 = coincidencia al azar.")
        elif fa.huella != fb.huella:
            st.caption("Las ejecuciones usan datos distintos (huella diferente): solo se comparan parámetros y métricas.")
