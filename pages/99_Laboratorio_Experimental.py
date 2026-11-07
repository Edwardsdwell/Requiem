"""Flujos de investigación sobre el mismo dataset y motor de la aplicación principal."""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Laboratorio de investigación · Aves", layout="wide")
st.markdown(
    f"<style>{(Path(__file__).resolve().parents[1] / 'ui' / 'styles.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)

from runtime import motor
from core import registry
from data import loaders
from ml.preprocessing import AUSENTES, ESCALADOS, Prep
from ui import pages as vistas


m = motor()

st.title("Laboratorio de investigación")
st.caption("Análisis reproducible de aves_db · etiquetas taxonómicas reservadas para evaluación externa")

with st.sidebar:
    max_muestras = st.slider("Máximo de muestras", 100, 10000, m.cfg.max_muestras, 100)
    semilla = st.number_input("Semilla", 0, 99999, m.cfg.semilla)
    m.cfg.semilla = int(semilla)
    prep = Prep(
        st.selectbox("Escalado", ESCALADOS, index=0),
        st.selectbox("Valores ausentes", AUSENTES, index=0),
        st.checkbox("Excluir variables constantes", True),
    )

resumen = m.resumen()
a, b, c, d, e = st.columns(5)
a.metric("Observaciones en aves_db", f"{m.registros_fuente:,}")
b.metric("Audio listo", f"{resumen['audios_ok']:,}")
c.metric("Imágenes listas", f"{resumen['imagenes_ok']:,}")
d.metric("Puntos geográficos", f"{resumen['puntos']:,}")
e.metric("Ejecuciones guardadas", f"{resumen['ejecuciones']:,}")

st.markdown("**Objetivos del laboratorio** · caracterizar modalidades · descubrir agrupamientos · contrastarlos con taxonomía · detectar problemas de calidad")
st.caption("Cada análisis usa registros sincronizados desde PostgreSQL. Especie se excluye del agrupamiento no supervisado y se usa aparte para validar grupos y entrenar el clasificador supervisado.")

tab_pares, tab_modalidad, tab_historial = st.tabs(
    ["Comparación audio–imagen", "Análisis por modalidad", "Experimentos"]
)

with tab_pares:
    st.subheader("¿Qué modalidad recupera mejor la estructura taxonómica?")
    vistas.descargar_multimedia_db(m, "lab_descargar_multimedia")
    audio = m.recursos("audio")
    imagen = m.recursos("img")
    por_evento = {"audio": {}, "img": {}}
    for recurso in audio:
        por_evento["audio"].setdefault(recurso.meta.get("event_id"), []).append(recurso)
    for recurso in imagen:
        por_evento["img"].setdefault(recurso.meta.get("event_id"), []).append(recurso)
    comunes = set(por_evento["audio"]) & set(por_evento["img"])
    comunes.discard(None)

    with m.bd() as con:
        eventos_con_referencias = con.execute("""
            SELECT COUNT(*) FROM events e
            WHERE EXISTS (SELECT 1 FROM audio a WHERE a.event_id=e.event_id)
              AND EXISTS (SELECT 1 FROM images i WHERE i.event_id=e.event_id)
        """).fetchone()[0]
        eventos_con_medios_locales = con.execute("""
            SELECT COUNT(*) FROM events e
            WHERE EXISTS (SELECT 1 FROM audio a WHERE a.event_id=e.event_id AND a.download_status='correcto' AND a.local_path IS NOT NULL)
              AND EXISTS (SELECT 1 FROM images i WHERE i.event_id=e.event_id AND i.download_status='correcto' AND i.local_path IS NOT NULL)
        """).fetchone()[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Eventos con audio + imagen registrados", f"{eventos_con_referencias:,}")
    k2.metric("Pares marcados como descargados", f"{eventos_con_medios_locales:,}")
    k3.metric("Pares utilizables por el análisis", f"{len(comunes):,}")
    if len(comunes) < eventos_con_referencias:
        st.caption("Para sumar eventos al pareo, descarga los medios pendientes de aves_db. Solo se cuentan como pareados los archivos descargados, íntegros y localizables; no se crean eventos artificiales.")

    if not comunes:
        st.info("Las observaciones tienen referencias multimedia en aves_db. Usa el botón de descarga de arriba para preparar la cohorte emparejada; los medios quedan en la base y en el almacén local.")
        if imagen and audio:
            st.warning("Hay imágenes y audios disponibles, pero no están asociados al mismo `event_id`; revisa cómo se sincronizaron las relaciones de medios en aves_db.")
    else:
        especies = sorted({
            str(recurso.meta.get("especie"))
            for clase in ("audio", "img")
            for recursos in por_evento[clase].values()
            for recurso in recursos
            if recurso.meta.get("especie")
        })
        especie_elegida = st.selectbox("Cohorte taxonómica", ["Todas las especies", *especies])
        event_ids = comunes
        if especie_elegida != "Todas las especies":
            event_ids = {
                event_id for tipo_evento in ("audio", "img")
                for event_id, recursos_evento in por_evento[tipo_evento].items()
                if any(r.meta.get("especie") == especie_elegida for r in recursos_evento)
            } & comunes
        import numpy as np

        event_ids = sorted(event_ids)
        if len(event_ids) > max_muestras:
            idx = np.sort(np.random.default_rng(int(semilla)).choice(len(event_ids), max_muestras, replace=False))
            event_ids = [event_ids[i] for i in idx]
        seleccion_audio = [por_evento["audio"][event_id][0] for event_id in event_ids]
        seleccion_img = [por_evento["img"][event_id][0] for event_id in event_ids]
        algoritmos_audio = registry.algoritmos("audio")
        algoritmos_img = registry.algoritmos("img")
        candidatos = [
            nombre for nombre in algoritmos_audio
            if nombre in algoritmos_img and nombre in {
                "Agrupación · KMeans", "KMeans (k óptimo)", "Clusterización jerárquica",
                "DBSCAN · HDBSCAN", "Modelo mixto gaussiano", "Agrupación por escaneo (OPTICS)",
            }
        ]
        st.write(f"**Cohorte emparejada:** {len(event_ids):,} eventos · una grabación y una imagen por evento")
        st.caption("Ambas modalidades usan los mismos identificadores de evento y el mismo submuestreo reproducible. Cada vista se agrupa sin usar especie como entrada.")
        if not candidatos:
            st.info("No hay algoritmos de agrupamiento comunes disponibles para audio e imagen.")
        else:
            nombre_algoritmo = st.selectbox("Algoritmo común", candidatos, index=0)
            entrada = algoritmos_audio[nombre_algoritmo]
            parametros = vistas.parametros(entrada, "audio", "lab_pares")

            tarea_key = "lab_comparacion_modalidades"
            if len(event_ids) < entrada.min_muestras:
                st.info(f"Se necesitan al menos {entrada.min_muestras} eventos comunes para este algoritmo.")
            if st.button("Comparar las dos modalidades", type="primary", key="lab_run_pares", disabled=len(event_ids) < entrada.min_muestras):
                def ejecutar_pares(ctx):
                    resultados = {}
                    granularidades = (m.cfg.audio_granularidad, m.cfg.img_granularidad)
                    m.cfg.audio_granularidad = m.cfg.img_granularidad = "archivo"
                    try:
                        for i, (tipo_recurso, recursos) in enumerate((("audio", seleccion_audio), ("img", seleccion_img))):
                            ctx.punto(i / 2, f"Analizando {loaders.NOMBRES[tipo_recurso].lower()}")
                            resultados[tipo_recurso] = m.ejecutar(
                                tipo_recurso, recursos, nombre_algoritmo, dict(parametros), prep,
                                max_muestras=len(event_ids), ctx=ctx,
                            )
                    finally:
                        m.cfg.audio_granularidad, m.cfg.img_granularidad = granularidades
                    from ml.paired_learning import comparar

                    return {"salidas": resultados, "pareo": comparar(resultados, semilla)}

                st.session_state[tarea_key] = m.tareas.lanzar("Comparación audio–imagen", ejecutar_pares)
                st.rerun()

            tarea = vistas.seguimiento(m, tarea_key)
            if tarea is not None:
                st.session_state["lab_resultados_modalidades"] = tarea.resultado
            resultados = st.session_state.get("lab_resultados_modalidades")
            if resultados:
                analisis = resultados["pareo"]
                resultados = resultados["salidas"]
                st.subheader("Lectura multimodal del pareo")
                if analisis["acuerdo_ari"] is not None:
                    st.caption(f"Filas emparejadas tras cargar los archivos: {analisis['n_comunes']:,} eventos. ARI entre grupos acústicos y visuales: {analisis['acuerdo_ari']:.3f} (1 = asignaciones iguales; cerca de 0 = acuerdo similar al azar). Etiquetas de especie concordantes: {analisis['cohorte_taxonomica']:,}/{analisis['n_comunes']:,}.")
                else:
                    st.caption("No hubo suficientes eventos emparejados para medir acuerdo entre agrupamientos.")
                st.dataframe(
                    [{"Modalidad": loaders.NOMBRES[tipo_recurso], **salida.res.kpis}
                     for tipo_recurso, salida in resultados.items()],
                    hide_index=True,
                    use_container_width=True,
                )
                st.markdown("**Predicción supervisada de especie · validación cruzada por evento**")
                supervisado = analisis["supervisado"]
                if supervisado:
                    tabla = [{"Modalidad": loaders.NOMBRES[tipo], "Eventos": v["eventos"], "Especies": v["especies"],
                              "Pliegues": v["pliegues"], "Exactitud balanceada": v["balanced_accuracy"], "F1 macro": v["f1_macro"],
                              "Referencia azar": analisis["referencia_azar"][tipo]}
                             for tipo, v in supervisado.items()]
                    st.dataframe(tabla, hide_index=True, use_container_width=True)
                    st.caption("ExtraTrees predice especie con descriptores de una sola modalidad; el F1 macro y la exactitud balanceada muestran el rendimiento por clase. Se excluyen especies con menos de dos eventos pareados. La referencia azar es 1/número de especies y no sustituye una línea base entrenada.")
                else:
                    st.info("La predicción supervisada requiere al menos dos especies con dos o más eventos pareados cada una.")
                st.caption("La agrupación no supervisada y el ARI audio–imagen usan únicamente descriptores. Las etiquetas de especie se reservan para la evaluación supervisada y nunca entrenan el agrupamiento.")
                izquierda, derecha = st.columns(2)
                for columna, tipo_recurso in ((izquierda, "audio"), (derecha, "img")):
                    with columna, st.expander(loaders.NOMBRES[tipo_recurso], expanded=False):
                        vistas.mostrar_algoritmo(resultados[tipo_recurso])

with tab_modalidad:
    st.subheader("¿Qué estructura aparece en cada tipo de evidencia?")
    etiquetas = {loaders.NOMBRES[t]: t for t in ("audio", "img", "texto", "geo")}
    nombre_tipo = st.selectbox("Modalidad", list(etiquetas))
    tipo = etiquetas[nombre_tipo]
    disponibles = m.recursos(tipo)
    if not disponibles:
        st.info(f"No hay recursos disponibles para {nombre_tipo.lower()} en aves_db.")
    else:
        por_nombre = {r.nombre: r for r in disponibles}
        elegidos = st.multiselect(
            "Recursos del dataset",
            list(por_nombre),
            default=list(por_nombre),
            key=f"lab_recursos_{tipo}",
        )
        seleccionados = [por_nombre[n] for n in elegidos]
        if seleccionados:
            objetivo = st.radio(
                "Objetivo",
                ["Control de calidad", "Agrupar y validar taxonomía", "Comparar algoritmos"],
                horizontal=True,
                key=f"lab_objetivo_{tipo}",
            )
            if objetivo == "Control de calidad":
                vistas.calidad(m, tipo, seleccionados, max_muestras)
            elif objetivo == "Agrupar y validar taxonomía":
                vistas.ejecutar_panel(m, tipo, seleccionados, prep, max_muestras, "algoritmo")
            else:
                vistas.multi(m, tipo, seleccionados, prep, max_muestras)

with tab_historial:
    vistas.pagina_experimentos(m)
