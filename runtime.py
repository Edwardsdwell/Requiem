"""Recursos compartidos por la página principal y las páginas de Streamlit."""
import streamlit as st

from config import CFG


@st.cache_resource(show_spinner=False)
def motor():
    from core.engine import Motor

    if not CFG.database_url:
        raise RuntimeError("Configura DATABASE_URL en .env para iniciar el análisis de aves_db.")
    return Motor(CFG)
