"""Recursos compartidos por la página principal y las páginas de Streamlit."""
import streamlit as st

from config import CFG


@st.cache_resource(show_spinner=False)
def motor():
    from core.engine import Motor

    if not CFG.database_url:
        raise RuntimeError("Configura DATABASE_URL en Streamlit Secrets o en .env para conectar con aves_db.")
    return Motor(CFG)
