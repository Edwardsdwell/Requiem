"""Inicia la aplicación usando el servidor de Streamlit."""
from pathlib import Path
import sys

try:
    import psycopg
except ModuleNotFoundError as exc:
    if exc.name != "psycopg":
        raise
    raise SystemExit(
        "Falta psycopg en este entorno. Instala las dependencias con:\n"
        '  .venv\\Scripts\\python.exe -m pip install -r requirements.txt'
    ) from exc

from streamlit.web.cli import main

sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py")), *sys.argv[1:]]
main()
