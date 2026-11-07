"""Acceso a los medios almacenados en PostgreSQL."""
from common import DATA_DIR, connect


def contenido(media_id: int) -> bytes:
    """Bytes del medio: `media_content.content`; si aún no se importó, el archivo `media.file_path` en DATA_DIR."""
    with connect() as con:
        r = con.execute("SELECT content FROM media_content WHERE media_id=%s", (media_id,)).fetchone()
        if r:
            return bytes(r[0])
        r = con.execute("SELECT file_path FROM media WHERE media_id=%s AND download_status='ok'", (media_id,)).fetchone()
    if not r or not r[0]:
        raise FileNotFoundError(f"El medio {media_id} no tiene contenido en la base de datos ni en disco.")
    return (DATA_DIR / r[0]).read_bytes()
