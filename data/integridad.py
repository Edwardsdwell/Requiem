"""Verificación de integridad del almacén: que cada registro «válido» tenga su archivo, íntegro y legible.

Se usa al terminar una importación y desde la pestaña **Datos → Verificar integridad**. Un registro que
deja de cumplir pasa a «pendiente» (si el archivo falta: se puede volver a descargar) o a «inválido»
(si está vacío, dañado o no coincide con su SHA-256), siempre con el motivo en la columna `error`.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from . import db
from .storage import sha256_archivo
from .validacion import Veredicto, validar_archivo

MEDIOS = (("audio", "audio", "audio_id"), ("img", "images", "image_id"))


def guardar_veredicto(con, tabla: str, idc: str, id_: str, v: Veredicto, ruta_rel: str | None = None, sha: str | None = None) -> str:
    """Actualiza estado, motivo y metadatos técnicos (duración, resolución…) de un registro de audio o imagen."""
    estado = db.CORRECTO if v.ok else db.INVALIDO
    i = v.info
    extra = (dict(duration_s=i.get("duracion_s"), sample_rate=i.get("sr"), channels=i.get("canales")) if tabla == "audio"
             else dict(width=i.get("ancho"), height=i.get("alto")))
    sets = ", ".join(f"{k}=COALESCE(?, {k})" for k in extra)
    con.execute(f"UPDATE {tabla} SET download_status=?, error=?, local_path=COALESCE(?, local_path), sha256=COALESCE(?, sha256), "
                f"size_bytes=COALESCE(?, size_bytes), {sets}, updated_utc=? WHERE {idc}=?",
                (estado, v.motivo or None, ruta_rel, sha, i.get("bytes"), *extra.values(), db.now(), id_))
    return estado


def verificar(con, almacen, ctx=None, recalcular_hash: bool = True, validar_contenido: bool = True) -> dict:
    """Recorre los registros marcados como correctos y los documentos locales.

    Devuelve un informe con lo revisado, lo que sigue válido y lo que cambió de estado (con motivos).
    """
    rep = dict(revisados=0, validos=0, faltantes=[], danados=[], invalidos=[], documentos_invalidos=[])
    filas = []
    for tipo, tabla, idc in MEDIOS:
        for f in con.execute(f"SELECT {idc} AS id, local_path, sha256 FROM {tabla} WHERE download_status=?", (db.CORRECTO,)):
            filas.append((tipo, tabla, idc, dict(f)))
    cache_sha: dict[Path, str] = {}
    for n, (tipo, tabla, idc, f) in enumerate(filas):
        if ctx is not None:
            ctx.punto(n / max(len(filas), 1), f"Verificando {n + 1}/{len(filas)}")
        rep["revisados"] += 1
        ruta = almacen.resolver(f["local_path"], tipo) or (almacen.buscar_sha(tipo, f["sha256"]) if f["sha256"] else None)
        if ruta is None:
            con.execute(f"UPDATE {tabla} SET download_status=?, error=?, updated_utc=? WHERE {idc}=?",
                        (db.PENDIENTE, "archivo no encontrado en el almacén (se puede volver a descargar)", db.now(), f["id"]))
            rep["faltantes"].append(f["id"])
            continue
        if recalcular_hash and f["sha256"]:
            real = cache_sha.get(ruta) or cache_sha.setdefault(ruta, sha256_archivo(ruta))
            if real != f["sha256"]:
                con.execute(f"UPDATE {tabla} SET download_status=?, error=?, updated_utc=? WHERE {idc}=?",
                            (db.INVALIDO, "el contenido no coincide con su SHA-256 (archivo modificado o dañado)", db.now(), f["id"]))
                rep["danados"].append(f["id"])
                continue
        if validar_contenido:
            v = validar_archivo(ruta, tipo)
            guardar_veredicto(con, tabla, idc, f["id"], v, almacen.relativa(ruta))
            if not v.ok:
                rep["invalidos"].append((f["id"], v.motivo))
                continue
        rep["validos"] += 1
    for d in con.execute("SELECT doc_id, kind, name, local_path, sha256 FROM documents").fetchall():
        ruta = almacen.resolver(d["local_path"], d["kind"])
        if ruta is None:
            rep["documentos_invalidos"].append((d["name"], "archivo no encontrado"))
        elif recalcular_hash and sha256_archivo(ruta) != d["sha256"]:
            rep["documentos_invalidos"].append((d["name"], "el contenido no coincide con su SHA-256"))
    rep["motivos"] = dict(Counter(m for _, m in rep["invalidos"]))
    db.auditar(con, "verificar_integridad", {k: (len(v) if isinstance(v, list) else v) for k, v in rep.items()}, rep["revisados"])
    con.commit()
    return rep
