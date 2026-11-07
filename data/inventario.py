"""Inventario de registros e informe de calidad de datos por modalidad.

* `inventario`: una fila por registro (audio, imagen, texto, punto geográfico o documento local), clasificada en
  **válido / inválido / pendiente** con su motivo y su identificador de origen. Nada queda fuera: si un registro
  no se usa en el análisis, aquí se ve por qué.
* `informe_calidad`: resumen por modalidad (conteos, motivos, formatos, duraciones, resoluciones, longitudes de
  texto, cobertura geográfica, completitud por evento) y recomendaciones. Exportable a Markdown y CSV.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

from . import db
from .validacion import validar_archivo, validar_coordenada

MODALIDADES = {"audio": "Audio", "img": "Imágenes", "texto": "Texto", "geo": "Coordenadas"}
COLS = ["modalidad", "registro", "event_id", "id_origen", "especie", "clase", "estado", "estado_bd", "motivo", "archivo", "sha256", "fuente"]


def _fila(**kw) -> dict:
    return {c: kw.get(c) for c in COLS}


def inventario(con, almacen) -> pd.DataFrame:
    filas: list[dict] = []
    # ---------------- medios enlazados desde aves_db
    for tipo, tabla, idc in (("audio", "audio", "audio_id"), ("img", "images", "image_id")):
        vistos: dict[str, str] = {}
        q = (f"SELECT t.{idc} AS id, t.event_id, t.local_path, t.sha256, t.download_status AS st, t.error, t.url, "
             f"e.source, e.source_id, e.species, e.class FROM {tabla} t LEFT JOIN events e USING(event_id) ORDER BY t.{idc}")
        for f in con.execute(q):
            st = f["st"] or db.PENDIENTE
            cat, motivo = db.CATEGORIA.get(st, "pendiente"), f["error"] or ""
            if st == db.CORRECTO:
                if almacen.resolver(f["local_path"], tipo) is None:
                    cat, motivo = "pendiente", "marcado como correcto pero el archivo no está en el almacén"
                elif f["sha256"] in vistos:
                    motivo = f"contenido idéntico a {vistos[f['sha256']]} (se analiza una sola vez)"
                else:
                    vistos[f["sha256"]] = f["id"]
            elif st == db.PENDIENTE and not motivo:
                motivo = "aún no descargado" if f["url"] else "sin URL"
            filas.append(_fila(modalidad=tipo, registro=f["id"], event_id=f["event_id"], id_origen=f["source_id"], especie=f["species"],
                               clase=f["class"], estado=cat, estado_bd=st, motivo=motivo, archivo=f["local_path"], sha256=f["sha256"], fuente=f["source"] or "aves_db"))
    # ---------------- puntos geográficos de aves_db
    for f in con.execute("SELECT g.event_id, g.lat, g.lon, g.geo_status, g.motivo, e.source, e.source_id, e.species, e.class "
                         "FROM geo g LEFT JOIN events e USING(event_id) ORDER BY g.event_id"):
        _, _, motivo = validar_coordenada(f["lat"], f["lon"])
        filas.append(_fila(modalidad="geo", registro=f"{f['event_id']}:geo", event_id=f["event_id"], id_origen=f["source_id"],
                           especie=f["species"], clase=f["class"], estado="inválido" if motivo else "válido",
                           estado_bd=f["geo_status"], motivo=motivo or f["motivo"] or "", fuente=f["source"] or "aves_db"))
    # ---------------- textos de la BD
    for f in con.execute("SELECT t.event_id, t.field, t.original, e.source, e.source_id, e.species, e.class FROM texts t "
                         "LEFT JOIN events e USING(event_id) ORDER BY t.event_id, t.field"):
        txt = (f["original"] or "").strip()
        motivo = "texto vacío" if not txt else ("texto demasiado corto (<3 caracteres)" if len(txt) < 3 else "")
        filas.append(_fila(modalidad="texto", registro=f"{f['event_id']}:{f['field']}", event_id=f["event_id"], id_origen=f["source_id"],
                           especie=f["species"], clase=f["class"], estado="inválido" if motivo else "válido", estado_bd=f["field"],
                           motivo=motivo, fuente=f["source"] or "aves_db"))
    # ---------------- textos de especie sincronizados de aves_db
    if con.execute("SELECT 1 FROM sqlite_master WHERE name='species_texts'").fetchone():
        for f in con.execute("SELECT species, source, lang, text, url FROM species_texts"):
            ok = bool((f["text"] or "").strip())
            filas.append(_fila(modalidad="texto", registro=f"{f['species']}:{f['source']}:{f['lang']}", especie=f["species"],
                               estado="válido" if ok else "inválido", estado_bd=f["source"], motivo="" if ok else "texto vacío",
                               archivo=f["url"], fuente=f["source"]))
    # ---------------- documentos locales (subidos o de una carpeta)
    for f in con.execute("SELECT doc_id, kind, name, local_path, sha256, source FROM documents ORDER BY kind, name"):
        ok = almacen.resolver(f["local_path"], f["kind"]) is not None
        filas.append(_fila(modalidad=f["kind"], registro=f["doc_id"], id_origen=f["name"], estado="válido" if ok else "pendiente",
                           estado_bd="local", motivo="" if ok else "archivo no encontrado en el almacén", archivo=f["local_path"],
                           sha256=f["sha256"], fuente=f["source"]))
    return pd.DataFrame(filas, columns=COLS)


# ------------------------------------------------------------------ informe de calidad
def _resumen(inv: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m, nombre in MODALIDADES.items():
        s = inv[inv.modalidad == m]
        c = s.estado.value_counts()
        dup = int(s.motivo.fillna("").str.startswith("contenido idéntico").sum())
        tot = len(s)
        out.append(dict(modalidad=nombre, registros=tot, válidos=int(c.get("válido", 0)), inválidos=int(c.get("inválido", 0)),
                        pendientes=int(c.get("pendiente", 0)), **{"% válidos": round(100 * c.get("válido", 0) / tot, 1) if tot else 0.0},
                        **{"contenido repetido": dup}))
    return pd.DataFrame(out)


def _stats(v) -> dict:
    v = pd.to_numeric(pd.Series(v), errors="coerce").dropna()
    if v.empty:
        return dict(n=0)
    return dict(n=int(len(v)), mín=round(float(v.min()), 2), mediana=round(float(v.median()), 2), media=round(float(v.mean()), 2), máx=round(float(v.max()), 2))


def _idioma(t: str) -> str:
    t = f" {t.lower()} "
    es = sum(t.count(f" {w} ") for w in ("de", "la", "el", "y", "en", "los", "las", "con", "por", "del"))
    en = sum(t.count(f" {w} ") for w in ("the", "of", "and", "in", "on", "with", "at", "to", "from", "near"))
    return "es" if es > en else ("en" if en > es else "indeterminado")


def informe_calidad(con, almacen, cfg, ctx=None) -> dict:
    inv = inventario(con, almacen)
    rep: dict = dict(resumen=_resumen(inv), inventario=inv, recomendaciones=[])
    mot = inv[(inv.estado != "válido") & inv.motivo.fillna("").ne("")]
    rep["motivos"] = (mot.groupby(["modalidad", "estado", "motivo"]).size().reset_index(name="registros")
                      .sort_values("registros", ascending=False) if len(mot) else pd.DataFrame(columns=["modalidad", "estado", "motivo", "registros"]))

    # ---------------- audio
    a = pd.DataFrame([dict(f) for f in con.execute(
        "SELECT audio_id, local_path, duration_s, sample_rate, channels, size_bytes FROM audio WHERE download_status=?", (db.CORRECTO,))])
    docs_a = [dict(f) for f in con.execute("SELECT name, local_path FROM documents WHERE kind='audio'")]
    for i, d in enumerate(docs_a):
        if ctx is not None:
            ctx.punto(i / max(len(docs_a), 1) * 0.3, "Midiendo audios locales")
        r = almacen.resolver(d["local_path"], "audio")
        if r:
            v = validar_archivo(r, "audio")
            a = pd.concat([a, pd.DataFrame([dict(audio_id=d["name"], local_path=d["local_path"], duration_s=v.info.get("duracion_s"),
                                                 sample_rate=v.info.get("sr"), channels=v.info.get("canales"), size_bytes=v.info.get("bytes"))])])
    if len(a):
        a["formato"] = a.local_path.fillna("").str.extract(r"\.([A-Za-z0-9]+)$")[0].str.lower()
        rep["audio"] = dict(formatos=a.formato.value_counts().to_dict(), duracion_s=_stats(a.duration_s), frecuencia_hz=a.sample_rate.value_counts().to_dict(),
                            canales=a.channels.value_counts().to_dict(), tamano_mb=_stats(a.size_bytes / 2**20),
                            sin_medir=int(a.duration_s.isna().sum()), tabla=a)
        n_m4a = int((a.formato == "m4a").sum())
        if n_m4a and not shutil.which("ffmpeg"):
            rep["recomendaciones"].append(f"{n_m4a} audios están en M4A: instale **ffmpeg** (y agréguelo al PATH) para poder leerlos.")
        cortos = int((pd.to_numeric(a.duration_s, errors="coerce") < cfg.audio_ventana_s).sum())
        if cortos:
            rep["recomendaciones"].append(f"{cortos} audios duran menos que la ventana de análisis ({cfg.audio_ventana_s} s): se rellenan con silencio.")
        largos = int((pd.to_numeric(a.duration_s, errors="coerce") > cfg.audio_max_s).sum())
        if largos:
            rep["recomendaciones"].append(f"{largos} audios superan {cfg.audio_max_s} s: solo se analizan los primeros {cfg.audio_max_s} s.")

    # ---------------- imágenes
    im = pd.DataFrame([dict(f) for f in con.execute(
        "SELECT image_id, local_path, width, height, size_bytes FROM images WHERE download_status=?", (db.CORRECTO,))])
    for d in con.execute("SELECT name, local_path FROM documents WHERE kind='img'").fetchall():
        r = almacen.resolver(d["local_path"], "img")
        if r:
            v = validar_archivo(r, "img")
            im = pd.concat([im, pd.DataFrame([dict(image_id=d["name"], local_path=d["local_path"], width=v.info.get("ancho"),
                                                   height=v.info.get("alto"), size_bytes=v.info.get("bytes"))])])
    if len(im):
        im["formato"] = im.local_path.fillna("").str.extract(r"\.([A-Za-z0-9]+)$")[0].str.lower()
        mp = pd.to_numeric(im.width, errors="coerce") * pd.to_numeric(im.height, errors="coerce") / 1e6
        rep["img"] = dict(formatos=im.formato.value_counts().to_dict(), ancho_px=_stats(im.width), alto_px=_stats(im.height), megapixeles=_stats(mp),
                          tamano_mb=_stats(im.size_bytes / 2**20), sin_medir=int(im.width.isna().sum()), tabla=im)
        peq = int((pd.to_numeric(im[["width", "height"]].min(axis=1), errors="coerce") < cfg.img_lado).sum())
        if peq:
            rep["recomendaciones"].append(f"{peq} imágenes miden menos de {cfg.img_lado} px de lado: se amplían para extraer parches.")

    # ---------------- texto
    tx = pd.DataFrame([dict(f) for f in con.execute("SELECT event_id, field, original FROM texts WHERE TRIM(COALESCE(original,''))<>''")])
    if con.execute("SELECT 1 FROM sqlite_master WHERE name='species_texts'").fetchone():
        st_ = pd.DataFrame([dict(event_id=f["species"], field=f"{f['source']}_{f['lang']}", original=f["text"])
                            for f in con.execute("SELECT * FROM species_texts WHERE TRIM(COALESCE(text,''))<>''")])
        tx = pd.concat([tx, st_]) if len(st_) else tx
    if len(tx):
        lon = tx.original.str.len()
        pal = tx.original.str.split().str.len()
        campo = tx.field.str.replace(r"_\d+$", "", regex=True)
        vocab = len({w for t in tx.original for w in re.findall(r"[a-záéíóúñü]{3,}", t.lower())})
        rep["texto"] = dict(textos=int(len(tx)), caracteres=_stats(lon), palabras=_stats(pal), vocabulario=vocab, campos=campo.value_counts().to_dict(),
                            idiomas=Counter(_idioma(t) for t in tx.original).most_common(), repetidos=int(tx.original.duplicated().sum()), tabla=tx)
        if pal.median() < 8:
            rep["recomendaciones"].append(f"Los textos son muy cortos (mediana {int(pal.median())} palabras, vocabulario de {vocab} términos): "
                                          "revise si `aves_db` contiene más descripciones de especie.")
    n_docs_txt = con.execute("SELECT COUNT(*) FROM documents WHERE kind='texto'").fetchone()[0]
    if n_docs_txt:
        rep.setdefault("texto", {})["documentos_locales"] = int(n_docs_txt)

    # ---------------- coordenadas
    g = pd.DataFrame([dict(f) for f in con.execute("SELECT g.event_id, g.lat, g.lon, g.uncertainty_m, g.country FROM geo g")])
    if len(g):
        val = g.apply(lambda r: validar_coordenada(r.lat, r.lon)[2] == "", axis=1)
        gv = g[val]
        ext_lat = float(np.ptp(pd.to_numeric(gv.lat))) if len(gv) else 0.0
        ext_lon = float(np.ptp(pd.to_numeric(gv.lon))) if len(gv) else 0.0
        modo = cfg.geo_proyeccion if cfg.geo_proyeccion in ("local", "esfera") else ("local" if max(ext_lat, ext_lon) <= cfg.geo_extension_local_deg else "esfera")
        rep["geo"] = dict(puntos=int(len(g)), validos=int(val.sum()), paises=gv.country.value_counts().to_dict(), extension_lat=round(ext_lat, 2),
                          extension_lon=round(ext_lon, 2), incertidumbre_m=_stats(gv.uncertainty_m), sin_incertidumbre=int(gv.uncertainty_m.isna().sum()),
                          coordenadas_repetidas=int(gv.duplicated(["lat", "lon"]).sum()), proyeccion=modo, tabla=g)
        if len(gv) < 30:
            rep["recomendaciones"].append(f"Solo hay {len(gv)} puntos geográficos válidos: los algoritmos por densidad (DBSCAN, OPTICS) tendrán pocos vecinos; "
                                          "revise si el dataset contiene más observaciones con coordenadas.")

    # ---------------- completitud por evento
    ev = pd.DataFrame([dict(f) for f in con.execute("""
        SELECT e.event_id, e.source_id, e.species, e.class, e.license,
          EXISTS(SELECT 1 FROM audio a WHERE a.event_id=e.event_id AND a.download_status='correcto') AS audio,
          EXISTS(SELECT 1 FROM images i WHERE i.event_id=e.event_id AND i.download_status='correcto') AS imagen,
          EXISTS(SELECT 1 FROM texts t WHERE t.event_id=e.event_id AND TRIM(COALESCE(t.original,''))<>'') AS texto,
          EXISTS(SELECT 1 FROM geo g WHERE g.event_id=e.event_id AND COALESCE(g.geo_status,'')<>'invalido') AS geo
        FROM events e""")])
    if len(ev):
        ev["modalidades"] = ev[["audio", "imagen", "texto", "geo"]].sum(1)
        lic = ev.license.fillna("(sin licencia)").str.replace(r"https?://creativecommons.org/", "", regex=True)
        rep["eventos"] = dict(eventos=int(len(ev)), completos=int((ev.modalidades == 4).sum()), especies=int(ev.species.nunique()),
                              por_clase=ev["class"].fillna("(sin clase)").value_counts().to_dict(),
                              cobertura={m: int(ev[m].sum()) for m in ("audio", "imagen", "texto", "geo")},
                              especies_un_solo_evento=int((ev.species.value_counts() == 1).sum()), licencias=lic.value_counts().to_dict(), tabla=ev)
        sin_texto = int((ev.texto == 0).sum())
        if sin_texto:
            rep["recomendaciones"].append(f"{sin_texto} de {len(ev)} eventos no tienen ningún texto asociado.")
    pend = int((inv.estado == "pendiente").sum())
    if pend:
        rep["recomendaciones"].append(f"{pend} registros están pendientes: use **Descargar pendientes** (requiere internet) o vuelva a importar.")
    inval = int((inv.estado == "inválido").sum())
    if inval:
        rep["recomendaciones"].append(f"{inval} registros son inválidos y no se analizan; revise la tabla de motivos.")
    if not rep["recomendaciones"]:
        rep["recomendaciones"].append("Sin problemas de calidad relevantes.")
    db.auditar(con, "informe_calidad", rep["resumen"].to_dict("records"), int(len(inv)))
    con.commit()
    return rep


def _sin_tablas(d):
    if isinstance(d, dict):
        return {k: _sin_tablas(v) for k, v in d.items() if not isinstance(v, pd.DataFrame)}
    return d


def tabla_md(df: pd.DataFrame) -> str:
    """Tabla Markdown sin dependencias externas."""
    cab = [str(c) for c in df.columns]
    filas = [[("" if pd.isna(v) else str(v)).replace("|", "/") for v in r] for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cab) + " |", "|" + "|".join("---" for _ in cab) + "|", *["| " + " | ".join(f) + " |" for f in filas]])


def informe_markdown(rep: dict) -> str:
    """Informe legible, listo para adjuntar a la entrega."""
    L = ["# Informe de calidad de datos", "", "## Resumen por modalidad", "", tabla_md(rep["resumen"]), ""]
    if len(rep["motivos"]):
        L += ["## Registros no válidos y motivos", "", tabla_md(rep["motivos"]), ""]
    nombres = dict(audio="Audio", img="Imágenes", texto="Texto", geo="Coordenadas", eventos="Eventos (completitud)")
    for k, titulo in nombres.items():
        if k in rep:
            L += [f"## {titulo}", ""]
            for c, v in _sin_tablas(rep[k]).items():
                L.append(f"- **{c}**: {json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else v}")
            L.append("")
    L += ["## Recomendaciones", ""] + [f"- {r}" for r in rep["recomendaciones"]]
    return "\n".join(L) + "\n"


def informe_zip(rep: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("informe_calidad.md", informe_markdown(rep))
        z.writestr("resumen.csv", rep["resumen"].to_csv(index=False))
        z.writestr("motivos.csv", rep["motivos"].to_csv(index=False))
        z.writestr("inventario.csv", rep["inventario"].to_csv(index=False))
        for k in ("audio", "img", "texto", "geo", "eventos"):
            t = rep.get(k, {}).get("tabla") if isinstance(rep.get(k), dict) else None
            if isinstance(t, pd.DataFrame):
                z.writestr(f"{k}.csv", t.to_csv(index=False))
    return buf.getvalue()
