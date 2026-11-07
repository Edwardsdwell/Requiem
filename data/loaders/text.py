"""Texto (TXT, MD, PDF y textos de la BD) -> fragmentos vectorizados con TF-IDF **disperso**."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from core.cache import hash_obj
from core.errors import ErrorDatos
from core.results import Datos
from . import Recurso, cache_caracteristicas, ensamblar

TIPO, NOMBRE, VERSION = "texto", "Texto", "texto-2"

_STOP = """a al algo ante antes como con contra cual cuando de del desde donde durante e el ella ellas ello ellos
en entre era eran es esa esas ese eso esos esta estaba estan estas este esto estos fue fueron ha han hasta hay
la las le les lo los mas me mi muy no nos o otra otro para pero por porque que quien se sea ser si sin sobre su
sus tambien te tiene tienen todo tras tu un una uno unos y ya también más según está están son sido cada pueden
puede the of and to in is are for with on as by that this it be an from or at which can has have not""".split()


def leer_texto(ruta: Path) -> str:
    if ruta.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # carga diferida 

        return "\n\n".join(p.extract_text() or "" for p in PdfReader(str(ruta)).pages)
    return ruta.read_text(encoding="utf-8", errors="ignore")


def trozos(txt: str, min_len: int = 60, oraciones: int = 4) -> list:
    """Párrafos; si hay pocos (p. ej. PDF), grupos de oraciones."""
    pars = [p for p in (re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", txt)) if len(p) >= min_len]
    if len(pars) < 15:
        s = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", txt))
        pars = [p for p in (" ".join(s[i:i + oraciones]) for i in range(0, len(s), oraciones)) if len(p) >= min_len]
    return pars



def fragmentos_recurso(r: Recurso, cfg, con=None):
    """[(texto, etiqueta_origen, meta)] de un recurso, con caché por contenido."""
    cache = cache_caracteristicas(cfg, TIPO)
    clave = hash_obj(VERSION, r.sha256, cfg.txt_min_len, cfg.txt_oraciones)
    z = cache.leer_pkl(clave)
    if z is not None:
        return z
    if r.virtual and r.meta.get("fuente") == "especies":
        from data import db

        c = con or db.connect(cfg)
        z = []
        for f in c.execute("SELECT s.species, s.source, s.lang, s.text, (SELECT class FROM events e WHERE e.species=s.species LIMIT 1) AS clase "
                           "FROM species_texts s WHERE TRIM(COALESCE(s.text,''))<>'' ORDER BY 1, 2, 3"):
            partes = trozos(f["text"], cfg.txt_min_len, cfg.txt_oraciones) or [f["text"]]
            z += [(t, f["species"], dict(especie=f["species"], clase=f["clase"], campo=f["source"], idioma=f["lang"])) for t in partes]
        cache.guardar_pkl(clave, z)
        return z
    if r.virtual:
        from data import db

        c = con or db.connect(cfg)
        filas = c.execute("SELECT t.event_id, t.field, t.original, e.source_id, e.species, e.class FROM texts t LEFT JOIN events e USING(event_id) "
                          "WHERE TRIM(COALESCE(t.original,''))<>'' ORDER BY t.event_id, t.field").fetchall()
        z = [(str(f["original"]).strip(), re.sub(r"_\d+$", "", f["field"]),
              dict(event_id=f["event_id"], id_origen=f["source_id"], especie=f["species"], clase=f["class"], campo=f["field"])) for f in filas]
    else:
        z = [(t, r.nombre, {}) for t in trozos(leer_texto(r.ruta), cfg.txt_min_len, cfg.txt_oraciones)]
    cache.guardar_pkl(clave, z)
    return z


def cargar(recursos: list[Recurso], cfg, ctx=None) -> Datos:
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs, org, meta, nombres_rec = [], [], [], []
    for i, r in enumerate(recursos):
        if ctx is not None:
            ctx.punto(i / len(recursos), f"Texto {i + 1}/{len(recursos)}: {r.nombre}")
        for t, o, m in fragmentos_recurso(r, cfg):
            docs.append(t)
            org.append(o)
            meta.append(dict(r.meta, archivo=r.nombre, sha256=r.sha256, **m))
    if len(docs) < 5:
        raise ErrorDatos("Muy poco texto extraído", "¿PDF escaneado sin texto? Se necesitan al menos 5 fragmentos.")
    v = TfidfVectorizer(max_features=cfg.txt_max_terminos, stop_words=_STOP, sublinear_tf=True,
                        min_df=2 if len(docs) >= 20 else 1, max_df=0.9, dtype=np.float32)
    X = v.fit_transform(docs).tocsr()                       # se conserva DISPERSA 
    if X.shape[1] < 2:
        raise ErrorDatos("El vocabulario resultante es demasiado pequeño", "Añada más texto.")
    hover = np.array([t[:160] + ("…" if len(t) > 160 else "") for t in docs])
    d = Datos(X, list(v.get_feature_names_out()), np.array(org), hover, False, tipo=TIPO, ref=docs, meta=meta,
              recursos={r.nombre: r.ruta for r in recursos if r.ruta},
              descripcion=f"{len(docs)} fragmentos, {X.shape[1]} términos (matriz dispersa, {X.nnz / np.prod(X.shape):.1%} no nulos)")
    d.huella = hash_obj(TIPO, sorted(r.sha256 for r in recursos), VERSION, cfg.txt_max_terminos, cfg.txt_min_len, cfg.txt_oraciones, cfg.max_muestras, cfg.semilla)
    return d                                                 # TF-IDF no se estandariza: LDA (Dirichlet) necesita X >= 0
