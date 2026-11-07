"""Importa la fuente PostgreSQL `aves_db` al catálogo de trabajo de la aplicación.

PostgreSQL es la única fuente de registros. SQLite se conserva para el estado de la
aplicación y las ejecuciones; sus tablas de datos se reconstruyen desde PostgreSQL.
"""
from __future__ import annotations

import hashlib


def _insertar(con, tabla, columnas, valores, ignorar=False):
    if len(columnas) != len(valores):
        raise ValueError(f"{tabla}: {len(columnas)} columnas y {len(valores)} valores")
    modo = "INSERT OR IGNORE" if ignorar else "INSERT"
    nombres = ",".join(columnas)
    marcadores = ",".join("?" for _ in valores)
    con.execute(f"{modo} INTO {tabla}({nombres}) VALUES({marcadores})", valores)


def sincronizar(cfg, almacen, con) -> int:
    if not cfg.database_url:
        raise RuntimeError("Falta DATABASE_URL; configura la conexión a la base PostgreSQL aves_db.")
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError("Falta psycopg. Instala las dependencias del proyecto.") from exc

    with psycopg.connect(cfg.database_url, row_factory=dict_row) as pg:
        pg.execute("SET TRANSACTION READ ONLY")
        tablas = {r["table_name"] for r in pg.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
        ).fetchall()}
        requeridas = {"observations", "taxa", "media"}
        faltan = requeridas - tablas
        if faltan:
            raise RuntimeError("El esquema PostgreSQL no contiene las tablas requeridas: " + ", ".join(sorted(faltan)))
        observaciones = pg.execute("""
            SELECT o.obs_id, o.source, o.source_id, o.dataset_key, o.taxon_key, o.event_date, o.lat, o.lon,
                   o.coord_uncertainty_m AS coordinate_uncertainty_m, o.country_code,
                   t.scientific_name, t.genus, t.family
            FROM observations o LEFT JOIN taxa t ON t.taxon_key = o.taxon_key
            ORDER BY o.obs_id
        """).fetchall()
        medios = pg.execute("""
            SELECT m.media_id, m.obs_id, m.type, m.license, m.url_original AS url,
                   m.file_path, m.sha256, m.bytes AS size_bytes, m.width, m.height, m.duration_s,
                   m.sample_rate
            FROM media m ORDER BY m.obs_id, m.media_id
        """).fetchall()
        hay_contenido = "media_content" in tablas and pg.execute(
            "SELECT EXISTS(SELECT 1 FROM media_content) AS hay_contenido"
        ).fetchone()["hay_contenido"]
        if hay_contenido:
            from data.storage import EXT, MIME_EXT

            for medio in medios:
                formato = str(medio.get("type") or "").lower()
                tipo = "audio" if formato in {"audio", "sound", "recording"} else "img" if formato in {"image", "stillimage", "photo", "photograph"} else None
                if tipo is None or almacen.resolver(medio.get("file_path"), tipo):
                    continue
                fila = pg.execute("SELECT content, mime_type FROM media_content WHERE media_id=%s", (medio["media_id"],)).fetchone()
                ext = MIME_EXT.get(fila["mime_type"].lower()) if fila else None
                if not fila or ext not in EXT[tipo]:
                    continue
                contenido = bytes(fila["content"])
                sha = hashlib.sha256(contenido).hexdigest()
                if medio.get("sha256") and sha != medio["sha256"].strip():
                    continue
                ruta, sha, _ = almacen.ingerir_bytes(contenido, tipo, f"pg-{medio['media_id']}{ext}")
                medio["file_path"], medio["sha256"], medio["size_bytes"] = almacen.relativa(ruta), sha, len(contenido)
        textos_taxon = []
        if "texts" in tablas:
            textos_taxon = pg.execute("""SELECT tx.*, t.scientific_name
                FROM texts tx LEFT JOIN taxa t USING(taxon_key)""").fetchall()
        nombres_taxon = []
        if "vernacular_names" in tablas:
            nombres_taxon = pg.execute("SELECT * FROM vernacular_names").fetchall()

    nombres_por_taxon = {}
    for nombre in nombres_taxon:
        valor = nombre.get("vernacular_name") or nombre.get("name")
        if valor:
            nombres_por_taxon.setdefault(nombre.get("taxon_key"), []).append(str(valor))

    # La copia local sirve como caché/índice operativo; nunca se mezcla con cargas antiguas.
    con.execute("DELETE FROM events")  # cascada elimina geo, texts, audio e images
    con.execute("DELETE FROM documents")
    con.execute("DELETE FROM species_texts")
    for o in observaciones:
        eid = str(o["obs_id"])
        especie = o.get("scientific_name")
        _insertar(con, "events",
                  ("event_id", "source", "source_id", "origin", "observed_raw", "species", "genus", "family", "class", "dataset_key", "license", "status", "created_utc"),
                  (eid, o.get("source") or "aves_db", str(o.get("source_id") or eid), "real",
                   o.get("event_date"), especie, o.get("genus"), o.get("family"), None,
                   o.get("dataset_key"), None, "ok", "postgres"))
        if o.get("lat") is not None and o.get("lon") is not None:
            _insertar(con, "geo", ("event_id", "lat", "lon", "crs", "uncertainty_m", "country", "locality", "geo_status"),
                      (eid, o["lat"], o["lon"], "EPSG:4326", o.get("coordinate_uncertainty_m"),
                       o.get("country_code"), o.get("locality"), "present"))
        for j, valor in enumerate(nombres_por_taxon.get(o.get("taxon_key"), [])):
            _insertar(con, "texts", ("event_id", "field", "original"), (eid, f"nombre_comun_{j}", valor), ignorar=True)

    for m in medios:
        tipo = str(m.get("type") or "").strip().lower()
        tipo = "audio" if tipo in {"audio", "sound", "recording"} else "img" if tipo in {"image", "stillimage", "photo", "photograph"} else None
        if tipo is None:
            continue
        eid, mid = str(m["obs_id"]), f"pg:{m['media_id']}"
        local = None
        digest = m.get("sha256")
        estado = "pendiente"
        path = m.get("file_path")
        ruta_fuente = almacen.resolver(path, tipo) if path else None
        if ruta_fuente:
            try:
                ruta, digest, _ = almacen.ingerir(ruta_fuente, tipo)
                local = almacen.relativa(ruta)
                estado = "correcto"
            except Exception:  # archivo no disponible o formato fuera de las modalidades admitidas
                pass
        if tipo == "audio":
            _insertar(con, "audio",
                      ("audio_id", "event_id", "url", "license", "local_path", "sha256", "duration_s", "sample_rate", "channels", "download_status", "size_bytes"),
                      (mid, eid, m.get("url"), m.get("license"), local, digest, m.get("duration_s"),
                       m.get("sample_rate"), m.get("channels"), estado, m.get("size_bytes")))
        else:
            _insertar(con, "images",
                      ("image_id", "event_id", "url", "license", "local_path", "sha256", "width", "height", "download_status", "size_bytes"),
                      (mid, eid, m.get("url"), m.get("license"), local, digest, m.get("width"),
                       m.get("height"), estado, m.get("size_bytes")))

    for t in textos_taxon:
        species = t.get("scientific_name")
        texto = t.get("content") or t.get("text")
        if not species or not texto:
            continue
        _insertar(con, "species_texts", ("species", "source", "lang", "text", "url", "license"),
                  (species, str(t.get("source") or "aves_db"), str(t.get("language") or t.get("lang") or ""),
                   str(texto), t.get("url"), t.get("license")), ignorar=True)
    con.commit()
    return len(observaciones)
