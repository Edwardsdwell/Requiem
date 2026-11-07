"""SQLite unificada: metadatos, procedencia y estados de adquisición.

Esquema relacional con claves foráneas e índices. Conserva las tablas del proyecto alternativo
(events, geo, texts, audio, images, exclusions, ingest_log, snapshots, snapshot_events) y añade
`documents` (textos y archivos geográficos locales), `runs` (trazabilidad experimental)
y `meta` (versión del esquema).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

VERSION_ESQUEMA = 3

# Estados de descarga. «invalido»: el archivo existe pero está vacío, corrupto o no corresponde
# a su tipo; se conserva el registro con el motivo en la columna `error` (no se pierde en silencio).
PENDIENTE, CORRECTO, OMITIDO, FALLIDO, INVALIDO = "pendiente", "correcto", "omitido", "fallido", "invalido"
ESTADOS = (PENDIENTE, CORRECTO, OMITIDO, FALLIDO, INVALIDO)
# Clasificación de alto nivel que ve el usuario (válidos / inválidos / pendientes de procesamiento)
CATEGORIA = {CORRECTO: "válido", INVALIDO: "inválido", OMITIDO: "inválido", FALLIDO: "pendiente", PENDIENTE: "pendiente"}

# Licencias: se descarta cualquier variante «sin derivadas».
BAD = ("by-nd", "by-nc-nd", "by_nd", "by_nc_nd")
GOOD = ("zero", "cc0", "/by/", "/by-nc/", "/by-sa/", "/by-nc-sa/", "cc_by")

ESQUEMA = """
CREATE TABLE IF NOT EXISTS meta(clave TEXT PRIMARY KEY, valor TEXT);
CREATE TABLE IF NOT EXISTS events(
  event_id TEXT PRIMARY KEY, source TEXT, source_id TEXT, origin TEXT DEFAULT 'real',
  observed_raw TEXT, observed_utc TEXT, species TEXT, genus TEXT, family TEXT, class TEXT,
  dataset_key TEXT, license TEXT, link_type TEXT DEFAULT 'confirmed', status TEXT DEFAULT 'ok',
  created_utc TEXT);
CREATE TABLE IF NOT EXISTS geo(
  event_id TEXT PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
  lat REAL, lon REAL, crs TEXT, uncertainty_m REAL, country TEXT, locality TEXT, geo_status TEXT);
CREATE TABLE IF NOT EXISTS texts(
  event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
  field TEXT NOT NULL, original TEXT, normalized TEXT, lang TEXT, PRIMARY KEY(event_id, field));
CREATE TABLE IF NOT EXISTS audio(
  audio_id TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
  url TEXT, license TEXT, creator TEXT, local_path TEXT, sha256 TEXT, wav_path TEXT,
  duration_s REAL, sample_rate INTEGER, channels INTEGER, quality_flags TEXT,
  download_status TEXT DEFAULT 'pendiente', size_bytes INTEGER, mime TEXT, error TEXT,
  attempts INTEGER DEFAULT 0, updated_utc TEXT);
CREATE TABLE IF NOT EXISTS images(
  image_id TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
  url TEXT, license TEXT, creator TEXT, local_path TEXT, sha256 TEXT, width INTEGER, height INTEGER,
  download_status TEXT DEFAULT 'pendiente', size_bytes INTEGER, mime TEXT, error TEXT,
  attempts INTEGER DEFAULT 0, updated_utc TEXT);
CREATE TABLE IF NOT EXISTS documents(
  doc_id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('texto','geo','audio','img')),
  name TEXT, local_path TEXT, sha256 TEXT UNIQUE, size_bytes INTEGER, source TEXT DEFAULT 'local',
  origin_url TEXT, created_utc TEXT);
CREATE TABLE IF NOT EXISTS exclusions(event_id TEXT, reason TEXT, utc TEXT);
CREATE TABLE IF NOT EXISTS ingest_log(utc TEXT, source TEXT, url TEXT, status TEXT, n_records INTEGER, latency_s REAL, note TEXT);
CREATE TABLE IF NOT EXISTS snapshots(snapshot_id TEXT PRIMARY KEY, created_utc TEXT, note TEXT, manifest_sha256 TEXT);
CREATE TABLE IF NOT EXISTS snapshot_events(snapshot_id TEXT REFERENCES snapshots(snapshot_id) ON DELETE CASCADE, event_id TEXT);
CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY, created_utc TEXT, tipo TEXT, clase TEXT, nombre TEXT, params TEXT,
  huella TEXT, n_muestras INTEGER, n_variables INTEGER, semilla INTEGER, duracion_s REAL, memoria_mb REAL,
  estado TEXT, clave_resultado TEXT, mensaje TEXT);
CREATE TABLE IF NOT EXISTS species_texts(
  species TEXT NOT NULL, source TEXT NOT NULL, lang TEXT NOT NULL DEFAULT '', text TEXT, url TEXT, license TEXT, retrieved_utc TEXT,
  PRIMARY KEY(species, source, lang));
CREATE TABLE IF NOT EXISTS auditoria(
  utc TEXT, accion TEXT, detalle TEXT, n INTEGER);
CREATE INDEX IF NOT EXISTS ix_audio_event ON audio(event_id);
CREATE INDEX IF NOT EXISTS ix_audio_status ON audio(download_status);
CREATE INDEX IF NOT EXISTS ix_audio_sha ON audio(sha256);
CREATE INDEX IF NOT EXISTS ix_images_event ON images(event_id);
CREATE INDEX IF NOT EXISTS ix_images_status ON images(download_status);
CREATE INDEX IF NOT EXISTS ix_images_sha ON images(sha256);
CREATE INDEX IF NOT EXISTS ix_events_species ON events(species);
CREATE INDEX IF NOT EXISTS ix_events_class ON events(class);
CREATE INDEX IF NOT EXISTS ix_docs_kind ON documents(kind);
CREATE INDEX IF NOT EXISTS ix_runs_fecha ON runs(created_utc);
"""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def lic_ok(s) -> bool:
    s = (s or "").lower()
    if any(b in s for b in BAD):
        return False
    return any(g in s for g in GOOD)


def connect(cfg=None, ruta: str | Path | None = None) -> sqlite3.Connection:
    from config import CFG

    cfg = cfg or CFG
    path = Path(ruta) if ruta else cfg.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


# columnas añadidas en versiones posteriores del esquema: se agregan a bases existentes sin perder datos
MIGRACIONES = {"runs": [("versiones", "TEXT"), ("metricas", "TEXT")], "geo": [("motivo", "TEXT")]}


def _migrar(con) -> None:
    for tabla, cols in MIGRACIONES.items():
        existentes = {r[1] for r in con.execute(f"PRAGMA table_info({tabla})")}
        for c, tipo in cols:
            if c not in existentes:
                con.execute(f"ALTER TABLE {tabla} ADD COLUMN {c} {tipo}")


def init(cfg=None, ruta=None) -> sqlite3.Connection:
    con = connect(cfg, ruta)
    con.executescript(ESQUEMA)
    _migrar(con)
    con.execute("INSERT OR REPLACE INTO meta VALUES('version', ?)", (str(VERSION_ESQUEMA),))
    con.commit()
    return con


def auditar(con, accion: str, detalle="", n: int | None = None) -> None:
    """Deja constancia de una operación sobre los datos o la configuración (importar, verificar, limpiar…)."""
    if not isinstance(detalle, str):
        detalle = json.dumps(detalle, ensure_ascii=False, default=str)
    con.execute("INSERT INTO auditoria VALUES(?,?,?,?)", (now(), accion, detalle[:4000], n))


@contextmanager
def transaccion(con: sqlite3.Connection):
    """Agrupa varias escrituras en una sola transacción."""
    try:
        con.execute("BEGIN IMMEDIATE") if not con.in_transaction else None
        yield con
        con.commit()
    except BaseException:
        con.rollback()
        raise


# ------------------------------------------------------------------ resúmenes y snapshots
def resumen(con) -> dict:
    q = lambda s: con.execute(s).fetchone()[0]
    return dict(
        eventos=q("SELECT COUNT(*) FROM events"), puntos=q("SELECT COUNT(*) FROM geo"), textos_bd=q("SELECT COUNT(*) FROM texts"),
        audios=q("SELECT COUNT(*) FROM audio"), audios_ok=q(f"SELECT COUNT(*) FROM audio WHERE download_status='{CORRECTO}'"),
        imagenes=q("SELECT COUNT(*) FROM images"), imagenes_ok=q(f"SELECT COUNT(*) FROM images WHERE download_status='{CORRECTO}'"),
        invalidos=q(f"SELECT (SELECT COUNT(*) FROM audio WHERE download_status IN ('{INVALIDO}','{OMITIDO}'))"
                    f"+(SELECT COUNT(*) FROM images WHERE download_status IN ('{INVALIDO}','{OMITIDO}'))"),
        documentos=q("SELECT COUNT(*) FROM documents"), pendientes=q(
            f"SELECT (SELECT COUNT(*) FROM audio WHERE download_status IN ('{PENDIENTE}','{FALLIDO}'))"
            f"+(SELECT COUNT(*) FROM images WHERE download_status IN ('{PENDIENTE}','{FALLIDO}'))"),
        ejecuciones=q("SELECT COUNT(*) FROM runs"))


def crear_snapshot(con, event_ids: list[str] | None = None, nota: str = "") -> str:
    """Instantánea de un conjunto de eventos con manifiesto de integridad SHA-256."""
    ids = sorted(event_ids or [r[0] for r in con.execute("SELECT event_id FROM events")])
    filas = []
    for eid in ids:
        a = [r[0] for r in con.execute("SELECT sha256 FROM audio WHERE event_id=? ORDER BY audio_id", (eid,))]
        i = [r[0] for r in con.execute("SELECT sha256 FROM images WHERE event_id=? ORDER BY image_id", (eid,))]
        filas.append([eid, a, i])
    manifiesto = hashlib.sha256(json.dumps(filas, sort_keys=True).encode()).hexdigest()
    sid = "snap-" + manifiesto[:12]
    with transaccion(con):
        con.execute("INSERT OR REPLACE INTO snapshots VALUES(?,?,?,?)", (sid, now(), nota, manifiesto))
        con.execute("DELETE FROM snapshot_events WHERE snapshot_id=?", (sid,))
        con.executemany("INSERT INTO snapshot_events VALUES(?,?)", [(sid, e) for e in ids])
    return sid


def verificar_snapshot(con, sid: str) -> bool:
    fila = con.execute("SELECT manifest_sha256 FROM snapshots WHERE snapshot_id=?", (sid,)).fetchone()
    if not fila:
        return False
    ids = sorted(r[0] for r in con.execute("SELECT event_id FROM snapshot_events WHERE snapshot_id=?", (sid,)))
    filas = []
    for eid in ids:
        a = [r[0] for r in con.execute("SELECT sha256 FROM audio WHERE event_id=? ORDER BY audio_id", (eid,))]
        i = [r[0] for r in con.execute("SELECT sha256 FROM images WHERE event_id=? ORDER BY image_id", (eid,))]
        filas.append([eid, a, i])
    return hashlib.sha256(json.dumps(filas, sort_keys=True).encode()).hexdigest() == fila[0]

def conjuntos_experimentales(con):
    """
    Identifica los eventos utilizables en cada experimento.
    No elimina ni modifica registros.
    """

    # Eventos con al menos un audio y una imagen descargados correctamente.
    multimodal = con.execute("""
        SELECT e.event_id, e.species, e.class
        FROM events e
        WHERE e.species IS NOT NULL
          AND TRIM(e.species) <> ''
          AND EXISTS (
              SELECT 1 FROM audio a
              WHERE a.event_id = e.event_id
                AND a.download_status = 'correcto'
                AND a.local_path IS NOT NULL
                AND a.sha256 IS NOT NULL
          )
          AND EXISTS (
              SELECT 1 FROM images i
              WHERE i.event_id = e.event_id
                AND i.download_status = 'correcto'
                AND i.local_path IS NOT NULL
                AND i.sha256 IS NOT NULL
          )
        ORDER BY e.species
    """).fetchall()

    # Todos los eventos, incluidos los que tienen modalidades incompletas.
    anomalías = con.execute("""
        SELECT
            e.event_id,
            e.species,
            e.class,
            e.source,
            e.status,
            g.lat,
            g.lon,
            g.geo_status,
            CASE WHEN EXISTS (
                SELECT 1 FROM audio a
                WHERE a.event_id = e.event_id
                  AND a.download_status = 'correcto'
            ) THEN 1 ELSE 0 END AS tiene_audio,
            CASE WHEN EXISTS (
                SELECT 1 FROM images i
                WHERE i.event_id = e.event_id
                  AND i.download_status = 'correcto'
            ) THEN 1 ELSE 0 END AS tiene_imagen,
            CASE WHEN EXISTS (
                SELECT 1 FROM texts t
                WHERE t.event_id = e.event_id
                  AND TRIM(COALESCE(t.original,'')) <> ''
            ) THEN 1 ELSE 0 END AS tiene_texto
        FROM events e
        LEFT JOIN geo g ON g.event_id = e.event_id
        ORDER BY e.event_id
    """).fetchall()

    return {
        "multimodal": [dict(x) for x in multimodal],
        "anomalias": [dict(x) for x in anomalías]
    }


if __name__ == "__main__":
    init()
    print("Base de datos lista")
