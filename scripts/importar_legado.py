"""Importa el proyecto alternativo (fauna_sonora: db/fauna.db + data/raw/{audio,images}) al esquema unificado.

    python -m scripts.importar_legado RUTA/AL/proyecto_original [--sin-medios]

* Acepta la estructura de fauna_sonora (`db/fauna.db` y `data/raw/`) y la plana (`fauna.db` junto a `raw/`),
  rutas de Windows o POSIX, con o sin comillas.
* Copia los registros fila a fila: una fila problemática se informa y **no** detiene la importación.
* Coordenadas inválidas (texto, fuera de rango, (0,0)) se conservan marcadas como `invalido` con su motivo.
* Copia cada archivo a `data_store/raw/<tipo>/<sha256>.<ext>` sin duplicar contenido, comprueba su SHA-256
  y que se pueda abrir (archivos vacíos o corruptos quedan `invalido` con el motivo).
* Al final verifica que no falte ningún archivo y devuelve un informe completo (válidos / inválidos / pendientes).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG  # noqa: E402
from data import db  # noqa: E402
from data.integridad import guardar_veredicto, verificar  # noqa: E402
from data.storage import Almacen, sha256_archivo  # noqa: E402
from data.validacion import limpiar_ruta_usuario, nombre_de_ruta, validar_archivo, validar_coordenada  # noqa: E402

TABLAS = ["events", "geo", "texts", "exclusions", "ingest_log", "snapshots", "snapshot_events"]
MEDIOS = (("audio", "audio", "audio_id", "audio"), ("img", "images", "image_id", "images"))


def localizar(origen: Path) -> tuple[Path, Path]:
    """(ruta de fauna.db, carpeta que contiene raw/). Busca hasta dos niveles por debajo de `origen`."""
    origen = limpiar_ruta_usuario(origen)
    if origen.is_file() and origen.name.lower().endswith(".db"):
        bases = [origen.parent]
        db_dir = origen.parent
    else:
        if not origen.is_dir():
            raise SystemExit(f"No existe la carpeta: {origen}")
        bases = [origen, *sorted(origen.glob("*")), *sorted(origen.glob("*/*"))]
        db_dir = next((b for b in bases if b.is_dir() and (b / "fauna.db").is_file()), None)
        if db_dir is None:
            raise SystemExit(f"No se encontró fauna.db dentro de {origen} (ni en sus subcarpetas)")
    ruta_db = origen if origen.is_file() else db_dir / "fauna.db"
    for cand in (db_dir, db_dir / "data", db_dir.parent / "data", db_dir.parent, origen / "data", origen):
        if (cand / "raw").is_dir():
            return ruta_db, cand
    return ruta_db, db_dir


def _buscar_archivo(raw: Path, carpeta: str, local_path: str | None, sha: str | None) -> Path | None:
    """Encuentra el archivo de un registro: por el nombre de su ruta (Windows o POSIX) o por su SHA-256."""
    d = raw / carpeta
    nombre = nombre_de_ruta(local_path)
    if nombre and (d / nombre).is_file():
        return d / nombre
    if sha:
        for p in d.glob(f"{sha}.*"):
            if p.is_file():
                return p
    if nombre:                                            # mayúsculas/minúsculas distintas (Windows no distingue)
        bajo = nombre.lower()
        for p in d.glob("*"):
            if p.name.lower() == bajo:
                return p
    return None


def _copiar_tabla(con, leg, t: str, rep: dict) -> None:
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
    lc = [r[1] for r in leg.execute(f"PRAGMA table_info({t})")]
    if not lc:
        return
    comunes = [c for c in cols if c in lc]
    ignoradas = [c for c in lc if c not in cols]
    if ignoradas:
        rep["columnas_ignoradas"][t] = ignoradas
    filas = leg.execute(f"SELECT {','.join(comunes)} FROM {t}").fetchall()
    ok = 0
    for f in filas:
        f = dict(zip(comunes, tuple(f)))
        if t == "geo":                                     # coordenadas inválidas: se conservan marcadas, no se pierden
            lat, lon, motivo = validar_coordenada(f.get("lat"), f.get("lon"))
            if motivo:
                f["geo_status"], f["motivo"] = "invalido", motivo
                rep["geo_invalidos"].append((f.get("event_id"), motivo))
            else:
                f["lat"], f["lon"] = lat, lon
        try:
            cs = list(f)
            cur = con.execute(f"INSERT OR IGNORE INTO {t}({','.join(cs)}) VALUES({','.join('?' * len(cs))})", tuple(f.values()))
            ok += cur.rowcount if cur.rowcount > 0 else 0
            if cur.rowcount == 0:
                rep["ya_existian"][t] = rep["ya_existian"].get(t, 0) + 1
        except sqlite3.Error as e:                         # p. ej. un texto cuyo evento no existe
            rep["filas_con_error"].append((t, str(f.get("event_id") or f)[:60], str(e)))
    rep["filas"][t] = len(filas)
    rep["insertadas"][t] = ok


def importar(origen: Path, cfg=CFG, medios: bool = True, verbose: bool = True, ctx=None) -> dict:
    ruta_db, base = localizar(Path(origen))
    almacen = Almacen(cfg)
    con = db.init(cfg)
    leg = sqlite3.connect(ruta_db.resolve().as_uri() + "?mode=ro", uri=True)   # el original se abre en solo lectura (URI válida también con espacios)
    leg.row_factory = sqlite3.Row
    rep = dict(origen=str(ruta_db), carpeta_raw=str(base / "raw"), filas={}, insertadas={}, ya_existian={}, columnas_ignoradas={},
               filas_con_error=[], geo_invalidos=[], copiados=0, duplicados=0, faltan=0, hash_distinto=0, invalidos=0,
               faltantes=[], motivos_invalidos={}, estados={})
    try:
        for t in TABLAS:
            _copiar_tabla(con, leg, t, rep)
        con.commit()
        motivos: Counter = Counter()
        for tipo, tabla, idc, carpeta in MEDIOS:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({tabla})")]
            lc = [r[1] for r in leg.execute(f"PRAGMA table_info({tabla})")]
            comunes = [c for c in cols if c in lc]
            todas = leg.execute(f"SELECT {','.join(comunes)} FROM {tabla}").fetchall()
            n = 0
            for j, f in enumerate(todas):
                if ctx is not None:
                    ctx.punto(j / max(len(todas), 1), f"Importando {tabla} {j + 1}/{len(todas)}")
                f = dict(zip(comunes, tuple(f)))
                sha_reg = (f.get("sha256") or "").lower() or None
                arch = _buscar_archivo(base / "raw", carpeta, f.get("local_path"), sha_reg) if medios else None
                estado, ruta_rel, sha, error, v = db.PENDIENTE, None, sha_reg, None, None
                if arch is not None:
                    v = validar_archivo(arch, tipo)                # primero: ¿vacío, corrupto, formato equivocado?
                    real = sha256_archivo(arch) if v.ok else None
                    if not v.ok:
                        estado, error = db.INVALIDO, v.motivo
                    elif sha_reg and real != sha_reg:
                        rep["hash_distinto"] += 1
                        estado, error = db.INVALIDO, "el contenido no coincide con el SHA-256 registrado (archivo modificado o dañado)"
                    else:
                        destino, real, dup = almacen.ingerir(arch, tipo, arch.name if arch.suffix else arch.name + ".bin")
                        rep["duplicados" if dup else "copiados"] += 1
                        estado, ruta_rel, sha, error = db.CORRECTO, almacen.relativa(destino), real, v.motivo or None
                elif not medios and (f.get("download_status") or "") in ("ok", db.CORRECTO):
                    estado, error = db.PENDIENTE, "importado sin medios: archivo no copiado"
                else:
                    rep["faltan"] += 1
                    rep["faltantes"].append(f.get(idc))
                    error = "archivo no encontrado en el proyecto alternativo (se puede volver a descargar)" if f.get("url") else "archivo no encontrado y sin URL"
                if estado == db.INVALIDO:
                    rep["invalidos"] += 1
                    motivos[error] += 1
                f.update(local_path=ruta_rel, sha256=sha, download_status=estado, error=error, updated_utc=db.now())
                cs = list(f)
                try:
                    con.execute(f"INSERT OR REPLACE INTO {tabla}({','.join(cs)}) VALUES({','.join('?' * len(cs))})", tuple(f.values()))
                    if v is not None and v.ok and estado == db.CORRECTO:
                        guardar_veredicto(con, tabla, idc, f[idc], v, ruta_rel, sha)
                    n += 1
                except sqlite3.Error as e:
                    rep["filas_con_error"].append((tabla, str(f.get(idc)), str(e)))
            rep["filas"][tabla] = len(todas)
            rep["insertadas"][tabla] = n
            con.commit()
        rep["motivos_invalidos"] = dict(motivos)
        rep["verificacion"] = {k: (len(v) if isinstance(v, list) else v) for k, v in verificar(con, almacen, validar_contenido=False).items()}
        for _, tabla, _, _ in MEDIOS:
            rep["estados"][tabla] = dict(con.execute(f"SELECT download_status, COUNT(*) FROM {tabla} GROUP BY 1").fetchall())
        db.auditar(con, "importar_legado", dict(origen=rep["origen"], copiados=rep["copiados"], invalidos=rep["invalidos"], faltan=rep["faltan"],
                                                 geo_invalidos=len(rep["geo_invalidos"]), filas_con_error=len(rep["filas_con_error"])),
                   sum(rep["filas"].values()))
        con.commit()
    finally:
        leg.close()
        con.close()
    if verbose:
        print("Importación terminada")
        for k in ("filas", "insertadas", "estados", "copiados", "duplicados", "faltan", "invalidos", "hash_distinto", "motivos_invalidos"):
            print(f"  {k}: {rep[k]}")
        if rep["geo_invalidos"]:
            print(f"  coordenadas inválidas conservadas: {len(rep['geo_invalidos'])}")
        if rep["filas_con_error"]:
            print(f"  filas con error: {rep['filas_con_error'][:5]}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("origen", help="carpeta del proyecto alternativo (fauna_sonora) o ruta a su fauna.db")
    ap.add_argument("--sin-medios", action="store_true", help="importa solo los registros, sin copiar archivos")
    a = ap.parse_args()
    importar(Path(a.origen), medios=not a.sin_medios)
