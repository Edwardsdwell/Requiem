"""Coordenadas (JSON, GeoJSON, CSV, TXT y puntos de la BD) -> matriz en km.

* Reconoce nombres habituales de latitud/longitud, incluidos los españoles.
* GeoJSON usa [lon, lat]: se invierte el orden al leer.
* Proyección **equirectangular local** (km respecto al centroide): solo es fiable en áreas pequeñas
  (unos cientos de km). Con extensiones mayores el modo «auto» usa coordenadas cartesianas sobre la
  esfera terrestre (x, y, z en km), que no distorsionan las distancias, y deja un aviso.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from core.cache import hash_obj
from core.errors import ErrorDatos
from core.results import Datos
from data.validacion import validar_coordenada
from . import Recurso

TIPO, NOMBRE, VERSION = "geo", "Coordenadas", "geo-3"
# nombres de columna habituales (incluye los de GBIF/Darwin Core y en español)
_LATS = ("lat", "latitude", "latitud", "decimallatitude", "decimal_latitude", "lat_dd", "y", "pickup_centroid_latitude")
_LONS = ("lon", "lng", "long", "longitude", "longitud", "decimallongitude", "decimal_longitude", "lon_dd", "lng_dd", "x",
         "pickup_centroid_longitude")


def _clave_col(nombres, candidatos):
    c = {str(x).lower().strip().replace(" ", "_"): x for x in nombres}
    return next((c[n] for n in candidatos if n in c), None)


def _extraer(o, sal: list, desc: list, geojson=False, ruta="") -> None:
    """Recorre cualquier estructura JSON y añade a `sal` (lat, lon, props) válidos y a `desc` (posición, motivo)
    los descartados. Un elemento malo no interrumpe la lectura del resto."""
    if isinstance(o, dict):
        g = o.get("geometry")
        if "geometry" in o and ("properties" in o or o.get("type") == "Feature"):          # Feature GeoJSON
            props = o.get("properties") or {}
            if not isinstance(g, dict) or g.get("type") != "Point":
                desc.append((ruta, f"geometría no puntual o ausente ({(g or {}).get('type') if isinstance(g, dict) else g})"))
                return
            c = g.get("coordinates")
            if not isinstance(c, (list, tuple)) or len(c) < 2:
                desc.append((ruta, "Point sin coordenadas"))
                return
            lat, lon, motivo = validar_coordenada(c[1], c[0])                                 # GeoJSON: [lon, lat]
            (desc.append((ruta, motivo)) if motivo else sal.append((lat, lon, props)))
            return
        k = _clave_col(o.keys(), _LATS), _clave_col(o.keys(), _LONS)
        if k[0] is not None and k[1] is not None and not isinstance(o[k[0]], (dict, list)):
            lat, lon, motivo = validar_coordenada(o[k[0]], o[k[1]])
            props = {a: v for a, v in o.items() if a not in k and not isinstance(v, (dict, list))}
            (desc.append((ruta, motivo)) if motivo else sal.append((lat, lon, props)))
            return
        for c, v in o.items():
            _extraer(v, sal, desc, geojson or c == "coordinates", f"{ruta}/{c}")
        return
    if isinstance(o, list):
        if len(o) >= 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in o[:2]):
            a, b = (o[1], o[0]) if geojson else (o[0], o[1])
            lat, lon, motivo = validar_coordenada(a, b)
            (desc.append((ruta, motivo)) if motivo else sal.append((lat, lon, {})))
            return
        for i, v in enumerate(o):
            _extraer(v, sal, desc, geojson, f"{ruta}[{i}]")


def _leer_tabla(ruta: Path, pts: list, desc: list) -> bool:
    """CSV/TXT con cabecera de latitud y longitud. Devuelve False si no hay columnas reconocibles."""
    import pandas as pd

    try:
        df = pd.read_csv(ruta, sep=None, engine="python", dtype=str, encoding_errors="replace", on_bad_lines="warn")
    except Exception:  # noqa: BLE001
        return False
    lat, lon = _clave_col(df.columns, _LATS), _clave_col(df.columns, _LONS)
    if lat is None or lon is None:
        return False
    for i, r in enumerate(df.to_dict("records")):
        a, b, motivo = validar_coordenada(r.get(lat), r.get(lon))
        if motivo:
            desc.append((f"fila {i + 2}", motivo))
        else:
            pts.append((a, b, {k: v for k, v in r.items() if k not in (lat, lon) and isinstance(v, str) and v.strip()}))
    return True


def leer_geo_informe(ruta: Path) -> tuple[list, list]:
    """(puntos válidos [(lat, lon, props)], descartados [(posición, motivo)]). Nunca aborta por un dato malo:
    coordenadas fuera de rango, texto, (0, 0) o geometrías no puntuales quedan descartadas **con su motivo**."""
    ruta = Path(ruta)
    pts, desc = [], []
    if ruta.stat().st_size == 0:
        return [], [("archivo", "archivo vacío")]
    if ruta.suffix.lower() in (".json", ".geojson"):
        try:
            obj = json.loads(ruta.read_text(encoding="utf-8-sig", errors="replace"))
        except json.JSONDecodeError as e:
            return [], [("archivo", f"JSON mal formado (línea {e.lineno}): {e.msg}")]
        _extraer(obj, pts, desc, ruta="$")
        return pts, desc
    if not _leer_tabla(ruta, pts, desc):                       # TXT sin cabecera: pares de números por línea
        for i, linea in enumerate(ruta.read_text(encoding="utf-8", errors="replace").splitlines()):
            n = re.findall(r"[-+]?\d+(?:[.,]\d+)?", linea)
            if len(n) >= 2:
                a, b, motivo = validar_coordenada(n[0], n[1])
                desc.append((f"línea {i + 1}", motivo)) if motivo else pts.append((a, b, {}))
            elif linea.strip():
                desc.append((f"línea {i + 1}", "no contiene un par de coordenadas"))
    return pts, desc


def leer_geo(ruta: Path) -> list:
    """Puntos válidos (lat, lon, propiedades). Ver `leer_geo_informe` para conocer los descartados."""
    return leer_geo_informe(ruta)[0]


def _puntos_bd(cfg, con=None, descartes: list | None = None):
    """Puntos de la BD con validación: los inválidos se informan en `descartes` (event_id, motivo)."""
    from data import db

    c = con or db.connect(cfg)
    q = ("SELECT g.event_id, g.lat, g.lon, g.country, e.source_id, e.species, e.class FROM geo g JOIN events e USING(event_id) "
         "ORDER BY g.event_id")
    out = []
    for f in c.execute(q):
        lat, lon, motivo = validar_coordenada(f["lat"], f["lon"])
        if motivo:
            if descartes is not None:
                descartes.append((f["event_id"], motivo))
            continue
        out.append((lat, lon, dict(event_id=f["event_id"], id_origen=f["source_id"], especie=f["species"],
                                   clase=f["class"] or "sin clase", pais=f["country"])))
    return out


def proyectar(ll: np.ndarray, cfg) -> tuple:
    """(X, nombres, centro, modo, avisos) según la extensión de los puntos."""
    ext = max(np.ptp(ll[:, 0]), np.ptp(ll[:, 1]))
    modo = cfg.geo_proyeccion if cfg.geo_proyeccion in ("local", "esfera") else ("local" if ext <= cfg.geo_extension_local_deg else "esfera")
    avisos = []
    if modo == "esfera":
        R, la, lo = 6371.0, np.radians(ll[:, 0]), np.radians(ll[:, 1])
        X = np.c_[R * np.cos(la) * np.cos(lo), R * np.cos(la) * np.sin(lo), R * np.sin(la)]
        if cfg.geo_proyeccion == "auto":
            avisos.append(f"Los puntos abarcan {ext:.0f}° de extensión: se usan coordenadas cartesianas sobre la esfera terrestre (x, y, z en km) "
                          "en lugar de la proyección local, que solo es válida en áreas pequeñas.")
        return X, ["x (km)", "y (km)", "z (km)"], None, modo, avisos
    lat0, lon0 = ll.mean(0)
    X = np.c_[(ll[:, 1] - lon0) * 111.32 * np.cos(np.radians(lat0)), (ll[:, 0] - lat0) * 110.57]
    if ext > cfg.geo_extension_local_deg:
        avisos.append(f"La proyección local (equirectangular) se está usando con {ext:.0f}° de extensión: las distancias lejos del centro "
                      "estarán distorsionadas (válida para áreas de unos cientos de km).")
    return X, ["x (km)", "y (km)"], (lat0, lon0), modo, avisos


def cargar(recursos: list[Recurso], cfg, ctx=None, con=None) -> Datos:
    pts, org, meta, descartes = [], [], [], []
    for i, r in enumerate(recursos):
        if ctx is not None:
            ctx.punto(i / len(recursos), f"Coordenadas {i + 1}/{len(recursos)}: {r.nombre}")
        if r.virtual:
            malos: list = []
            p = _puntos_bd(cfg, con, malos)
            etiquetas = [x[2]["clase"] for x in p]
        else:
            p, malos = leer_geo_informe(r.ruta)
            etiquetas = [r.nombre] * len(p)
        descartes += [(r.nombre, pos, mot) for pos, mot in malos]
        pts += p
        org += etiquetas
        meta += [{**r.meta, **{str(k): v for k, v in x[2].items()}, "archivo": r.nombre, "sha256": r.sha256} for x in p]
    if len(pts) < 5:
        from collections import Counter

        por = "; ".join(f"{m} ({n})" for m, n in Counter(m for *_, m in descartes).most_common(3))
        raise ErrorDatos(f"Solo hay {len(pts)} coordenadas válidas (se necesitan al menos 5)",
                         f"{len(descartes)} descartadas: {por}" if descartes else "No se encontraron columnas de latitud/longitud.")
    ll = np.array([(a, b) for a, b, _ in pts], dtype=float)
    X, nombres, centro, modo, avisos = proyectar(ll, cfg)
    if descartes:                                             # nada se pierde en silencio
        from collections import Counter

        top = "; ".join(f"{m} ({n})" for m, n in Counter(m for *_, m in descartes).most_common(3))
        avisos = [f"{len(descartes)} coordenadas descartadas por no ser válidas: {top}"] + avisos
    hov = np.array([f"{(m.get('especie') + ' · ') if m.get('especie') else ''}{o} · {a:.4f}, {b:.4f}" for o, m, (a, b) in zip(org, meta, ll)])
    d = Datos(X, nombres, np.array(org), hov, False, ll, centro, f"{len(ll)} puntos ({'esfera terrestre, km' if modo == 'esfera' else 'proyección local, km'})",
              TIPO, [(a, b) for a, b in ll], meta, {r.nombre: r.ruta for r in recursos if r.ruta}, avisos=avisos, proyeccion=modo)
    d.huella = hash_obj(TIPO, sorted(r.sha256 for r in recursos), VERSION, modo, cfg.max_muestras, cfg.semilla)
    d.descartes = descartes
    return d
