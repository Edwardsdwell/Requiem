"""Servicio de descarga multimedia, independiente del análisis.

Conserva la lógica del `download.py` original (licencia, URL, streaming de 64 KB, MIME, límites de
80/25 MB, SHA-256 durante la descarga, pausa de 1 s, errores individuales) y corrige sus puntos débiles:

* un temporal **único** por descarga (el original compartía `_tmp` por carpeta);
* escritura atómica: el archivo definitivo solo existe si la transferencia terminó bien;
* si el SHA-256 ya existe en el almacén no se guarda de nuevo (sin duplicados);
* reintentos con espera incremental solo ante fallos temporales;
* URLs restringidas a http/https, sin IP privadas ni redirecciones peligrosas (anti-SSRF);
* cancelación cooperativa sin dejar temporales ni estados a medias;
* descargar NO procesa: no se importa librosa, PIL ni scikit-learn.
"""
from __future__ import annotations

import hashlib
import ipaddress
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

from core.errors import Cancelado, ErrorDescarga
from . import db
from .storage import EXT, MIME_EXT, Almacen

TEMPORALES = (429, 500, 502, 503, 504)


# ------------------------------------------------------------------ seguridad 
def validar_url(url: str, permitir_privadas: bool = False) -> str:
    u = urlparse((url or "").strip())
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ErrorDescarga("URL no válida: solo se admiten http(s)")
    if u.username or u.password:
        raise ErrorDescarga("URL con credenciales incrustadas no admitida")
    if not permitir_privadas:
        try:
            infos = socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
        except socket.gaierror as e:
            raise ErrorDescarga(f"No se pudo resolver {u.hostname}") from e
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
                raise ErrorDescarga("Dirección privada o interna bloqueada")
    return u.geturl()


@dataclass
class Resultado:
    estado: str
    ruta: Path | None = None
    sha256: str | None = None
    bytes: int = 0
    mime: str | None = None
    duplicado: bool = False
    error: str | None = None


class Descargador:
    def __init__(self, cfg, almacen: Almacen | None = None, sesion=None):
        self.cfg = cfg
        self.almacen = almacen or Almacen(cfg)
        self._sesion = sesion

    @property
    def sesion(self):
        if self._sesion is None:
            import requests  # carga diferida 

            self._sesion = requests.Session()
            self._sesion.headers["User-Agent"] = self.cfg.user_agent
        return self._sesion

    # ------------------------------------------------------------------ una descarga
    def _abrir(self, url: str):
        """GET en streaming siguiendo las redirecciones a mano para validar cada salto."""
        actual = validar_url(url, self.cfg.permitir_privadas)
        for _ in range(self.cfg.max_redirecciones + 1):
            r = self.sesion.get(actual, stream=True, timeout=self.cfg.timeout_s, allow_redirects=False)
            if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
                actual = validar_url(urljoin(actual, r.headers["Location"]), self.cfg.permitir_privadas)
                r.close()
                continue
            return r
        raise ErrorDescarga("Demasiadas redirecciones")

    def descargar(self, url: str, tipo: str, ctx=None) -> Resultado:
        """Descarga una URL al almacén. `tipo` es 'audio' o 'img'. Lanza ErrorDescarga o Cancelado."""
        limite = self.almacen.limite_bytes(tipo)
        tmp = self.almacen.nuevo_temporal()
        r = None
        try:
            r = self._abrir(url)
            if r.status_code != 200:
                e = ErrorDescarga(f"HTTP {r.status_code}")
                e.temporal = r.status_code in TEMPORALES                       # type: ignore[attr-defined]
                raise e
            mime = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            ext = MIME_EXT.get(mime) or self._ext_url(url, tipo)
            if ext is None or ext not in EXT[tipo]:
                raise ErrorDescarga(f"Contenido no compatible con «{tipo}» (MIME {mime or 'desconocido'})")
            declarado = int(r.headers.get("Content-Length") or 0)
            if declarado > limite:
                raise ErrorDescarga(f"Supera el límite de {limite // 2**20} MB")
            h, n = hashlib.sha256(), 0
            with open(tmp, "wb") as f:
                for b in r.iter_content(self.cfg.chunk_bytes):
                    if ctx is not None:
                        ctx.punto()
                    n += len(b)
                    if n > limite:
                        raise ErrorDescarga(f"Supera el límite de {limite // 2**20} MB")
                    h.update(b)
                    f.write(b)
            if n == 0:
                raise ErrorDescarga("Archivo vacío")
            if declarado and n != declarado and not r.headers.get("Content-Encoding"):
                e = ErrorDescarga("Transferencia incompleta")
                e.temporal = True                                              # type: ignore[attr-defined]
                raise e
            ruta, dup = self.almacen.colocar(tmp, tipo, h.hexdigest(), ext)
            return Resultado(db.CORRECTO, ruta, h.hexdigest(), n, mime or None, dup)
        finally:
            if r is not None:
                r.close()
            tmp.unlink(missing_ok=True)                    # nunca queda un temporal huérfano

    @staticmethod
    def _ext_url(url: str, tipo: str) -> str | None:
        ext = Path(urlparse(url).path).suffix.lower()
        return ext if ext in EXT[tipo] else None

    def con_reintentos(self, url: str, tipo: str, ctx=None) -> Resultado:
        """Reintenta con espera incremental (base·2^i) solo fallos temporales de red."""
        ultimo: Exception | None = None
        for i in range(self.cfg.reintentos + 1):
            try:
                return self.descargar(url, tipo, ctx)
            except Cancelado:
                raise
            except ErrorDescarga as e:
                if not getattr(e, "temporal", False):
                    raise
                ultimo = e
            except Exception as e:  # noqa: BLE001 - errores de red de requests
                if type(e).__module__.split(".")[0] != "requests":
                    raise
                ultimo = ErrorDescarga(f"Red: {e}")
            if i < self.cfg.reintentos:
                self._esperar(self.cfg.espera_base_s * 2**i, ctx)
        raise ErrorDescarga(f"Falló tras {self.cfg.reintentos + 1} intentos: {ultimo}")

    @staticmethod
    def _esperar(s: float, ctx=None):
        fin = time.time() + s
        while time.time() < fin:
            if ctx is not None:
                ctx.punto()
            time.sleep(min(0.1, max(0.0, fin - time.time())))

    # ------------------------------------------------------------------ lote desde la base de datos
    def pendientes(self, con, tipo: str | None = None, limite: int | None = None, reintentar_fallidos: bool = True) -> list[dict]:
        """Registros por descargar. Los ya disponibles (con archivo íntegro) no se vuelven a pedir."""
        estados = [db.PENDIENTE, ""] + ([db.FALLIDO] if reintentar_fallidos else [])
        out = []
        for t, tabla, idc in (("audio", "audio", "audio_id"), ("img", "images", "image_id")):
            if tipo and tipo != t:
                continue
            q = (f"SELECT {idc} AS id, url, license, sha256, local_path, download_status FROM {tabla} "
                 f"WHERE (download_status IS NULL OR download_status IN ({','.join('?' * len(estados))})) AND attempts < ? ORDER BY {idc}")
            for f in con.execute(q, (*estados, self.cfg.reintentos + 3)):
                out.append(dict(f, tipo=t, tabla=tabla, idc=idc))
        return out[:limite] if limite else out

    def descargar_pendientes(self, con, tipo=None, limite=None, ctx=None, pausa: float | None = None) -> dict:
        """Procesa los pendientes con errores individuales aislados y confirmaciones por registro."""
        pausa = self.cfg.pausa_descarga_s if pausa is None else pausa
        cola = self.pendientes(con, tipo, limite)
        res = dict(total=len(cola), correctos=0, omitidos=0, fallidos=0, invalidos=0, duplicados=0,
                   guardados_en_postgres=0, fallidos_en_postgres=0, errores_postgres=[], cancelada=False)
        for i, reg in enumerate(cola):
            try:
                if ctx is not None:
                    ctx.punto(i / max(len(cola), 1), f"Descargando {i + 1}/{len(cola)}")
                tabla, idc = reg["tabla"], reg["idc"]
                # No se decide elegibilidad jurídica aquí: el usuario revisa la
                # licencia conservada en los metadatos antes de utilizar cada medio.
                if not reg["url"]:
                    self._marcar(con, tabla, idc, reg["id"], db.OMITIDO, error="sin URL")
                    res["omitidos"] += 1
                    continue
                # ¿ya hay un archivo íntegro con ese hash? (otro registro apunta al mismo contenido)
                if reg["sha256"]:
                    ex = self.almacen.buscar_sha(reg["tipo"], reg["sha256"])
                    if ex and self.almacen.verificar(ex, reg["sha256"]):
                        self._marcar(con, tabla, idc, reg["id"], db.CORRECTO, ruta=ex, sha=reg["sha256"])
                        res["correctos"] += 1
                        if self.cfg.database_url and str(reg["id"]).startswith("pg:"):
                            try:
                                self._guardar_contenido_postgres(reg, ex, reg["sha256"], None, duplicado=True)
                                res["guardados_en_postgres"] += 1
                            except Exception as e:
                                res["fallidos_en_postgres"] += 1
                                res["errores_postgres"].append(f"{reg['id']}: {e}")
                        continue
                try:
                    r = self.con_reintentos(reg["url"], reg["tipo"], ctx)
                except Cancelado:
                    raise
                except Exception as e:  # noqa: BLE001 - un fallo individual no detiene el lote
                    self._marcar(con, tabla, idc, reg["id"], db.FALLIDO, error=str(e)[:300], intento=True)
                    res["fallidos"] += 1
                else:
                    from .integridad import guardar_veredicto
                    from .validacion import validar_archivo

                    self._marcar(con, tabla, idc, reg["id"], db.CORRECTO, ruta=r.ruta, sha=r.sha256, bytes_=r.bytes, mime=r.mime)
                    v = validar_archivo(r.ruta, reg["tipo"])          # descargado ≠ válido: se comprueba que se pueda abrir
                    guardar_veredicto(con, tabla, idc, reg["id"], v)
                    con.commit()
                    if v.ok:
                        res["correctos"] += 1
                        res["duplicados"] += int(r.duplicado)
                        if self.cfg.database_url and str(reg["id"]).startswith("pg:"):
                            try:
                                self._guardar_contenido_postgres(reg, r.ruta, r.sha256, r.mime, r.duplicado)
                                res["guardados_en_postgres"] += 1
                            except Exception as e:  # el medio local sigue disponible aunque falle la escritura del BLOB
                                res["fallidos_en_postgres"] += 1
                                res["errores_postgres"].append(f"{reg['id']}: {e}")
                    else:
                        res["invalidos"] = res.get("invalidos", 0) + 1
                    if pausa and i < len(cola) - 1:
                        self._esperar(pausa, ctx)
            except Cancelado:
                res["cancelada"] = True
                break
        self.almacen.limpiar_temporales(forzar=False)
        return res

    def _guardar_contenido_postgres(self, registro, ruta: Path, sha: str, mime: str | None, duplicado=False) -> None:
        import mimetypes
        import psycopg

        media_id = int(str(registro["id"]).split(":", 1)[1])
        datos = ruta.read_bytes()
        tipo_mime = mime or mimetypes.guess_type(ruta.name)[0] or "application/octet-stream"
        ruta_relativa = self.almacen.relativa(ruta)
        with psycopg.connect(self.cfg.database_url) as pg:
            repetido = pg.execute("""SELECT 1 FROM media
                WHERE media_id<>%s AND sha256=%s AND download_status='ok' LIMIT 1""", (media_id, sha)).fetchone()
            estado = "duplicate" if duplicado or repetido else "ok"
            pg.execute("""INSERT INTO media_content(media_id,content,mime_type,size_bytes,sha256,source_path)
                VALUES(%s,%s,%s,%s,%s,%s)
                ON CONFLICT(media_id) DO UPDATE SET content=EXCLUDED.content,mime_type=EXCLUDED.mime_type,
                    size_bytes=EXCLUDED.size_bytes,sha256=EXCLUDED.sha256,source_path=EXCLUDED.source_path,
                    imported_at=now()""", (media_id, datos, tipo_mime, len(datos), sha, ruta_relativa))
            pg.execute("""UPDATE media SET file_path=%s,sha256=%s,bytes=%s,download_status=%s,error=NULL
                WHERE media_id=%s""", (ruta_relativa, sha, len(datos), estado, media_id))

    def _marcar(self, con, tabla, idc, id_, estado, ruta: Path | None = None, sha=None, bytes_=None, mime=None, error=None, intento=False):
        con.execute(
            f"UPDATE {tabla} SET download_status=?, local_path=COALESCE(?, local_path), sha256=COALESCE(?, sha256), "
            f"size_bytes=COALESCE(?, size_bytes), mime=COALESCE(?, mime), error=?, updated_utc=?, attempts=attempts+? WHERE {idc}=?",
            (estado, self.almacen.relativa(ruta) if ruta else None, sha, bytes_, mime, error, db.now(), int(intento), id_))
        con.commit()
