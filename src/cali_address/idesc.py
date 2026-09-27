"""Thin client for the IDESC (Alcaldía de Santiago de Cali) address geocoder.

Reverse engineering notes
------------------------
The public web app at ``https://geocodificador.ia.cali.gov.co`` ships a single
Vite bundle (``/assets/index-DxZufy6u.js``) that contains the REST routes:

* ``/geoapi/proceso/geocodificador_individual``  – normalize + geocode one address
* ``/geoapi/proceso/geo_inversa_individual``     – reverse geocode a coordinate
* ``/geoapi/proceso/geocodificador_archivo``     – bulk file upload
* ``/geoapi/proceso/validar_avance``             – bulk job progress
* ``/geoapi/proceso/descargar_archivo``          – bulk result download

The browser attaches a reCAPTCHA v3 ``captchaToken`` to the JSON body, but the
server does **not** validate it: a plain
``POST {"direccion": "<raw>"}`` with ``Content-Type: application/json``
returns the same payload. Response shape::

    data.datosUbicacion = {
        estado: "A - Normalizado y georreferenciado exacto" | "D - ..." | "F - ...",
        estrato, comuna,
        direcciones: {direccion, dir_ajusta},
        barrio: {codigo, nombre},
        geo_localizacion: {longitud, latitud},
    }

``dir_ajusta`` is the normalized address in the same canonical family as the
cadastral ``direccion`` field (letters glued to the number: ``KR 98F # 98 - 66``).
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import requests

__all__ = ["IdescClient", "ESTADO_CLASSES"]

ENDPOINT = "https://geocodificador.ia.cali.gov.co/geoapi/proceso/geocodificador_individual"

#: First letter of ``estado`` -> human readable meaning (as returned by IDESC).
ESTADO_CLASSES = {
    "A": "Normalizado y georreferenciado exacto",
    "D": "Normalizado y no georreferenciado",
    "F": "Normalizado por cruce y no georreferenciado",
}


class IdescClient:
    """Cached, thread-pooled client. Never re-queries an address already cached."""

    def __init__(
        self,
        cache_path: str,
        max_workers: int = 4,
        timeout: int = 60,
        retries: int = 3,
        flush_every: int = 25,
    ) -> None:
        self.cache_path = cache_path
        self.max_workers = max_workers
        self.timeout = timeout
        self.retries = retries
        self.flush_every = flush_every
        self._lock = threading.Lock()
        self._since_flush = 0
        self.cache: dict[str, dict] = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    self.cache = json.load(fh)
            except (json.JSONDecodeError, OSError):
                self.cache = {}
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "cali-address-normalizer/1.0 (research)",
                "Origin": "https://geocodificador.ia.cali.gov.co",
                "Referer": "https://geocodificador.ia.cali.gov.co/",
            }
        )

    # -- persistence ------------------------------------------------------
    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.cache, fh, ensure_ascii=False)
        os.replace(tmp, self.cache_path)
        self._since_flush = 0

    # -- single call ------------------------------------------------------
    def _request(self, address: str) -> dict:
        last_err = None
        for attempt in range(self.retries):
            try:
                r = self._session.post(
                    ENDPOINT, data=json.dumps({"direccion": address}), timeout=self.timeout
                )
                if r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                return {"ok": True, "payload": r.json()}
            except Exception as exc:  # noqa: BLE001 - network layer, keep going
                last_err = f"{type(exc).__name__}: {exc}"
                time.sleep(1.5 * (attempt + 1))
        return {"ok": False, "error": last_err}

    def query(self, address: str) -> dict:
        key = str(address)
        with self._lock:
            if key in self.cache:
                return self.cache[key]
        result = self._request(key)
        with self._lock:
            self.cache[key] = result
            self._since_flush += 1
            if self._since_flush >= self.flush_every:
                self._save_locked()
        return result

    def query_many(self, addresses: Iterable[str], progress: bool = True) -> list[dict]:
        items = [str(a) for a in addresses]
        todo = [a for a in items if a not in self.cache]
        if todo:
            done = 0
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                for _ in ex.map(self.query, todo):
                    done += 1
                    if progress and done % 25 == 0:
                        rate = done / max(time.time() - t0, 1e-9)
                        print(
                            f"idesc {done}/{len(todo)} ({rate:.2f}/s, "
                            f"eta {(len(todo)-done)/max(rate,1e-9)/60:.1f} min)",
                            flush=True,
                        )
            self.save()
        return [self.cache.get(a, {"ok": False, "error": "missing"}) for a in items]

    # -- parsing ----------------------------------------------------------
    @staticmethod
    def flatten(result: dict) -> dict:
        """Flatten one cached result into scalar columns."""
        out = {
            "idesc_ok": False,
            "idesc_estado": None,
            "idesc_estado_class": None,
            "idesc_dir_ajusta": None,
            "idesc_direccion": None,
            "idesc_comuna": None,
            "idesc_barrio_codigo": None,
            "idesc_barrio_nombre": None,
            "idesc_estrato": None,
            "idesc_lat": None,
            "idesc_lon": None,
            "idesc_error": result.get("error") if isinstance(result, dict) else "bad_result",
        }
        if not isinstance(result, dict) or not result.get("ok"):
            return out
        payload = result.get("payload") or {}
        data = payload.get("data") or {}
        du = data.get("datosUbicacion") or {}
        if not du:
            out["idesc_error"] = "empty_datosUbicacion"
            return out
        out["idesc_ok"] = True
        out["idesc_error"] = None
        estado = du.get("estado")
        out["idesc_estado"] = estado
        if isinstance(estado, str) and estado:
            out["idesc_estado_class"] = estado.strip()[0].upper()
        dirs = du.get("direcciones") or {}
        out["idesc_dir_ajusta"] = dirs.get("dir_ajusta")
        out["idesc_direccion"] = dirs.get("direccion")
        out["idesc_comuna"] = du.get("comuna")
        out["idesc_estrato"] = du.get("estrato")
        barrio = du.get("barrio") or {}
        out["idesc_barrio_codigo"] = barrio.get("codigo")
        out["idesc_barrio_nombre"] = barrio.get("nombre")
        geo = du.get("geo_localizacion") or {}
        for key, dest in (("latitud", "idesc_lat"), ("longitud", "idesc_lon")):
            val = geo.get(key)
            try:
                out[dest] = float(val) if val not in (None, "", "None") else None
            except (TypeError, ValueError):
                out[dest] = None
        return out
