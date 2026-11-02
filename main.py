"""
EWS Risk Service — USGS-backed, drop-in replacement for EWSStub.

The original EWSStub returned deterministic hash-based scores. This version
sources real earthquake data from the USGS FDSN Web Service
(`https://earthquake.usgs.gov/fdsnws/event/1/`) and derives a risk score from
the recent seismicity around the queried point. The contract is identical to
EWSStub so the RiskAndHazardML inference client needs no changes.

Contract (mirrors `inference-contracts/output-schema.md`):
    GET /api/v1/risk/point?lat=<float>&lon=<float>
    -> 200 { "data": {
            "composite": float in [0, 1],
            "confidence": float in [0, 1],
            "exposure": float in [0, 1],
            "vulnerability": float in [0, 1],
            "kecamatan_code": str | null,
            "assessed_at": iso8601,
            "notes": list[str]
        } }

Scoring (intentionally simple; this docstring is the reference):
    - Pull the 200 largest M>=2.5 earthquakes within 300 km of the point,
      from 00:00 UTC 30 days ago (so a 30-31 day window). USGS FDSN keeps
      this fast and free.
    - composite = clamp(0.6 * mag_factor + 0.4 * proximity_factor, 0, 1)
        mag_factor      = min(1, max_magnitude / 8.0)
        proximity_factor = min(1, 1 / (1 + min_distance_km / 50))
    - exposure = clamp(event_count / 30, 0, 1)
    - vulnerability = 0.5 (placeholder; can be replaced with BNPB InaRisk data
      later when the proper layer id is identified)
    - confidence = 0.4 if no events, else min(1, 0.3 + 0.07 * event_count),
      which reaches 1.0 at 10 events. One event gives 0.37, slightly below
      the no-event value.
    - kecamatan_code: deterministic pseudo-id from (lat, lon) until the BNPB
      boundary layer is wired. Carries a "USGS" prefix so downstream code can
      tell it isn't a real administrative code.

Upstream handling:
    - USGS results are cached per point for 60 s (bounded at 1024 points).
    - Timeouts, network errors and 5xx are retried once within a 12 s budget.
    - Any failed or malformed USGS response returns 502, never a zero score.
"""
from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="EWS Risk (USGS)", version="0.2.0-usgs")
log = logging.getLogger("ews_usgs")

USGS_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"
WINDOW_DAYS = 30
RADIUS_KM = 300.0
MIN_MAGNITUDE = 2.5
QUERY_LIMIT = 200
HTTP_TIMEOUT_S = 8.0
# The gateway waits 15 s for this service (Ews:TimeoutSeconds), so all USGS
# attempts plus backoff must fit in this budget.
HTTP_BUDGET_S = 12.0
HTTP_ATTEMPTS = 2
RETRY_BACKOFF_S = 1.0
# 429 is deliberately not retried: USGS is asking clients to slow down.
RETRY_STATUS = frozenset({500, 502, 503, 504})
# Identical points are answered from memory for this long, so a dashboard that
# polls one point does not turn into one USGS query per request.
CACHE_TTL_S = 60.0
CACHE_MAX_ENTRIES = 1024

_started_at = time.time()
# One pooled client per process instead of a new TLS connection per request.
_client = httpx.Client(
    timeout=HTTP_TIMEOUT_S,
    headers={"User-Agent": f"AI-MINT-EWSUsgs/{app.version}"},
)
_cache: dict[tuple[float, float], tuple[float, list[dict]]] = {}
_cache_lock = threading.Lock()


def _cache_get(key: tuple[float, float]) -> Optional[list[dict]]:  # noqa: UP045 - Qodana's default Python level rejects X | None
    with _cache_lock:
        hit = _cache.get(key)
    if hit is None or time.monotonic() - hit[0] >= CACHE_TTL_S:
        return None
    return hit[1]


def _cache_put(key: tuple[float, float], events: list[dict]) -> None:
    now = time.monotonic()
    with _cache_lock:
        _cache.pop(key, None)
        if len(_cache) >= CACHE_MAX_ENTRIES:
            # Expired entries first, then the oldest, so a caller cycling
            # through distinct points cannot grow memory without bound.
            expired = [k for k, (t, _) in _cache.items() if now - t >= CACHE_TTL_S]
            for k in expired:
                del _cache[k]
            while len(_cache) >= CACHE_MAX_ENTRIES:
                del _cache[next(iter(_cache))]
        _cache[key] = (now, events)


def _hash_id(*nums: float, prefix: str = "USGS") -> str:
    """Stable pseudo-id for kecamatan_code when no boundary lookup is wired."""
    h = hashlib.sha256("|".join(f"{n:.4f}" for n in nums).encode()).hexdigest()
    return prefix + h[:10].upper()


def _score(events: list[dict], lat: float, lon: float) -> dict:
    if not events:
        return {
            "composite": 0.0,
            "confidence": 0.4,
            "exposure": 0.0,
            "vulnerability": 0.5,
            "kecamatan_code": _hash_id(lat, lon),
            "assessed_at": datetime.now(UTC).isoformat(),
            "notes": [
                (
                    f"no recent M>={MIN_MAGNITUDE} earthquakes within "
                    f"{RADIUS_KM:.0f} km / {WINDOW_DAYS} days"
                ),
                f"lat={lat:.4f} lon={lon:.4f}",
            ],
        }

    max_mag = float(max(e["magnitude"] for e in events))
    min_dist = float(min(e["distance_km"] for e in events))

    mag_factor = min(1.0, max_mag / 8.0)
    proximity_factor = 1.0 / (1.0 + min_dist / 50.0)
    composite = max(0.0, min(1.0, 0.6 * mag_factor + 0.4 * proximity_factor))

    exposure = min(1.0, len(events) / 30.0)

    # Confidence: scale with sample size; fully confident from 10 events.
    confidence = min(1.0, 0.3 + 0.07 * len(events))

    notes = [
        (
            f"USGS FDSN: {len(events)} events, max M{max_mag:.1f}, "
            f"nearest {min_dist:.0f} km"
        ),
        f"lat={lat:.4f} lon={lon:.4f}",
    ]
    if len(events) >= QUERY_LIMIT:
        # USGS returns the largest events first, so in a dense sequence the
        # nearest small event can fall outside the page.
        notes.append(
            f"USGS query limit of {QUERY_LIMIT} events reached; "
            "nearest distance may be overstated"
        )

    return {
        "composite": round(composite, 6),
        "confidence": round(confidence, 4),
        "exposure": round(exposure, 4),
        "vulnerability": 0.5,
        "kecamatan_code": _hash_id(lat, lon),
        "assessed_at": datetime.now(UTC).isoformat(),
        "notes": notes,
    }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _get_usgs_json(params: dict[str, str | int | float]) -> object:
    """GET the FDSN query; one retry with backoff on timeouts, network errors
    and 5xx, as long as a useful attempt still fits in HTTP_BUDGET_S."""
    deadline = time.monotonic() + HTTP_BUDGET_S
    attempt = 1
    while True:
        try:
            r = _client.get(
                USGS_BASE,
                params=params,
                timeout=min(HTTP_TIMEOUT_S, deadline - time.monotonic()),
            )
            r.raise_for_status()
            return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            transient = isinstance(e, httpx.TransportError) or (
                e.response.status_code in RETRY_STATUS
            )
            backoff = RETRY_BACKOFF_S * attempt
            if (
                not transient
                or attempt >= HTTP_ATTEMPTS
                or deadline - time.monotonic() < backoff + 2.0
            ):
                raise
            log.warning("USGS attempt %d failed (%r), retrying", attempt, e)
            time.sleep(backoff)
            attempt += 1


def _fetch_usgs(lat: float, lon: float) -> list[dict]:
    """Returns list of {magnitude, distance_km, place, time} dicts.

    Results are cached per point (rounded to 4 decimals, ~11 m) for
    CACHE_TTL_S; errors are not cached.
    """
    key = (round(lat, 4), round(lon, 4))
    cached = _cache_get(key)
    if cached is not None:
        return cached
    start = (datetime.now(UTC) - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    params: dict[str, str | int | float] = {
        "format": "geojson",
        "starttime": start,
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": RADIUS_KM,
        "minmagnitude": MIN_MAGNITUDE,
        "limit": QUERY_LIMIT,
        "orderby": "magnitude",
    }
    try:
        payload = _get_usgs_json(params)
    except httpx.HTTPStatusError as e:
        # The gateway forwards our error body verbatim, so keep the upstream
        # URL and exception text in our log rather than in the response.
        log.warning("USGS returned HTTP %s", e.response.status_code)
        raise HTTPException(
            status_code=502,
            detail=f"USGS upstream error: HTTP {e.response.status_code}",
        ) from e
    except (httpx.HTTPError, ValueError) as e:
        log.warning("USGS request failed: %r", e)
        raise HTTPException(
            status_code=502, detail=f"USGS upstream error: {type(e).__name__}"
        ) from e

    # A body without a "features" list is not "no earthquakes": scoring it as
    # an empty list would publish a false all-clear (composite 0.0).
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        log.warning("USGS response is not a GeoJSON FeatureCollection")
        raise HTTPException(
            status_code=502, detail="USGS upstream error: malformed GeoJSON"
        )
    out: list[dict] = []
    for f in features:
        try:
            mag = float(f["properties"]["mag"])
            # GeoJSON order is [lon, lat, depth_km]; depth is not scored.
            coords = f["geometry"]["coordinates"]
            lon_eq, lat_eq = float(coords[0]), float(coords[1])
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        if not all(math.isfinite(v) for v in (mag, lat_eq, lon_eq)):
            continue
        out.append({
            "magnitude": mag,
            "distance_km": _haversine_km(lat, lon, lat_eq, lon_eq),
            "place": f["properties"].get("place", ""),
            "time": f["properties"].get("time"),
        })
    _cache_put(key, out)
    return out


@app.get("/api/v1/risk/point")
def risk_point(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
) -> dict:
    if math.isnan(lat) or math.isnan(lon) or math.isinf(lat) or math.isinf(lon):
        raise HTTPException(status_code=400, detail="lat/lon must be finite")
    events = _fetch_usgs(lat, lon)
    return {"data": _score(events, lat, lon)}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "ews-usgs",
        "version": app.version,
        "uptime_seconds": int(time.time() - _started_at),
    }
