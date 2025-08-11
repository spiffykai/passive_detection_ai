#!/usr/bin/env python3
"""
OpenSky Sanford Antenna-Direction Filter with Aircraft Metadata
---------------------------------------------------------------
Queries OpenSky's REST API for aircraft in a bounding box around a sensor,
filters by antenna azimuth/beam, and logs results to CSV with aircraft type info.

Features:
- config.json or CLI args (CLI overrides config)
- Robust OAuth2 authentication with token caching
- Auto-retry with exponential backoff
- Fetch aircraft metadata (model, manufacturer, registration, typecode)
- Caches metadata in memory + persistent JSON cache
- One CSV per run or loop iteration
- Prints summary after each run

Default config.json (Sanford test):
{
  "auth": {
    "mode": "auto",
    "client_id": "YOUR_CLIENT_ID",
    "client_secret": "YOUR_CLIENT_SECRET",
    "username": "YOUR_USERNAME",
    "password": "YOUR_PASSWORD",
    "token_cache_file": ".opensky_token.json"
  },
  "api": {
    "base_url": "https://opensky-network.org/api",
    "timeout_seconds": 30
  },
  "retry": {
    "total": 5,
    "backoff_factor": 1.0,
    "status_forcelist": [429, 500, 502, 503, 504]
  },
  "sensor": {
    "lat": 28.7775,
    "lon": -81.3070,
    "azimuth": 90,
    "beam_width": 360,
    "radius_km": 15
  }
}
"""

from __future__ import annotations

import os
import math
import json
import csv
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CACHE_FILE = "aircraft_cache.json"

DEFAULT_API_BASE = "https://opensky-network.org/api"
DEFAULT_AUTH_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
DEFAULT_TIMEOUT = 30
DEFAULT_STATUS_FORCELIST = (429, 500, 502, 503, 504)


# ----------------------------- Robust OpenSky Client -----------------------------

@dataclass
class RetryConfig:
    total: int = 5
    backoff_factor: float = 1.0
    status_forcelist: Tuple[int, ...] = DEFAULT_STATUS_FORCELIST
    raise_on_status: bool = False


@dataclass
class APIConfig:
    base_url: str = DEFAULT_API_BASE
    timeout_seconds: int = DEFAULT_TIMEOUT


@dataclass
class AuthConfig:
    # mode: "oauth" (prefer), "basic" (always), or "auto" (try oauth, fallback to basic)
    mode: str = "oauth"
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    token_url: str = DEFAULT_AUTH_URL
    token_cache_file: Optional[str] = ".opensky_token.json"  # set to None to disable on-disk cache


@dataclass
class ClientConfig:
    auth: AuthConfig
    api: APIConfig = APIConfig()
    retry: RetryConfig = RetryConfig()


class OpenSkyClient:
    def __init__(self, config: ClientConfig):
        self.config = config
        self.session = self._build_session(config.retry)
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0
        # Attempt to load cached token on init (if enabled)
        self._load_cached_token()

    @classmethod
    def from_file(cls, path: str) -> "OpenSkyClient":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        auth = AuthConfig(**raw.get("auth", {}))
        api = APIConfig(**raw.get("api", {}))
        retry = RetryConfig(**raw.get("retry", {}))

        # Normalize/validate mode
        auth.mode = (auth.mode or "oauth").lower()
        if auth.mode not in ("oauth", "basic", "auto"):
            raise ValueError("auth.mode must be 'oauth', 'basic', or 'auto'")

        return cls(ClientConfig(auth=auth, api=api, retry=retry))

    def _build_session(self, rc: RetryConfig) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=rc.total,
            connect=rc.total,
            read=rc.total,
            status=rc.total,
            backoff_factor=rc.backoff_factor,
            status_forcelist=rc.status_forcelist,
            allowed_methods=("GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"),
            raise_on_status=rc.raise_on_status,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"Accept": "application/json"})
        return session

    def _load_cached_token(self):
        cache = self.config.auth.token_cache_file
        if not cache:
            return
        try:
            if os.path.isfile(cache):
                with open(cache, "r", encoding="utf-8") as f:
                    data = json.load(f)
                token = data.get("access_token")
                expiry = data.get("expires_at", 0)
                # Only trust cache if we still have runway
                if token and expiry and expiry - time.time() > 60:
                    self._token = token
                    self._token_expiry = float(expiry)
        except Exception:
            pass

    def _save_cached_token(self):
        cache = self.config.auth.token_cache_file
        if not cache or not self._token:
            return
        try:
            tmp = cache + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"access_token": self._token, "expires_at": self._token_expiry}, f)
            try:
                os.chmod(tmp, 0o600)
            except Exception:
                pass
            os.replace(tmp, cache)
        except Exception:
            pass

    def _oauth_get_token(self) -> Tuple[str, float]:
        a = self.config.auth
        if not a.client_id or not a.client_secret:
            raise ValueError("OAuth mode requires client_id and client_secret in config.auth")

        # OpenSky expects credentials in the request body, not Basic auth
        data = {
            "grant_type": "client_credentials",
            "client_id": a.client_id,
            "client_secret": a.client_secret
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        resp = self.session.post(
            a.token_url,
            data=data,
            headers=headers,
            timeout=self.config.api.timeout_seconds,
        )
        resp.raise_for_status()
        payload = resp.json()
        token = payload.get("access_token")
        expires_in = float(payload.get("expires_in", 3600))
        if not token:
            raise RuntimeError("Token endpoint did not return access_token")
        # Leave 60s safety margin
        return token, time.time() + max(300.0, (expires_in - 60.0))

    def _ensure_token(self):
        if self._token and time.time() < self._token_expiry:
            return
        token, expiry = self._oauth_get_token()
        self._token = token
        self._token_expiry = expiry
        self._save_cached_token()

    def _apply_auth(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        mode = self.config.auth.mode
        if mode == "basic":
            return self._apply_basic(kwargs)
        if mode == "oauth":
            return self._apply_oauth(kwargs)
        return self._apply_oauth(kwargs)

    def _apply_oauth(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_token()
        headers = dict(kwargs.get("headers") or {})
        headers["Authorization"] = f"Bearer {self._token}"
        kwargs["headers"] = headers
        return kwargs

    def _apply_basic(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        a = self.config.auth
        if not a.username or not a.password:
            raise ValueError("Basic auth mode requires username and password in config.auth")
        kwargs["auth"] = (a.username, a.password)
        return kwargs

    def request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None,
                json_body: Optional[Dict[str, Any]] = None, **kwargs) -> requests.Response:
        url = self._join(self.config.api.base_url, path)
        timeout = kwargs.pop("timeout", self.config.api.timeout_seconds)

        try:
            prepared_kwargs = dict(kwargs)
            prepared_kwargs = self._apply_auth(prepared_kwargs)
            resp = self.session.request(
                method.upper(), url, params=params, json=json_body, timeout=timeout, **prepared_kwargs
            )
            # If token expired or invalid, refresh once and retry
            if resp.status_code == 401 and self.config.auth.mode in ("oauth", "auto"):
                self._token = None
                self._token_expiry = 0.0
                prepared_kwargs = dict(kwargs)
                prepared_kwargs = self._apply_oauth(prepared_kwargs)
                resp = self.session.request(
                    method.upper(), url, params=params, json=json_body, timeout=timeout, **prepared_kwargs
                )
            # If still unauthorized and in auto mode, fall back to basic (if provided)
            if resp.status_code == 401 and self.config.auth.mode == "auto" and self.config.auth.username and self.config.auth.password:
                prepared_kwargs = dict(kwargs)
                prepared_kwargs = self._apply_basic(prepared_kwargs)
                resp = self.session.request(
                    method.upper(), url, params=params, json=json_body, timeout=timeout, **prepared_kwargs
                )
            resp.raise_for_status()
            return resp
        except requests.exceptions.ReadTimeout as e:
            raise requests.exceptions.ReadTimeout(
                f"Read timed out after {timeout}s calling {url}"
            ) from e

    def get(self, path: str, *, params: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        resp = self.request("GET", path, params=params, **kwargs)
        return self._maybe_json(resp)

    def post(self, path: str, *, json_body: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        resp = self.request("POST", path, json_body=json_body, **kwargs)
        return self._maybe_json(resp)

    @staticmethod
    def _join(base: str, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        if not base.endswith("/"):
            base += "/"
        return base + path.lstrip("/")

    @staticmethod
    def _maybe_json(resp: requests.Response) -> Dict[str, Any]:
        if "application/json" in resp.headers.get("Content-Type", ""):
            return resp.json()
        return {"status_code": resp.status_code, "text": resp.text}


# ----------------------------- Geometry utils -----------------------------

def deg_box_from_radius_km(lat: float, radius_km: float) -> Tuple[float, float]:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 1e-6))
    return dlat, dlon


def calculate_bearing(lat1, lon1, lat2, lon2) -> float:
    dlon = math.radians(lon2 - lon1)
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def calculate_ground_distance(lat1, lon1, lat2, lon2) -> float:
    """
    Calculate ground distance between two points using Haversine formula.
    Returns distance in kilometers.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers
    r = 6371
    return c * r


def calculate_3d_distance(lat1, lon1, alt1, lat2, lon2, alt2) -> float:
    """
    Calculate 3D distance between two points including altitude.
    Altitudes should be in meters, returns distance in kilometers.
    """
    ground_dist = calculate_ground_distance(lat1, lon1, lat2, lon2)
    if ground_dist is None or alt1 is None or alt2 is None:
        return None

    # Convert altitude difference to kilometers
    alt_diff_km = abs(alt2 - alt1) / 1000.0

    # Convert ground distance to meters for calculation
    ground_dist_km = ground_dist

    # Pythagorean theorem in 3D
    return math.sqrt(ground_dist_km**2 + alt_diff_km**2)


def is_within_beam(sensor_lat, sensor_lon, target_lat, target_lon, azimuth, beam_width) -> bool:
    if target_lat is None or target_lon is None:
        return False
    bearing = calculate_bearing(sensor_lat, sensor_lon, target_lat, target_lon)
    diff = (bearing - azimuth + 360) % 360
    return diff <= beam_width / 2 or diff >= 360 - beam_width / 2


# ----------------------------- API Calls using robust client -----------------------------

def get_aircraft_metadata(icao24: str, client: OpenSkyClient, cache: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Look up aircraft metadata, using in-memory + persistent cache.
    """
    if icao24 in cache:
        return cache[icao24]

    try:
        data = client.get(f"/metadata/aircraft/icao/{icao24}")
        result = {
            "registration": data.get("registration"),
            "manufacturericao": data.get("manufacturericao"),
            "model": data.get("model"),
            "typecode": data.get("typecode")
        }
    except Exception:
        result = {
            "registration": None,
            "manufacturericao": None,
            "model": None,
            "typecode": None
        }

    cache[icao24] = result
    return result


# ----------------------------- Filtering -----------------------------

def filter_in_beam(data: Dict[str, Any], sensor_lat: float, sensor_lon: float,
                   azimuth: float, beam_width: float,
                   min_alt: Optional[float], max_alt: Optional[float]) -> List[Dict[str, Any]]:
    states = data.get("states", []) or []
    # Get the overall timestamp for this data snapshot
    data_timestamp = data.get("time")

    results = []
    for s in states:
        callsign = (s[1] or "").strip() or "N/A"
        lon = s[5]
        lat = s[6]
        alt = s[7]
        # Individual aircraft timestamps
        time_position = s[3]  # Unix timestamp of last position update
        last_contact = s[4]   # Unix timestamp of last contact

        if not is_within_beam(sensor_lat, sensor_lon, lat, lon, azimuth, beam_width):
            continue
        if min_alt is not None and (alt is None or alt < min_alt):
            continue
        if max_alt is not None and (alt is not None and alt > max_alt):
            continue
        bearing = calculate_bearing(sensor_lat, sensor_lon, lat, lon) if lat and lon else None
        ground_dist_km = calculate_ground_distance(sensor_lat, sensor_lon, lat, lon) if lat and lon else None
        # For 3D distance, assume sensor is at ground level (0m altitude)
        distance_3d_km = calculate_3d_distance(sensor_lat, sensor_lon, 0, lat, lon, alt) if lat and lon and alt is not None else None

        results.append({
            "icao24": s[0],
            "callsign": callsign,
            "lat": lat,
            "lon": lon,
            "baro_alt_m": alt,
            "bearing_deg": bearing,
            "ground_distance_km": ground_dist_km,
            "distance_3d_km": distance_3d_km,
            "velocity_ms": s[9],
            "heading_deg": s[10],
            "data_timestamp": data_timestamp,
            "time_position": time_position,
            "last_contact": last_contact,
        })
    return results


# ----------------------------- Config & CLI -----------------------------

def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_settings(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Handle legacy and new config formats
    auth_cfg = cfg.get("auth", {})
    client_id = args.client_id or cfg.get("client_id") or auth_cfg.get("client_id")
    client_secret = args.client_secret or cfg.get("client_secret") or auth_cfg.get("client_secret")
    username = getattr(args, 'username', None) or cfg.get("username") or auth_cfg.get("username")
    password = getattr(args, 'password', None) or cfg.get("password") or auth_cfg.get("password")
    token = args.token or os.getenv("OPENSKY_TOKEN") or cfg.get("bearer_token")

    # Support new "sensor" section and legacy format
    sensor_cfg = cfg.get("sensor", {})
    lat = args.lat if args.lat is not None else sensor_cfg.get("lat") or cfg.get("sensor_lat")
    lon = args.lon if args.lon is not None else sensor_cfg.get("lon") or cfg.get("sensor_lon")
    azimuth = args.azimuth if args.azimuth is not None else sensor_cfg.get("azimuth") or cfg.get("antenna_azimuth")
    beam = args.beam if args.beam is not None else sensor_cfg.get("beam_width") or cfg.get("beam_width")

    radius_km = args.radius_km if args.radius_km is not None else sensor_cfg.get("radius_km") or cfg.get("radius_km", 25.0)
    min_alt = args.min_alt if args.min_alt is not None else sensor_cfg.get("min_alt") or cfg.get("min_alt")
    max_alt = args.max_alt if args.max_alt is not None else sensor_cfg.get("max_alt") or cfg.get("max_alt")

    missing = []
    if not token and not (client_id and client_secret) and not (username and password):
        missing.append("token OR client_id+client_secret OR username+password")
    for key, value in [("lat", lat), ("lon", lon), ("azimuth", azimuth), ("beam", beam)]:
        if value is None:
            missing.append(key)
    if missing:
        raise SystemExit("Missing: " + ", ".join(missing))

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "username": username,
        "password": password,
        "token": token,
        "lat": float(lat),
        "lon": float(lon),
        "azimuth": float(azimuth) % 360.0,
        "beam": float(beam),
        "radius_km": float(radius_km),
        "min_alt": None if min_alt is None else float(min_alt),
        "max_alt": None if max_alt is None else float(max_alt),
    }


def create_client_from_settings(settings: Dict[str, Any]) -> OpenSkyClient:
    """Create OpenSky client from resolved settings"""
    if settings.get("token"):
        # If we have a direct token, create a minimal config
        auth_config = AuthConfig(
            mode="oauth",
            client_id="dummy",  # Required but not used with direct token
            client_secret="dummy",
            token_cache_file=None  # Don't cache direct tokens
        )
        client_config = ClientConfig(auth=auth_config)
        client = OpenSkyClient(client_config)
        client._token = settings["token"]
        client._token_expiry = time.time() + 3600  # Assume 1 hour validity
        return client
    else:
        # Use OAuth with client_id/secret and fallback to basic auth
        auth_mode = "auto" if (settings.get("username") and settings.get("password")) else "oauth"
        auth_config = AuthConfig(
            mode=auth_mode,
            client_id=settings.get("client_id"),
            client_secret=settings.get("client_secret"),
            username=settings.get("username"),
            password=settings.get("password")
        )
        client_config = ClientConfig(auth=auth_config)
        return OpenSkyClient(client_config)


def print_examples():
    msg = r"""
Examples:

  # Using CLI args only (token from env var)
  export OPENSKY_TOKEN="YOUR_BEARER_TOKEN"
  python adbs_client.py --lat 28.7775 --lon -81.3070 --azimuth 90 --beam 180 --radius-km 15

  # Using client_id/secret from config.json
  python adbs_client.py --config config.json

  # Loop every 60 seconds
  python adbs_client.py --config config.json --loop 60

Updated config.json format (with robust client features):
{
  "auth": {
    "mode": "auto",
    "client_id": "YOUR_CLIENT_ID",
    "client_secret": "YOUR_CLIENT_SECRET",
    "username": "YOUR_USERNAME",
    "password": "YOUR_PASSWORD",
    "token_cache_file": ".opensky_token.json"
  },
  "api": {
    "timeout_seconds": 30
  },
  "retry": {
    "total": 5,
    "backoff_factor": 1.0
  },
  "sensor": {
    "lat": 28.7775,
    "lon": -81.3070,
    "azimuth": 90,
    "beam_width": 360,
    "radius_km": 15
  }
}
"""
    print(msg.strip())


# ----------------------------- Main -----------------------------

def main():
    p = argparse.ArgumentParser(description="OpenSky Sanford antenna-direction filter with metadata")
    p.add_argument("--config", help="Path to config.json")
    p.add_argument("--examples", action="store_true", help="Show usage examples and exit")
    p.add_argument("--token", help="Bearer token")
    p.add_argument("--client-id", dest="client_id", help="OpenSky client_id")
    p.add_argument("--client-secret", dest="client_secret", help="OpenSky client_secret")
    p.add_argument("--username", help="OpenSky username (for basic auth fallback)")
    p.add_argument("--password", help="OpenSky password (for basic auth fallback)")
    p.add_argument("--lat", type=float, help="Sensor latitude")
    p.add_argument("--lon", type=float, help="Sensor longitude")
    p.add_argument("--azimuth", type=float, help="Antenna azimuth degrees")
    p.add_argument("--beam", type=float, help="Beam width degrees")
    p.add_argument("--radius-km", type=float, help="Search radius km")
    p.add_argument("--min-alt", type=float, help="Min altitude m")
    p.add_argument("--max-alt", type=float, help="Max altitude m")
    p.add_argument("--loop", type=int, default=0, help="Loop every N seconds (0 = run once)")
    args = p.parse_args()

    if args.examples:
        print_examples()
        return

    cfg = load_config(args.config)
    settings = resolve_settings(args, cfg)

    # Create robust client
    try:
        if args.config and "auth" in cfg:
            # Use robust client with full config
            client = OpenSkyClient.from_file(args.config)
        else:
            # Create client from resolved settings
            client = create_client_from_settings(settings)
    except Exception as e:
        print(f"Failed to create OpenSky client: {e}")
        return

    os.makedirs("logs", exist_ok=True)

    # Load persistent cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            metadata_cache = json.load(f)
    else:
        metadata_cache = {}

    def save_cache():
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata_cache, f)

    def run_once():
        dlat, dlon = deg_box_from_radius_km(settings["lat"], settings["radius_km"])
        lamin, lamax = settings["lat"] - dlat, settings["lat"] + dlat
        lomin, lomax = settings["lon"] - dlon, settings["lon"] + dlon

        try:
            data = client.get("/states/all", params={
                "lamin": lamin, "lamax": lamax, "lomin": lomin, "lomax": lomax
            })
        except Exception as e:
            print(f"Error querying OpenSky states: {e}")
            return

        filtered = filter_in_beam(data, settings["lat"], settings["lon"],
                                  settings["azimuth"], settings["beam"],
                                  settings["min_alt"], settings["max_alt"])

        # Add metadata
        for ac in filtered:
            meta = get_aircraft_metadata(ac["icao24"], client, metadata_cache)
            ac.update(meta)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_name = f"logs/opensky_results_{ts}_az{int(settings['azimuth'])}_beam{int(settings['beam'])}.csv"

        with open(csv_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "icao24", "callsign", "lat", "lon", "baro_alt_m",
                "bearing_deg", "ground_distance_km", "distance_3d_km", "velocity_ms", "heading_deg",
                "data_timestamp", "time_position", "last_contact",
                "registration", "manufacturericao", "model", "typecode"
            ])
            writer.writeheader()
            for row in filtered:
                writer.writerow(row)

        save_cache()

        total_states = len(data.get("states", []) or [])
        print(f"Found {total_states} total aircraft; {len(filtered)} within antenna beam.")
        print(f"Saved results to: {csv_name}\n")

    if args.loop > 0:
        while True:
            try:
                run_once()
            except KeyboardInterrupt:
                print("\nStopped by user")
                break
            except Exception as e:
                print(f"Error in run loop: {e}")
            time.sleep(args.loop)
    else:
        run_once()


if __name__ == "__main__":
    main()
