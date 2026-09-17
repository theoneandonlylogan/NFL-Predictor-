"""Stadium weather: nflverse historical fields plus Open-Meteo for upcoming games."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_predictor.config import (
    COLD_TEMP_F,
    DATA_RAW,
    HEAT_TEMP_F,
    RAIN_INCHES,
    SNOW_INCHES,
    TEAM_STADIUMS,
    WINDY_MPH,
)

logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _weather_cache_path() -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    return DATA_RAW / "weather_cache.json"


def _load_cache() -> dict[str, Any]:
    path = _weather_cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    _weather_cache_path().write_text(json.dumps(cache), encoding="utf-8")


def _parse_kickoff(row: pd.Series) -> datetime | None:
    gameday = row.get("gameday")
    if pd.isna(gameday):
        return None
    day = pd.to_datetime(gameday)
    gametime = row.get("gametime")
    if pd.isna(gametime) or str(gametime).strip() == "":
        return day.to_pydatetime().replace(hour=13)
    try:
        clock = datetime.strptime(str(gametime).strip()[:5], "%H:%M")
        return day.to_pydatetime().replace(hour=clock.hour, minute=clock.minute)
    except ValueError:
        return day.to_pydatetime().replace(hour=13)


def _is_indoor(row: pd.Series) -> bool:
    roof = str(row.get("roof") or "").lower()
    if roof in {"dome", "closed"}:
        return True
    home = str(row.get("home_team") or "")
    stadium = TEAM_STADIUMS.get(home, {})
    if stadium.get("roof") == "dome" and roof in {"", "nan", "none"}:
        return True
    return False


def _open_meteo(lat: float, lon: float, kickoff: datetime, tz: str, archive: bool) -> dict[str, float] | None:
    ahead = (kickoff.date() - datetime.now().date()).days
    if not archive and ahead < -1:
        archive = True
    if not archive and ahead > 16:
        return None

    param_sets = [
        {
            "hourly": "temperature_2m,precipitation,snowfall,wind_speed_10m",
            "wind_speed_unit": "mph",
        },
        {
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "wind_speed_unit": "mph",
        },
        {
            "hourly": "temperature_2m,precipitation,windspeed_10m",
            "windspeed_unit": "mph",
        },
    ]
    last_error: Exception | None = None
    for extra in param_sets:
        params = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "start_date": kickoff.strftime("%Y-%m-%d"),
            "end_date": kickoff.strftime("%Y-%m-%d"),
            "timezone": "auto",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            **extra,
        }
        base = ARCHIVE_URL if archive else FORECAST_URL
        url = f"{base}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            continue
        target = kickoff.strftime("%Y-%m-%dT%H:00")
        try:
            idx = times.index(target)
        except ValueError:
            idx = min(range(len(times)), key=lambda i: abs(i - kickoff.hour))

        def _at(key: str, fallback_key: str | None = None) -> float:
            values = hourly.get(key) or (hourly.get(fallback_key) if fallback_key else None) or []
            if not values or idx >= len(values) or values[idx] is None:
                return float("nan")
            return float(values[idx])

        return {
            "temp": _at("temperature_2m"),
            "precip": _at("precipitation"),
            "snowfall": _at("snowfall"),
            "wind": _at("wind_speed_10m", "windspeed_10m"),
        }
    if last_error is not None:
        logger.warning("Weather fetch failed: %s", last_error)
    return None


def _weather_from_text(text: str) -> dict[str, bool]:
    lowered = (text or "").lower()
    return {
        "is_snow": "snow" in lowered,
        "is_rain": any(token in lowered for token in ("rain", "shower", "drizzle")),
    }


def weather_flags(temp, wind, precip=0.0, snowfall=0.0, weather_text: str = "", indoor: bool = False) -> dict[str, float]:
    if indoor:
        return {
            "is_indoor": 1.0,
            "is_heat": 0.0,
            "is_cold": 0.0,
            "is_snow": 0.0,
            "is_rain": 0.0,
            "is_windy": 0.0,
        }
    text_flags = _weather_from_text(weather_text)
    temp_f = pd.to_numeric(pd.Series([temp]), errors="coerce").iloc[0]
    wind_mph = pd.to_numeric(pd.Series([wind]), errors="coerce").iloc[0]
    precip = pd.to_numeric(pd.Series([precip]), errors="coerce").iloc[0]
    snowfall = pd.to_numeric(pd.Series([snowfall]), errors="coerce").iloc[0]
    return {
        "is_indoor": 0.0,
        "is_heat": float(temp_f >= HEAT_TEMP_F) if pd.notna(temp_f) else 0.0,
        "is_cold": float(temp_f <= COLD_TEMP_F) if pd.notna(temp_f) else 0.0,
        "is_snow": float(bool(text_flags["is_snow"] or (pd.notna(snowfall) and snowfall >= SNOW_INCHES))),
        "is_rain": float(bool(text_flags["is_rain"] or (pd.notna(precip) and precip >= RAIN_INCHES and not (pd.notna(snowfall) and snowfall >= SNOW_INCHES)))),
        "is_windy": float(wind_mph >= WINDY_MPH) if pd.notna(wind_mph) else 0.0,
    }


def enrich_weather(schedules: pd.DataFrame, fetch_missing: bool = True) -> pd.DataFrame:
    """Fill temp/wind and weather bins. Forecasts upcoming outdoor games."""
    df = schedules.copy()
    if df.empty:
        return df

    cache = _load_cache()
    cache_dirty = False
    temps: list[float] = []
    winds: list[float] = []
    precips: list[float] = []
    snows: list[float] = []
    indoor_flags: list[bool] = []

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)

    for _, row in df.iterrows():
        indoor = _is_indoor(row)
        indoor_flags.append(indoor)
        temp = pd.to_numeric(pd.Series([row.get("temp")]), errors="coerce").iloc[0]
        wind = pd.to_numeric(pd.Series([row.get("wind")]), errors="coerce").iloc[0]
        precip = float("nan")
        snowfall = float("nan")
        gameday = pd.to_datetime(row.get("gameday"), errors="coerce")
        upcoming = pd.notna(gameday) and gameday >= now.normalize()

        # Historical weather comes from nflverse; only forecast upcoming outdoor games.
        needs_fetch = fetch_missing and upcoming and (pd.isna(temp) or pd.isna(wind)) and not indoor
        if needs_fetch:
            game_id = str(row.get("game_id") or f"{row.get('gameday')}-{row.get('home_team')}")
            if game_id in cache:
                cached = cache[game_id]
                temp = cached.get("temp", temp)
                wind = cached.get("wind", wind)
                precip = cached.get("precip", precip)
                snowfall = cached.get("snowfall", snowfall)
            else:
                home = str(row.get("home_team") or "")
                stadium = TEAM_STADIUMS.get(home)
                kickoff = _parse_kickoff(row)
                if stadium and kickoff is not None:
                    archive = pd.Timestamp(kickoff) < now - pd.Timedelta(days=2)
                    fetched = _open_meteo(stadium["lat"], stadium["lon"], kickoff, stadium["tz"], archive=archive)
                    if fetched:
                        temp = fetched["temp"] if pd.isna(temp) else temp
                        wind = fetched["wind"] if pd.isna(wind) else wind
                        precip = fetched["precip"]
                        snowfall = fetched["snowfall"]
                        cache[game_id] = {
                            "temp": temp,
                            "wind": wind,
                            "precip": precip,
                            "snowfall": snowfall,
                        }
                        cache_dirty = True

        temps.append(temp if pd.notna(temp) else float("nan"))
        winds.append(wind if pd.notna(wind) else float("nan"))
        precips.append(precip if pd.notna(precip) else 0.0)
        snows.append(snowfall if pd.notna(snowfall) else 0.0)

    if cache_dirty:
        _save_cache(cache)

    df["temp"] = temps
    df["wind"] = winds
    weather_text = df["weather"] if "weather" in df.columns else pd.Series([""] * len(df), index=df.index)

    flags = [
        weather_flags(t, w, p, s, str(txt or ""), indoor)
        for t, w, p, s, txt, indoor in zip(temps, winds, precips, snows, weather_text.tolist(), indoor_flags)
    ]
    flag_df = pd.DataFrame(flags, index=df.index)
    for col in flag_df.columns:
        df[col] = flag_df[col]
    df["precip"] = precips
    df["snowfall"] = snows
    return df
