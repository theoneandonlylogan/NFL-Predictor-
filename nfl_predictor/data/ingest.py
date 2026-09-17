"""Download and cache nflverse schedules, play-by-play aggregates, injuries, and snaps."""

from __future__ import annotations

import logging
import time
import urllib.request
from pathlib import Path

import pandas as pd

from nfl_predictor.config import DATA_RAW, LIVE_CACHE_HOURS, PBP_COLUMNS, current_nfl_season

logger = logging.getLogger(__name__)

NFLVERSE_RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
USER_AGENT = "nfl-predictor/1.0 (https://github.com/theoneandonlylogan/NFL-Predictor-)"


def _ensure_raw_dir() -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    return DATA_RAW


def _cache_path(name: str, year: int) -> Path:
    return _ensure_raw_dir() / f"{name}_{year}.parquet"


def _is_live_year(year: int) -> bool:
    return year >= current_nfl_season()


def _cache_fresh(path: Path, year: int) -> bool:
    if not path.exists():
        return False
    if not _is_live_year(year):
        return True
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < LIVE_CACHE_HOURS


def _read_or_fetch(name: str, year: int, fetch, refresh: bool = False) -> pd.DataFrame:
    path = _cache_path(name, year)
    if not refresh and _cache_fresh(path, year):
        return pd.read_parquet(path)
    logger.info("Fetching %s %s", name, year)
    df = fetch(year)
    if df is None:
        df = pd.DataFrame()
    if df.empty and not _is_live_year(year):
        logger.warning("No %s data for %s; not caching empty historical file", name, year)
        return df
    df.to_parquet(path, index=False)
    return df


def _concat_years(name: str, years: list[int], fetch, refresh: bool = False) -> pd.DataFrame:
    frames = [_read_or_fetch(name, year, fetch, refresh=refresh) for year in years]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _download_parquet(url: str) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
    tmp = _ensure_raw_dir() / f"_download_{url.split('/')[-1]}.tmp"
    tmp.write_bytes(data)
    try:
        return pd.read_parquet(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def _nflverse_file(release: str, filename: str) -> pd.DataFrame:
    url = f"{NFLVERSE_RELEASE}/{release}/{filename}"
    print(f"Downloading {filename} ...")
    logger.info("GET %s", url)
    return _download_parquet(url)


def _cached_asset(cache_name: str, release: str, filename: str, refresh: bool, live: bool) -> pd.DataFrame:
    path = _ensure_raw_dir() / f"{cache_name}.parquet"
    if not refresh and path.exists():
        if not live:
            return pd.read_parquet(path)
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours < LIVE_CACHE_HOURS:
            return pd.read_parquet(path)
    df = _nflverse_file(release, filename)
    if df.empty:
        if path.exists():
            return pd.read_parquet(path)
        return df
    df.to_parquet(path, index=False)
    return df


def load_schedules(years: list[int], refresh: bool = False) -> pd.DataFrame:
    live = any(_is_live_year(year) for year in years)
    df = _cached_asset("schedules_games", "schedules", "games.parquet", refresh=refresh, live=live)
    if df.empty:
        return df
    if "season" in df.columns:
        df = df[df["season"].isin(years)].copy()
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
    return df


def _aggregate_team_game_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    if pbp.empty:
        return pd.DataFrame()

    plays = pbp.copy()
    for col, default in (("epa", 0.0), ("success", 0.0), ("yards_gained", 0.0)):
        if col not in plays.columns:
            plays[col] = default
    if "pass" not in plays.columns:
        plays["pass"] = (plays.get("play_type") == "pass").astype(int)
    if "rush" not in plays.columns:
        plays["rush"] = plays.get("play_type").isin(["run", "rush"]).astype(int)

    plays["pass"] = pd.to_numeric(plays["pass"], errors="coerce").fillna(0).astype(int)
    plays["rush"] = pd.to_numeric(plays["rush"], errors="coerce").fillna(0).astype(int)
    plays = plays[(plays["pass"] == 1) | (plays["rush"] == 1)]
    plays = plays[plays["posteam"].notna() & plays["defteam"].notna()]
    if plays.empty:
        return pd.DataFrame()

    pass_plays = plays[plays["pass"] == 1]
    rush_plays = plays[plays["rush"] == 1]

    def _side(df: pd.DataFrame, team_col: str, prefix: str, play_kind: str | None = None) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        grouped = df.groupby(["game_id", "season", "week", team_col], dropna=True)
        out = grouped.agg(
            plays=("epa", "count"),
            epa=("epa", "mean"),
            success=("success", "mean"),
            yards=("yards_gained", "mean"),
        ).reset_index()
        if play_kind:
            rename = {
                team_col: "team",
                "plays": f"{prefix}_{play_kind}_plays",
                "epa": f"{prefix}_{play_kind}_epa",
                "success": f"{prefix}_{play_kind}_success",
                "yards": f"{prefix}_{play_kind}_yards",
            }
        else:
            rename = {
                team_col: "team",
                "plays": f"{prefix}_plays",
                "epa": f"{prefix}_epa",
                "success": f"{prefix}_success",
                "yards": f"{prefix}_yards",
            }
        return out.rename(columns=rename)

    off = _side(plays, "posteam", "off")
    keys = ["game_id", "season", "week", "team"]
    out = off
    for extra in (
        _side(pass_plays, "posteam", "off", "pass"),
        _side(rush_plays, "posteam", "off", "rush"),
        _side(plays, "defteam", "def"),
        _side(pass_plays, "defteam", "def", "pass"),
        _side(rush_plays, "defteam", "def", "rush"),
    ):
        if extra.empty:
            continue
        out = out.merge(extra, on=keys, how="outer")
    return out


def load_team_game_pbp(years: list[int], refresh: bool = False) -> pd.DataFrame:
    def fetch(year: int) -> pd.DataFrame:
        try:
            pbp = _nflverse_file("pbp", f"play_by_play_{year}.parquet")
            cols = [c for c in PBP_COLUMNS if c in pbp.columns]
            if cols:
                pbp = pbp[cols]
            return _aggregate_team_game_pbp(pbp)
        except Exception as exc:  # noqa: BLE001 - in-progress seasons can 404
            logger.warning("Play-by-play unavailable for %s: %s", year, exc)
            return pd.DataFrame()

    return _concat_years("pbp_team_game", years, fetch, refresh=refresh)


def load_injuries(years: list[int], refresh: bool = False) -> pd.DataFrame:
    def fetch(year: int) -> pd.DataFrame:
        try:
            return _nflverse_file("injuries", f"injuries_{year}.parquet")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Injury data unavailable for %s: %s", year, exc)
            return pd.DataFrame()

    return _concat_years("injuries", years, fetch, refresh=refresh)


def load_snap_counts(years: list[int], refresh: bool = False) -> pd.DataFrame:
    def fetch(year: int) -> pd.DataFrame:
        try:
            return _nflverse_file("snap_counts", f"snap_counts_{year}.parquet")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Snap counts unavailable for %s: %s", year, exc)
            return pd.DataFrame()

    return _concat_years("snap_counts", years, fetch, refresh=refresh)
