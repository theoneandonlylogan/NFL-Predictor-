"""End-to-end helpers used by the notebooks."""

from __future__ import annotations

import logging

import pandas as pd

from nfl_predictor.config import season_years
from nfl_predictor.data.ingest import load_injuries, load_schedules, load_snap_counts, load_team_game_pbp
from nfl_predictor.data.injuries import compute_injury_penalties
from nfl_predictor.data.weather import enrich_weather
from nfl_predictor.features.build import build_game_features

logger = logging.getLogger(__name__)


def build_feature_table(
    years: list[int] | None = None,
    refresh: bool = False,
    fetch_weather: bool = True,
) -> pd.DataFrame:
    years = years or season_years()
    print(f"Loading schedules {years[0]}-{years[-1]}...")
    schedules = load_schedules(years, refresh=refresh)
    print("Loading play-by-play team-game stats (first run is slow)...")
    pbp = load_team_game_pbp(years, refresh=refresh)
    print("Loading injuries and snap counts...")
    injuries = load_injuries(years, refresh=refresh)
    snaps = load_snap_counts(years, refresh=refresh)
    injury_table = compute_injury_penalties(injuries, snaps)
    print("Filling weather for upcoming outdoor games...")
    schedules = enrich_weather(schedules, fetch_missing=fetch_weather)
    print("Building pre-kickoff features...")
    return build_game_features(schedules, pbp, injury_table)


def load_week_matchups(
    season: int,
    week: int,
    years: list[int] | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    features = build_feature_table(years=years or season_years(end_season=season), refresh=refresh)
    return features[(features["season"] == season) & (features["week"] == week)].copy()
