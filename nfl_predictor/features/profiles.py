"""Team profile tables for the exploration notebook."""

from __future__ import annotations

import pandas as pd

from nfl_predictor.config import classify_style


def team_profiles(features: pd.DataFrame, season: int, week: int | None = None) -> pd.DataFrame:
    """Latest pre-game profile for each team as of the given season/week."""
    df = features[features["season"] == season].copy()
    if df.empty:
        return pd.DataFrame()
    if week is not None:
        df = df[df["week"] <= week]
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("week")

    home = pd.DataFrame(
        {
            "week": df["week"],
            "team": df["home_team"],
            "rush_rate": df["home_rush_rate"],
            "off_epa": df["home_off_epa"],
            "rush_epa": df["home_off_rush_epa"],
            "pass_epa": df["home_off_pass_epa"],
            "def_rush_epa": df["home_def_rush_epa"],
            "def_pass_epa": df["home_def_pass_epa"],
            "ppg": df["home_ppg"],
            "papg": df["home_papg"],
            "home_pd": df["home_home_pd"],
            "road_pd": pd.NA,
            "form_pd": df["home_form_pd"],
            "injury_penalty": df["home_injury_penalty"],
            "starters_out": df.get("home_n_starters_out", 0),
            "qb_out": df["home_qb_out"],
        }
    )
    away = pd.DataFrame(
        {
            "week": df["week"],
            "team": df["away_team"],
            "rush_rate": df["away_rush_rate"],
            "off_epa": df["away_off_epa"],
            "rush_epa": df["away_off_rush_epa"],
            "pass_epa": df["away_off_pass_epa"],
            "def_rush_epa": df["away_def_rush_epa"],
            "def_pass_epa": df["away_def_pass_epa"],
            "ppg": df["away_ppg"],
            "papg": df["away_papg"],
            "home_pd": pd.NA,
            "road_pd": df["away_away_pd"],
            "form_pd": df["away_form_pd"],
            "injury_penalty": df["away_injury_penalty"],
            "starters_out": df.get("away_n_starters_out", 0),
            "qb_out": df["away_qb_out"],
        }
    )
    stacked = pd.concat([home, away], ignore_index=True)
    latest = stacked.sort_values("week").groupby("team", as_index=False).last()
    home_pd = (
        stacked.dropna(subset=["home_pd"]).sort_values("week").groupby("team")["home_pd"].last()
    )
    road_pd = (
        stacked.dropna(subset=["road_pd"]).sort_values("week").groupby("team")["road_pd"].last()
    )
    latest["home_pd"] = latest["team"].map(home_pd)
    latest["road_pd"] = latest["team"].map(road_pd)
    league_rush = latest["rush_rate"].mean(skipna=True)
    latest["style"] = latest["rush_rate"].apply(
        lambda x: classify_style(float(x), league_rush) if pd.notna(x) else "balanced"
    )
    cols = [
        "team",
        "style",
        "rush_rate",
        "off_epa",
        "rush_epa",
        "pass_epa",
        "def_rush_epa",
        "def_pass_epa",
        "ppg",
        "papg",
        "home_pd",
        "road_pd",
        "form_pd",
        "injury_penalty",
        "starters_out",
        "qb_out",
    ]
    return latest[cols].sort_values("team").reset_index(drop=True)
