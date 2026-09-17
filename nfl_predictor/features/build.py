"""Build one pre-kickoff feature row per scheduled game."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_predictor.config import (
    FIRST_CURRENT_GAME_WEIGHT,
    PREV_SEASON_BLEND_GAMES,
    classify_style,
)

TEAM_STAT_COLS = [
    "off_epa",
    "off_rush_epa",
    "off_pass_epa",
    "off_success",
    "def_epa",
    "def_rush_epa",
    "def_pass_epa",
    "points_for",
    "points_against",
    "point_diff",
    "rush_rate",
]


def _team_game_rows(schedules: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "game_id": schedules["game_id"],
            "season": schedules["season"],
            "week": schedules["week"],
            "gameday": schedules["gameday"],
            "team": schedules["home_team"],
            "opponent": schedules["away_team"],
            "is_home": 1,
            "points_for": schedules["home_score"],
            "points_against": schedules["away_score"],
            "game_type": schedules["game_type"] if "game_type" in schedules.columns else "REG",
            "rest": schedules["home_rest"] if "home_rest" in schedules.columns else pd.NA,
        }
    )
    away = pd.DataFrame(
        {
            "game_id": schedules["game_id"],
            "season": schedules["season"],
            "week": schedules["week"],
            "gameday": schedules["gameday"],
            "team": schedules["away_team"],
            "opponent": schedules["home_team"],
            "is_home": 0,
            "points_for": schedules["away_score"],
            "points_against": schedules["home_score"],
            "game_type": schedules["game_type"] if "game_type" in schedules.columns else "REG",
            "rest": schedules["away_rest"] if "away_rest" in schedules.columns else pd.NA,
        }
    )
    return pd.concat([home, away], ignore_index=True)


def _add_pbp_stats(team_games: pd.DataFrame, pbp_team_game: pd.DataFrame) -> pd.DataFrame:
    df = team_games.merge(pbp_team_game, on=["game_id", "season", "week", "team"], how="left")
    pass_plays = pd.to_numeric(df["off_pass_plays"], errors="coerce") if "off_pass_plays" in df.columns else 0
    rush_plays = pd.to_numeric(df["off_rush_plays"], errors="coerce") if "off_rush_plays" in df.columns else 0
    if not isinstance(pass_plays, pd.Series):
        pass_plays = pd.Series(0, index=df.index, dtype="float64")
    if not isinstance(rush_plays, pd.Series):
        rush_plays = pd.Series(0, index=df.index, dtype="float64")
    pass_plays = pass_plays.fillna(0)
    rush_plays = rush_plays.fillna(0)
    total = pass_plays + rush_plays
    df["rush_rate"] = (rush_plays / total.mask(total == 0)).astype("float64")
    df["point_diff"] = pd.to_numeric(df["points_for"], errors="coerce") - pd.to_numeric(
        df["points_against"], errors="coerce"
    )
    return df


def _prev_season_keep(n_current: int) -> int:
    """How many late prior-season REG games to mix in given current-season sample size."""
    return max(0, PREV_SEASON_BLEND_GAMES - max(0, n_current - 1))


def _weighted_mean(values: pd.Series, weights: np.ndarray) -> float:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(vals[mask], weights=w[mask]))


def _priors_for_team(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
    if "game_type" not in g.columns:
        g["game_type"] = "REG"
    for col in TEAM_STAT_COLS:
        if col not in g.columns:
            g[col] = np.nan

    records: list[dict] = []
    for i in range(len(g)):
        season = g.at[i, "season"]
        past = g.iloc[:i]
        past = past[past["points_for"].notna()]
        current = past[past["season"] == season]
        n_curr = len(current)
        n_prev = _prev_season_keep(n_curr)
        previous = past[(past["season"] == season - 1) & (past["game_type"].fillna("REG") == "REG")]
        previous = previous.sort_values(["week", "gameday"]).tail(n_prev)

        pieces: list[pd.DataFrame] = []
        weights: list[float] = []
        if not previous.empty:
            pieces.append(previous)
            weights.extend([1.0] * len(previous))
        if n_curr == 1:
            pieces.append(current)
            weights.append(FIRST_CURRENT_GAME_WEIGHT)
        elif n_curr > 1:
            pieces.append(current)
            weights.extend([1.0] * n_curr)

        row_priors = {f"prior_{col}": np.nan for col in TEAM_STAT_COLS}
        row_priors["prior_form_pd"] = np.nan
        row_priors["prior_home_pd"] = np.nan
        row_priors["prior_away_pd"] = np.nan
        row_priors["prior_games"] = float(sum(weights)) if weights else 0.0

        if pieces:
            window = pd.concat(pieces, ignore_index=True)
            w = np.array(weights, dtype=float)
            for col in TEAM_STAT_COLS:
                row_priors[f"prior_{col}"] = _weighted_mean(window[col], w)
            row_priors["prior_form_pd"] = _weighted_mean(window["point_diff"], w)
            home_mask = window["is_home"].to_numpy() == 1
            away_mask = ~home_mask
            if home_mask.any():
                row_priors["prior_home_pd"] = _weighted_mean(window.loc[home_mask, "point_diff"], w[home_mask])
            if away_mask.any():
                row_priors["prior_away_pd"] = _weighted_mean(window.loc[away_mask, "point_diff"], w[away_mask])
        records.append(row_priors)

    return pd.concat([g, pd.DataFrame(records)], axis=1)


def _add_rolling_priors(team_games: pd.DataFrame) -> pd.DataFrame:
    df = team_games.sort_values(["team", "season", "week", "gameday"]).copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    parts = [_priors_for_team(group) for _, group in df.groupby("team", sort=False)]
    df = pd.concat(parts, ignore_index=True)
    league_rush = df["prior_rush_rate"].mean(skipna=True)
    df["style"] = df["prior_rush_rate"].apply(
        lambda x: classify_style(float(x), league_rush) if pd.notna(x) else "balanced"
    )
    return df


def _prefix_team_priors(priors: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = [
        "game_id",
        "style",
        "prior_off_epa",
        "prior_off_rush_epa",
        "prior_off_pass_epa",
        "prior_def_epa",
        "prior_def_rush_epa",
        "prior_def_pass_epa",
        "prior_points_for",
        "prior_points_against",
        "prior_rush_rate",
        "prior_form_pd",
        "prior_home_pd",
        "prior_away_pd",
        "prior_games",
    ]
    keep = [c for c in keep if c in priors.columns]
    out = priors[keep].copy()
    rename = {c: f"{prefix}_{c}" for c in out.columns if c != "game_id"}
    return out.rename(columns=rename)


def build_game_features(
    schedules: pd.DataFrame,
    pbp_team_game: pd.DataFrame,
    injury_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one row per game with only information available before kickoff."""
    if schedules.empty:
        return pd.DataFrame()

    team_games = _team_game_rows(schedules)
    team_games = _add_pbp_stats(team_games, pbp_team_game if pbp_team_game is not None else pd.DataFrame())
    priors = _add_rolling_priors(team_games)

    home_priors = _prefix_team_priors(priors[priors["is_home"] == 1], "home")
    away_priors = _prefix_team_priors(priors[priors["is_home"] == 0], "away")

    games = schedules.copy()
    games["season"] = pd.to_numeric(games["season"], errors="coerce")
    games["week"] = pd.to_numeric(games["week"], errors="coerce")
    games = games.merge(home_priors, on="game_id", how="left")
    games = games.merge(away_priors, on="game_id", how="left")

    if injury_table is not None and not injury_table.empty:
        inj = injury_table.rename(columns={"team": "home_team"})
        games = games.merge(
            inj,
            left_on=["season", "week", "home_team"],
            right_on=["season", "week", "home_team"],
            how="left",
        )
        games = games.rename(
            columns={
                "injury_penalty": "home_injury_penalty",
                "n_starters_out": "home_n_starters_out",
                "qb_out": "home_qb_out",
            }
        )
        inj_away = injury_table.rename(columns={"team": "away_team"})
        games = games.merge(
            inj_away,
            left_on=["season", "week", "away_team"],
            right_on=["season", "week", "away_team"],
            how="left",
        )
        games = games.rename(
            columns={
                "injury_penalty": "away_injury_penalty",
                "n_starters_out": "away_n_starters_out",
                "qb_out": "away_qb_out",
            }
        )
    else:
        games["home_injury_penalty"] = 0.0
        games["away_injury_penalty"] = 0.0
        games["home_n_starters_out"] = 0.0
        games["away_n_starters_out"] = 0.0
        games["home_qb_out"] = 0.0
        games["away_qb_out"] = 0.0

    for col in [
        "home_injury_penalty",
        "away_injury_penalty",
        "home_n_starters_out",
        "away_n_starters_out",
        "home_qb_out",
        "away_qb_out",
    ]:
        if col in games.columns:
            games[col] = games[col].fillna(0.0)

    games["home_off_epa"] = games["home_prior_off_epa"]
    games["away_off_epa"] = games["away_prior_off_epa"]
    games["home_off_rush_epa"] = games["home_prior_off_rush_epa"]
    games["away_off_rush_epa"] = games["away_prior_off_rush_epa"]
    games["home_off_pass_epa"] = games["home_prior_off_pass_epa"]
    games["away_off_pass_epa"] = games["away_prior_off_pass_epa"]
    games["home_rush_rate"] = games["home_prior_rush_rate"]
    games["away_rush_rate"] = games["away_prior_rush_rate"]
    games["home_def_rush_epa"] = games["home_prior_def_rush_epa"]
    games["away_def_rush_epa"] = games["away_prior_def_rush_epa"]
    games["home_def_pass_epa"] = games["home_prior_def_pass_epa"]
    games["away_def_pass_epa"] = games["away_prior_def_pass_epa"]
    games["home_ppg"] = games["home_prior_points_for"]
    games["away_ppg"] = games["away_prior_points_for"]
    games["home_papg"] = games["home_prior_points_against"]
    games["away_papg"] = games["away_prior_points_against"]
    games["home_form_pd"] = games["home_prior_form_pd"]
    games["away_form_pd"] = games["away_prior_form_pd"]
    games["home_home_pd"] = games["home_prior_home_pd"]
    games["away_away_pd"] = games["away_prior_away_pd"]
    games["home_prior_games"] = games["home_prior_games"]
    games["away_prior_games"] = games["away_prior_games"]

    # Positive edge: your rush EPA minus opponent EPA allowed on rushes.
    games["rush_edge_home"] = games["home_off_rush_epa"] - games["away_def_rush_epa"]
    games["rush_edge_away"] = games["away_off_rush_epa"] - games["home_def_rush_epa"]
    games["pass_edge_home"] = games["home_off_pass_epa"] - games["away_def_pass_epa"]
    games["pass_edge_away"] = games["away_off_pass_epa"] - games["home_def_pass_epa"]
    games["rush_edge_net"] = games["rush_edge_home"] - games["rush_edge_away"]
    games["pass_edge_net"] = games["pass_edge_home"] - games["pass_edge_away"]

    if "home_rest" not in games.columns:
        games["home_rest"] = 7
    if "away_rest" not in games.columns:
        games["away_rest"] = 7
    games["home_rest"] = pd.to_numeric(games["home_rest"], errors="coerce").fillna(7)
    games["away_rest"] = pd.to_numeric(games["away_rest"], errors="coerce").fillna(7)
    games["rest_diff"] = games["home_rest"] - games["away_rest"]

    if "is_indoor" not in games.columns:
        games["is_indoor"] = 0.0
    for col in ["is_heat", "is_cold", "is_snow", "is_rain", "is_windy"]:
        if col not in games.columns:
            games[col] = 0.0
    if "temp" not in games.columns:
        games["temp"] = pd.NA
    if "wind" not in games.columns:
        games["wind"] = pd.NA

    games["home_pass_heavy"] = (games.get("home_style") == "pass-heavy").astype(float)
    games["away_pass_heavy"] = (games.get("away_style") == "pass-heavy").astype(float)
    bad_weather = (
        games["is_snow"].fillna(0)
        + games["is_windy"].fillna(0)
        + games["is_rain"].fillna(0)
        + games["is_cold"].fillna(0)
    ).clip(upper=1)
    games["home_pass_weather"] = games["home_pass_heavy"] * bad_weather
    games["away_pass_weather"] = games["away_pass_heavy"] * bad_weather

    completed = games["home_score"].notna() & games["away_score"].notna()
    games["home_win"] = pd.NA
    games.loc[completed, "home_win"] = (games.loc[completed, "home_score"] > games.loc[completed, "away_score"]).astype(
        int
    )
    games["tie"] = False
    games.loc[completed, "tie"] = games.loc[completed, "home_score"] == games.loc[completed, "away_score"]
    return games
