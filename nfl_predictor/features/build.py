"""Build one pre-kickoff feature row per scheduled game."""

from __future__ import annotations

import pandas as pd

from nfl_predictor.config import FORM_WINDOW, classify_style

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
            "rest": schedules["away_rest"] if "away_rest" in schedules.columns else pd.NA,
        }
    )
    return pd.concat([home, away], ignore_index=True)


def _add_pbp_stats(team_games: pd.DataFrame, pbp_team_game: pd.DataFrame) -> pd.DataFrame:
    df = team_games.merge(pbp_team_game, on=["game_id", "season", "week", "team"], how="left")
    pass_plays = df.get("off_pass_plays", 0).fillna(0)
    rush_plays = df.get("off_rush_plays", 0).fillna(0)
    total = pass_plays + rush_plays
    df["rush_rate"] = (rush_plays / total.replace(0, pd.NA)).astype(float)
    df["point_diff"] = df["points_for"] - df["points_against"]
    return df


def _expanding_prior(series: pd.Series) -> pd.Series:
    return series.shift(1).expanding(min_periods=1).mean()


def _add_rolling_priors(team_games: pd.DataFrame) -> pd.DataFrame:
    df = team_games.sort_values(["team", "season", "week", "gameday"]).copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")

    grouped = df.groupby(["team", "season"], group_keys=False)
    for col in TEAM_STAT_COLS:
        if col not in df.columns:
            df[col] = pd.NA
        df[f"roll_{col}"] = grouped[col].transform(_expanding_prior)

    df["roll_form_pd"] = grouped["point_diff"].transform(
        lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
    )
    df["roll_prior_games"] = grouped["points_for"].transform(lambda s: s.shift(1).expanding().count())

    home_only = df[df["is_home"] == 1].copy()
    away_only = df[df["is_home"] == 0].copy()
    home_only["roll_home_pd"] = home_only.groupby(["team", "season"], group_keys=False)["point_diff"].transform(
        _expanding_prior
    )
    away_only["roll_away_pd"] = away_only.groupby(["team", "season"], group_keys=False)["point_diff"].transform(
        _expanding_prior
    )
    df = df.merge(home_only[["game_id", "team", "roll_home_pd"]], on=["game_id", "team"], how="left")
    df = df.merge(away_only[["game_id", "team", "roll_away_pd"]], on=["game_id", "team"], how="left")

    prev = (
        df.groupby(["team", "season"], as_index=False)[TEAM_STAT_COLS]
        .mean(numeric_only=True)
        .rename(columns={c: f"prev_{c}" for c in TEAM_STAT_COLS})
    )
    prev["season"] = prev["season"] + 1
    df = df.merge(prev, on=["team", "season"], how="left")

    for col in TEAM_STAT_COLS:
        df[f"prior_{col}"] = df[f"roll_{col}"].fillna(df[f"prev_{col}"])
    df["prior_form_pd"] = df["roll_form_pd"].fillna(df["prior_point_diff"])
    df["prior_home_pd"] = df["roll_home_pd"].fillna(df["prior_point_diff"])
    df["prior_away_pd"] = df["roll_away_pd"].fillna(df["prior_point_diff"])
    df["prior_games"] = df["roll_prior_games"].fillna(0)

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
