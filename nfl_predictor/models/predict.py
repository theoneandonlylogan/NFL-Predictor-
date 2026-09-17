"""Score a week's matchups and explain the largest feature contributions."""

from __future__ import annotations

import pandas as pd

from nfl_predictor import numpy_compat as _numpy_compat  # noqa: F401
from nfl_predictor.config import FEATURE_LABELS, current_nfl_season
from nfl_predictor.models.train import ensure_model


def default_week(features: pd.DataFrame, season: int | None = None) -> int:
    """Upcoming week if one exists, otherwise the latest week in the season."""
    season = season or current_nfl_season()
    slate = features[features["season"] == season]
    if slate.empty:
        return 1
    upcoming = slate[slate["home_score"].isna()]
    if not upcoming.empty:
        return int(upcoming["week"].min())
    return int(slate["week"].max())


def explain_row(row: pd.Series, artifact: dict, top_n: int = 5) -> list[str]:
    """Human-readable reasons from standardized feature * coefficient."""
    model = artifact["model"]
    cols = artifact["feature_columns"]
    imputer = model.named_steps["impute"]
    scaler = model.named_steps["scale"]
    clf = model.named_steps["model"]

    values = pd.DataFrame([row.reindex(cols)])
    imputed = imputer.transform(values)
    scaled = scaler.transform(imputed)[0]
    contrib = scaled * clf.coef_[0]
    ranked = sorted(zip(cols, contrib, scaled), key=lambda x: abs(x[1]), reverse=True)

    reasons: list[str] = []
    for name, weight, _ in ranked[:top_n]:
        label = FEATURE_LABELS.get(name, name)
        direction = "helps home" if weight > 0 else "helps away"
        reasons.append(f"{label} {direction}")
    return reasons


def predict_week(features: pd.DataFrame, season: int, week: int, artifact: dict | None = None) -> pd.DataFrame:
    week_games = features[(features["season"] == season) & (features["week"] == week)].copy()
    if week_games.empty:
        return week_games

    artifact = artifact or ensure_model(features)
    model = artifact["model"]
    cols = artifact["feature_columns"]
    for col in cols:
        week_games[col] = pd.to_numeric(week_games[col], errors="coerce")
    week_games["home_win_prob"] = model.predict_proba(week_games[cols])[:, 1]
    week_games["away_win_prob"] = 1.0 - week_games["home_win_prob"]
    week_games["predicted_winner"] = week_games.apply(
        lambda r: r["home_team"] if r["home_win_prob"] >= 0.5 else r["away_team"],
        axis=1,
    )
    week_games["predicted_win_prob"] = week_games[["home_win_prob", "away_win_prob"]].max(axis=1)
    week_games["reasons"] = week_games.apply(lambda r: "; ".join(explain_row(r, artifact)), axis=1)

    display_cols = [
        "season",
        "week",
        "gameday",
        "away_team",
        "home_team",
        "home_style",
        "away_style",
        "home_injury_penalty",
        "away_injury_penalty",
        "temp",
        "wind",
        "is_snow",
        "is_heat",
        "is_indoor",
        "home_win_prob",
        "away_win_prob",
        "predicted_winner",
        "predicted_win_prob",
        "reasons",
        "home_score",
        "away_score",
    ]
    display_cols = [c for c in display_cols if c in week_games.columns]
    return week_games[display_cols].sort_values(["gameday", "home_team"]).reset_index(drop=True)
