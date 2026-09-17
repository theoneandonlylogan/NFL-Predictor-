"""Offline smoke test for feature building, injuries, weather flags, and training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nfl_predictor.config import FEATURE_COLUMNS, classify_style
from nfl_predictor.data.injuries import compute_injury_penalties, normalize_name
from nfl_predictor.data.weather import enrich_weather, weather_flags
from nfl_predictor.features.build import build_game_features
from nfl_predictor.features.profiles import team_profiles
from nfl_predictor.models.predict import default_week, predict_week
from nfl_predictor.models.train import train_final_model


def main() -> None:
    assert classify_style(0.50) == "run-heavy"
    assert classify_style(0.30) == "pass-heavy"
    assert classify_style(0.42) == "balanced"
    assert normalize_name("Patrick Mahomes II") == "patrickmahomes"
    flags = weather_flags(20, 20, snowfall=0.2, weather_text="snow", indoor=False)
    assert flags["is_snow"] == 1.0 and flags["is_cold"] == 1.0 and flags["is_windy"] == 1.0
    dome = weather_flags(20, 20, snowfall=0.2, indoor=True)
    assert dome["is_indoor"] == 1.0 and dome["is_snow"] == 0.0

    teams = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE"]
    rows = []
    rng = np.random.default_rng(0)
    gid = 0
    for week in range(1, 11):
        shuffled = teams.copy()
        rng.shuffle(shuffled)
        for i in range(0, 8, 2):
            gid += 1
            home, away = shuffled[i], shuffled[i + 1]
            rows.append(
                {
                    "game_id": f"2024_{week:02d}_{gid}",
                    "season": 2024,
                    "week": week,
                    "game_type": "REG",
                    "gameday": f"2024-09-{week:02d}",
                    "home_team": home,
                    "away_team": away,
                    "home_score": int(rng.integers(10, 40)),
                    "away_score": int(rng.integers(10, 40)),
                    "home_rest": 7,
                    "away_rest": 7,
                    "roof": "outdoor",
                    "temp": 70,
                    "wind": 8,
                    "weather": "clear",
                }
            )
    schedules = pd.DataFrame(rows)

    pbp_rows = []
    for r in rows:
        for team in (r["home_team"], r["away_team"]):
            pbp_rows.append(
                {
                    "game_id": r["game_id"],
                    "season": 2024,
                    "week": r["week"],
                    "team": team,
                    "off_plays": 60,
                    "off_epa": rng.normal(0, 0.1),
                    "off_pass_plays": 35,
                    "off_pass_epa": rng.normal(0, 0.12),
                    "off_rush_plays": 25,
                    "off_rush_epa": rng.normal(0, 0.08),
                    "off_success": 0.45,
                    "off_pass_success": 0.44,
                    "off_rush_success": 0.46,
                    "off_yards": 5.5,
                    "off_pass_yards": 7.0,
                    "off_rush_yards": 4.2,
                    "def_plays": 60,
                    "def_epa": rng.normal(0, 0.1),
                    "def_pass_plays": 35,
                    "def_pass_epa": rng.normal(0, 0.12),
                    "def_rush_plays": 25,
                    "def_rush_epa": rng.normal(0, 0.08),
                    "def_success": 0.45,
                    "def_pass_success": 0.44,
                    "def_rush_success": 0.46,
                    "def_yards": 5.5,
                    "def_pass_yards": 7.0,
                    "def_rush_yards": 4.2,
                }
            )
    pbp = pd.DataFrame(pbp_rows)

    inj = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 5,
                "team": "BUF",
                "full_name": "Josh Allen",
                "position": "QB",
                "report_status": "Out",
            }
        ]
    )
    snaps = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 4,
                "team": "BUF",
                "player": "Josh Allen",
                "position": "QB",
                "offense_pct": 99,
                "defense_pct": 0,
            }
        ]
    )
    pen = compute_injury_penalties(inj, snaps)
    assert (pen.loc[pen["team"] == "BUF", "qb_out"] == 1).all()
    assert pen.loc[pen["team"] == "BUF", "injury_penalty"].iloc[0] > 0.5

    schedules = enrich_weather(schedules, fetch_missing=False)
    feats = build_game_features(schedules, pbp, pen)
    missing = [c for c in FEATURE_COLUMNS if c not in feats.columns]
    assert not missing, missing
    assert feats.loc[feats["week"] >= 3, "home_off_epa"].notna().mean() > 0.5

    profiles = team_profiles(feats, 2024, week=10)
    assert set(profiles["team"]) == set(teams)

    artifact = train_final_model(feats, save=False)
    preds = predict_week(feats, 2024, 10, artifact=artifact)
    assert len(preds) == 4
    assert preds["predicted_winner"].notna().all()
    assert default_week(feats, 2024) == 10
    print(f"smoke ok: {len(feats)} games, week-10 mean P(home)={preds['home_win_prob'].mean():.3f}")


if __name__ == "__main__":
    main()
