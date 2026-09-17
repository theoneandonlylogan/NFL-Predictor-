"""Run this week's NFL predictions.

Uses the pandas/sklearn pipeline when those libraries load. On Windows
Application Control hosts that block those DLLs, falls back to numpy + csv.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import sys
import urllib.request
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nfl_predictor.config import (  # noqa: E402
    COLD_TEMP_F,
    DATA_RAW,
    FIRST_CURRENT_GAME_WEIGHT,
    HEAT_TEMP_F,
    POSITION_WEIGHTS,
    PREV_SEASON_BLEND_GAMES,
    STATUS_WEIGHTS,
    TEAM_STADIUMS,
    WINDY_MPH,
    classify_style,
    current_nfl_season,
    season_years,
)

USER_AGENT = "nfl-predictor/1.0"
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
FEATURE_NAMES = [
    "home_off_epa",
    "away_off_epa",
    "home_off_rush_epa",
    "away_off_rush_epa",
    "home_off_pass_epa",
    "away_off_pass_epa",
    "home_rush_rate",
    "away_rush_rate",
    "home_def_rush_epa",
    "away_def_rush_epa",
    "home_def_pass_epa",
    "away_def_pass_epa",
    "home_ppg",
    "away_ppg",
    "home_papg",
    "away_papg",
    "rush_edge_net",
    "pass_edge_net",
    "home_form_pd",
    "away_form_pd",
    "home_injury_penalty",
    "away_injury_penalty",
    "home_qb_out",
    "away_qb_out",
    "is_indoor",
    "is_heat",
    "is_cold",
    "is_snow",
    "is_windy",
    "home_pass_weather",
    "away_pass_weather",
]


def _download(url: str, dest: Path, refresh: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not refresh:
        return dest
    print(f"Downloading {dest.name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as resp:
        dest.write_bytes(resp.read())
    return dest


def _open_table(path: Path):
    if path.suffix == ".gz" or path.name.endswith(".csv.gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _f(value) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: list[float]) -> float:
    nums = [v for v in values if v == v]
    return sum(nums) / len(nums) if nums else float("nan")


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    nums = []
    wts = []
    for value, weight in zip(values, weights):
        if value == value and weight > 0:
            nums.append(value)
            wts.append(weight)
    if not nums:
        return float("nan")
    return sum(v * w for v, w in zip(nums, wts)) / sum(wts)


def _prev_season_keep(n_current: int) -> int:
    return max(0, PREV_SEASON_BLEND_GAMES - max(0, n_current - 1))


def load_games(years: list[int]) -> list[dict]:
    path = _download(f"{NFLVERSE}/schedules/games.csv", DATA_RAW / "games.csv")
    games = []
    with _open_table(path) as fh:
        for row in csv.DictReader(fh):
            try:
                season = int(float(row["season"]))
            except (TypeError, ValueError, KeyError):
                continue
            if season not in years:
                continue
            if row.get("game_type") not in {"REG", "WC", "DIV", "CON", "SB", ""}:
                continue
            games.append(row)
    return games


def _accum() -> dict:
    return {
        "pass_n": 0,
        "rush_n": 0,
        "pass_epa": 0.0,
        "rush_epa": 0.0,
        "epa": 0.0,
        "n": 0,
    }


def _add_play(bucket: dict, is_pass: bool, is_rush: bool, epa: float) -> None:
    if epa != epa:
        return
    bucket["n"] += 1
    bucket["epa"] += epa
    if is_pass:
        bucket["pass_n"] += 1
        bucket["pass_epa"] += epa
    if is_rush:
        bucket["rush_n"] += 1
        bucket["rush_epa"] += epa


def _finalize(bucket: dict) -> dict:
    n = bucket["n"] or float("nan")
    pn = bucket["pass_n"] or float("nan")
    rn = bucket["rush_n"] or float("nan")
    return {
        "off_epa": bucket["epa"] / n if bucket["n"] else float("nan"),
        "off_pass_epa": bucket["pass_epa"] / pn if bucket["pass_n"] else float("nan"),
        "off_rush_epa": bucket["rush_epa"] / rn if bucket["rush_n"] else float("nan"),
        "rush_rate": bucket["rush_n"] / n if bucket["n"] else float("nan"),
        "pass_n": bucket["pass_n"],
        "rush_n": bucket["rush_n"],
    }


def load_pbp_team_games(years: list[int]) -> dict[tuple[str, str], dict]:
    """(game_id, team) -> offense stats plus def_* from the other side of the ball."""
    offense: dict[tuple[str, str], dict] = defaultdict(_accum)
    defense: dict[tuple[str, str], dict] = defaultdict(_accum)
    for year in years:
        path = _download(
            f"{NFLVERSE}/pbp/play_by_play_{year}.csv.gz",
            DATA_RAW / f"play_by_play_{year}.csv.gz",
        )
        print(f"Aggregating {year} play-by-play ...")
        with _open_table(path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                play_type = (row.get("play_type") or "").lower()
                is_pass = str(row.get("pass") or "").strip() in {"1", "true", "True"} or play_type == "pass"
                is_rush = str(row.get("rush") or "").strip() in {"1", "true", "True"} or play_type in {"run", "rush"}
                if not is_pass and not is_rush:
                    continue
                posteam = row.get("posteam") or ""
                defteam = row.get("defteam") or ""
                game_id = row.get("game_id") or ""
                if not posteam or not defteam or not game_id:
                    continue
                epa = _f(row.get("epa"))
                _add_play(offense[(game_id, posteam)], is_pass, is_rush, epa)
                _add_play(defense[(game_id, defteam)], is_pass, is_rush, epa)
    out = {}
    keys = set(offense) | set(defense)
    for key in keys:
        off = _finalize(offense[key]) if key in offense else _finalize(_accum())
        deff = _finalize(defense[key]) if key in defense else _finalize(_accum())
        out[key] = {
            **off,
            "def_epa": deff["off_epa"],
            "def_pass_epa": deff["off_pass_epa"],
            "def_rush_epa": deff["off_rush_epa"],
        }
    return out


def load_injuries(years: list[int]) -> dict[tuple[int, int, str], dict]:
    table: dict[tuple[int, int, str], dict] = defaultdict(lambda: {"penalty": 0.0, "qb_out": 0.0})
    for year in years:
        try:
            path = _download(
                f"{NFLVERSE}/injuries/injuries_{year}.csv",
                DATA_RAW / f"injuries_{year}.csv",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Injuries {year} skipped: {exc}")
            continue
        with _open_table(path) as fh:
            for row in csv.DictReader(fh):
                status = str(row.get("report_status") or "").strip().lower()
                weight = STATUS_WEIGHTS.get(status, 0.0)
                if weight <= 0:
                    continue
                try:
                    season = int(float(row["season"]))
                    week = int(float(row["week"]))
                except (TypeError, ValueError, KeyError):
                    continue
                team = str(row.get("team") or "")
                pos = str(row.get("position") or "").strip().upper()
                pos_w = POSITION_WEIGHTS.get(pos, 0.15)
                key = (season, week, team)
                table[key]["penalty"] += weight * pos_w
                if pos == "QB" and weight >= 0.75:
                    table[key]["qb_out"] = 1.0
    return table


def _weather(row: dict) -> dict:
    roof = str(row.get("roof") or "").lower()
    home = str(row.get("home_team") or "")
    indoor = roof in {"dome", "closed"} or TEAM_STADIUMS.get(home, {}).get("roof") == "dome"
    if indoor:
        return {"is_indoor": 1.0, "is_heat": 0.0, "is_cold": 0.0, "is_snow": 0.0, "is_windy": 0.0, "temp": float("nan")}
    temp = _f(row.get("temp"))
    wind = _f(row.get("wind"))
    text = str(row.get("weather") or "").lower()
    return {
        "is_indoor": 0.0,
        "is_heat": 1.0 if temp == temp and temp >= HEAT_TEMP_F else 0.0,
        "is_cold": 1.0 if temp == temp and temp <= COLD_TEMP_F else 0.0,
        "is_snow": 1.0 if "snow" in text else 0.0,
        "is_windy": 1.0 if wind == wind and wind >= WINDY_MPH else 0.0,
        "temp": temp,
        "wind": wind,
    }


def _fetch_forecast(row: dict) -> dict:
    home = str(row.get("home_team") or "")
    stadium = TEAM_STADIUMS.get(home)
    gameday = row.get("gameday") or ""
    if not stadium or not gameday:
        return {}
    try:
        kickoff = datetime.strptime(str(gameday)[:10], "%Y-%m-%d")
    except ValueError:
        return {}
    params = (
        f"latitude={stadium['lat']}&longitude={stadium['lon']}"
        f"&hourly=temperature_2m,precipitation,snowfall,wind_speed_10m"
        f"&start_date={kickoff:%Y-%m-%d}&end_date={kickoff:%Y-%m-%d}"
        f"&timezone={stadium['tz']}&temperature_unit=fahrenheit&wind_speed_unit=mph"
    )
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}
    hourly = payload.get("hourly") or {}
    temps = hourly.get("temperature_2m") or []
    winds = hourly.get("wind_speed_10m") or hourly.get("windspeed_10m") or []
    snows = hourly.get("snowfall") or []
    hour = 13
    gametime = str(row.get("gametime") or "")
    if ":" in gametime:
        try:
            hour = int(gametime.split(":")[0])
        except ValueError:
            hour = 13
    idx = min(hour, len(temps) - 1) if temps else 0
    out = {}
    if temps:
        out["temp"] = temps[idx]
    if winds:
        out["wind"] = winds[idx]
    if snows and snows[idx] and snows[idx] > 0:
        out["weather"] = "snow"
    return out


def _team_game_stream(games: list[dict], pbp: dict, injuries: dict) -> dict[tuple[str, int, int], dict]:
    """Chronological per-team completed-game stats with early-season prior blend."""
    history: dict[str, list[dict]] = defaultdict(list)

    def prior(team: str, season: int) -> dict:
        rows = history[team]
        current = [r for r in rows if r.get("season") == season]
        n_curr = len(current)
        n_prev = _prev_season_keep(n_curr)
        previous = [
            r
            for r in rows
            if r.get("season") == season - 1 and (r.get("game_type") or "REG") == "REG"
        ]
        previous = previous[-n_prev:] if n_prev else []
        window = previous + current
        if not window:
            return {}
        if n_curr == 1:
            weights = [1.0] * len(previous) + [FIRST_CURRENT_GAME_WEIGHT]
        else:
            weights = [1.0] * len(window)
        return {
            "off_epa": _weighted_mean([r["off_epa"] for r in window], weights),
            "off_rush_epa": _weighted_mean([r["off_rush_epa"] for r in window], weights),
            "off_pass_epa": _weighted_mean([r["off_pass_epa"] for r in window], weights),
            "def_rush_epa": _weighted_mean([r["def_rush_epa"] for r in window], weights),
            "def_pass_epa": _weighted_mean([r["def_pass_epa"] for r in window], weights),
            "rush_rate": _weighted_mean([r["rush_rate"] for r in window], weights),
            "ppg": _weighted_mean([r["pf"] for r in window], weights),
            "papg": _weighted_mean([r["pa"] for r in window], weights),
            "form_pd": _weighted_mean([r["pf"] - r["pa"] for r in window], weights),
            "n": sum(weights),
        }

    ordered = sorted(games, key=lambda r: (r.get("gameday") or "", r.get("game_id") or ""))
    features = {}
    today = date.today().isoformat()
    for row in ordered:
        home = row.get("home_team") or ""
        away = row.get("away_team") or ""
        gid = row.get("game_id") or ""
        try:
            season = int(float(row["season"]))
            week = int(float(row["week"]))
        except (TypeError, ValueError, KeyError):
            continue
        hp = prior(home, season)
        ap = prior(away, season)
        inj_h = injuries.get((season, week, home), {"penalty": 0.0, "qb_out": 0.0})
        inj_a = injuries.get((season, week, away), {"penalty": 0.0, "qb_out": 0.0})
        wx_row = dict(row)
        gameday = str(row.get("gameday") or "")
        if gameday >= today and (row.get("temp") in (None, "")):
            wx_row.update(_fetch_forecast(row))
        wx = _weather(wx_row)
        h_style = classify_style(hp.get("rush_rate", 0.42) if hp.get("rush_rate") == hp.get("rush_rate") else 0.42)
        a_style = classify_style(ap.get("rush_rate", 0.42) if ap.get("rush_rate") == ap.get("rush_rate") else 0.42)
        bad = max(wx["is_snow"], wx["is_windy"], wx["is_cold"])
        rush_edge_h = hp.get("off_rush_epa", float("nan")) - ap.get("def_rush_epa", float("nan"))
        rush_edge_a = ap.get("off_rush_epa", float("nan")) - hp.get("def_rush_epa", float("nan"))
        pass_edge_h = hp.get("off_pass_epa", float("nan")) - ap.get("def_pass_epa", float("nan"))
        pass_edge_a = ap.get("off_pass_epa", float("nan")) - hp.get("def_pass_epa", float("nan"))
        feat = {
            "game_id": gid,
            "season": season,
            "week": week,
            "gameday": gameday,
            "home_team": home,
            "away_team": away,
            "home_style": h_style,
            "away_style": a_style,
            "home_off_epa": hp.get("off_epa", float("nan")),
            "away_off_epa": ap.get("off_epa", float("nan")),
            "home_off_rush_epa": hp.get("off_rush_epa", float("nan")),
            "away_off_rush_epa": ap.get("off_rush_epa", float("nan")),
            "home_off_pass_epa": hp.get("off_pass_epa", float("nan")),
            "away_off_pass_epa": ap.get("off_pass_epa", float("nan")),
            "home_rush_rate": hp.get("rush_rate", float("nan")),
            "away_rush_rate": ap.get("rush_rate", float("nan")),
            "home_def_rush_epa": hp.get("def_rush_epa", float("nan")),
            "away_def_rush_epa": ap.get("def_rush_epa", float("nan")),
            "home_def_pass_epa": hp.get("def_pass_epa", float("nan")),
            "away_def_pass_epa": ap.get("def_pass_epa", float("nan")),
            "home_ppg": hp.get("ppg", float("nan")),
            "away_ppg": ap.get("ppg", float("nan")),
            "home_papg": hp.get("papg", float("nan")),
            "away_papg": ap.get("papg", float("nan")),
            "rush_edge_net": rush_edge_h - rush_edge_a,
            "pass_edge_net": pass_edge_h - pass_edge_a,
            "home_form_pd": hp.get("form_pd", float("nan")),
            "away_form_pd": ap.get("form_pd", float("nan")),
            "home_injury_penalty": inj_h["penalty"],
            "away_injury_penalty": inj_a["penalty"],
            "home_qb_out": inj_h["qb_out"],
            "away_qb_out": inj_a["qb_out"],
            "is_indoor": wx["is_indoor"],
            "is_heat": wx["is_heat"],
            "is_cold": wx["is_cold"],
            "is_snow": wx["is_snow"],
            "is_windy": wx["is_windy"],
            "home_pass_weather": 1.0 if h_style == "pass-heavy" and bad else 0.0,
            "away_pass_weather": 1.0 if a_style == "pass-heavy" and bad else 0.0,
            "temp": wx.get("temp", float("nan")),
            "wind": wx.get("wind", float("nan")),
            "home_score": _f(row.get("home_score")),
            "away_score": _f(row.get("away_score")),
        }
        hs, aws = feat["home_score"], feat["away_score"]
        feat["home_win"] = 1.0 if hs == hs and aws == aws and hs > aws else (0.0 if hs == hs and aws == aws and hs != aws else float("nan"))
        features[(gid, season, week)] = feat

        if hs == hs and aws == aws:
            for team, opp_score_for, opp_score_against, is_home in (
                (home, hs, aws, True),
                (away, aws, hs, False),
            ):
                stats = pbp.get((gid, team), {})
                history[team].append(
                    {
                        "season": season,
                        "week": week,
                        "game_type": row.get("game_type") or "REG",
                        "is_home": is_home,
                        "off_epa": stats.get("off_epa", float("nan")),
                        "off_rush_epa": stats.get("off_rush_epa", float("nan")),
                        "off_pass_epa": stats.get("off_pass_epa", float("nan")),
                        "def_rush_epa": stats.get("def_rush_epa", float("nan")),
                        "def_pass_epa": stats.get("def_pass_epa", float("nan")),
                        "rush_rate": stats.get("rush_rate", float("nan")),
                        "pf": opp_score_for,
                        "pa": opp_score_against,
                    }
                )
    return features


def _xy(rows: list[dict]):
    import numpy as np

    y = []
    xrows = []
    kept = []
    for row in rows:
        if row["home_win"] != row["home_win"]:
            continue
        vec = [row[name] for name in FEATURE_NAMES]
        xrows.append(vec)
        y.append(row["home_win"])
        kept.append(row)
    X = np.array(xrows, dtype=float)
    y = np.array(y, dtype=float)
    return X, y, kept


def _standardize(X, mu=None, sd=None):
    import numpy as np

    if mu is None:
        mu = np.nanmean(X, axis=0)
    if sd is None:
        sd = np.nanstd(X, axis=0)
        sd = np.where(sd == 0, 1.0, sd)
    filled = np.where(np.isnan(X), mu, X)
    return (filled - mu) / sd, mu, sd


def fit_logreg(X, y, steps: int = 250, lr: float = 0.35):
    import numpy as np

    Xs, mu, sd = _standardize(X)
    Xs = np.column_stack([np.ones(len(Xs)), Xs])
    w = np.zeros(Xs.shape[1])
    for _ in range(steps):
        z = np.clip(Xs @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xs.T @ (p - y) / len(y)
        w -= lr * grad
    return w, mu, sd


def predict_proba(X, w, mu, sd):
    import numpy as np

    Xs, _, _ = _standardize(X, mu, sd)
    Xs = np.column_stack([np.ones(len(Xs)), Xs])
    z = np.clip(Xs @ w, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def _reasons(row: dict, w) -> str:
    contrib = []
    for i, name in enumerate(FEATURE_NAMES):
        val = row[name]
        if val != val:
            continue
        weight = w[i + 1]
        contrib.append((abs(weight * (val if abs(val) < 50 else 1)), name, weight))
    contrib.sort(reverse=True)
    parts = []
    for _, name, weight in contrib[:4]:
        side = "helps home" if weight > 0 else "helps away"
        parts.append(f"{name} {side}")
    return "; ".join(parts)


def run_numpy_fallback(years: list[int] | None = None) -> None:
    import numpy as np

    years = years or season_years()
    print(f"Numpy fallback: seasons {years[0]}-{years[-1]}")
    games = load_games(years)
    print(f"Loaded {len(games)} scheduled games")
    pbp = load_pbp_team_games(years)
    print(f"PBP team-games: {len(pbp)}")
    injuries = load_injuries(years)
    feats = _team_game_stream(games, pbp, injuries)
    rows = list(feats.values())
    X, y, labeled = _xy(rows)
    print(f"Training on {len(labeled)} completed games")
    w, mu, sd = fit_logreg(X, y)
    pred = predict_proba(X, w, mu, sd)
    acc = float(np.mean((pred >= 0.5) == y))
    print(f"In-sample accuracy: {acc:.1%}")

    season = current_nfl_season()
    season_rows = [r for r in rows if r["season"] == season]
    upcoming = [r for r in season_rows if r["home_win"] != r["home_win"]]
    if upcoming:
        week = min(r["week"] for r in upcoming)
    elif season_rows:
        week = max(r["week"] for r in season_rows)
    else:
        print("No games found for the current season.")
        return
    slate = [r for r in season_rows if r["week"] == week]
    print(f"\n=== {season} week {week} predictions ===\n")
    Xu = []
    for row in slate:
        Xu.append([row[name] for name in FEATURE_NAMES])
    probs = predict_proba(np.array(Xu, dtype=float), w, mu, sd)
    header = f"{'Kickoff':<12} {'Matchup':<18} {'Pred':<6} {'Win%':>6}  Why"
    print(header)
    print("-" * len(header))
    for row, p in sorted(zip(slate, probs), key=lambda t: t[0]["gameday"]):
        winner = row["home_team"] if p >= 0.5 else row["away_team"]
        winp = p if p >= 0.5 else 1 - p
        matchup = f"{row['away_team']} @ {row['home_team']}"
        print(
            f"{str(row['gameday'])[:10]:<12} {matchup:<18} {winner:<6} {winp:6.1%}  "
            f"{row['away_style']}/{row['home_style']}; {_reasons(row, w)}"
        )


def run_pandas_pipeline() -> bool:
    from nfl_predictor.config import current_nfl_season
    from nfl_predictor.models.predict import default_week, predict_week
    from nfl_predictor.models.train import ensure_model
    from nfl_predictor.pipeline import build_feature_table

    features = build_feature_table()
    season = current_nfl_season()
    week = default_week(features, season)
    print(f"Predicting {season} week {week}")
    artifact = ensure_model(features)
    preds = predict_week(features, season=season, week=week, artifact=artifact)
    if preds.empty:
        print("No games found.")
        return True
    cols = [
        c
        for c in [
            "gameday",
            "away_team",
            "home_team",
            "away_style",
            "home_style",
            "predicted_winner",
            "predicted_win_prob",
            "reasons",
        ]
        if c in preds.columns
    ]
    print(preds[cols].to_string(index=False))
    return True


def main() -> None:
    try:
        run_pandas_pipeline()
    except ImportError as exc:
        print(f"pandas/sklearn unavailable ({exc}); using numpy fallback.\n")
        years = list(range(max(2022, current_nfl_season() - 4), current_nfl_season() + 1))
        run_numpy_fallback(years)


if __name__ == "__main__":
    main()
