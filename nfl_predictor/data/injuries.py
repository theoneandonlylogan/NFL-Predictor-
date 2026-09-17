"""Starter-weighted injury penalties by team-week."""

from __future__ import annotations

import re

import pandas as pd

from nfl_predictor.config import (
    DEFAULT_POSITION_WEIGHT,
    NON_STARTER_INJURY_SCALE,
    POSITION_WEIGHTS,
    STARTER_LOOKBACK_WEEKS,
    STARTER_SNAP_PCT,
    STATUS_WEIGHTS,
)

_SUFFIXES = re.compile(r"\s+(jr\.?|sr\.?|ii|iii|iv|v)$", re.I)


def normalize_name(value: object) -> str:
    text = str(value or "").strip().lower()
    text = _SUFFIXES.sub("", text)
    return re.sub(r"[^a-z0-9]", "", text)


def _status_weight(status: object) -> float:
    if status is None or (isinstance(status, float) and pd.isna(status)):
        return 0.0
    key = str(status).strip().lower()
    if key in STATUS_WEIGHTS:
        return STATUS_WEIGHTS[key]
    for token, weight in STATUS_WEIGHTS.items():
        if token in key:
            return weight
    return 0.0


def _position_weight(position: object) -> float:
    pos = str(position or "").strip().upper()
    if pos in POSITION_WEIGHTS:
        return POSITION_WEIGHTS[pos]
    # WR1 / CB2 style labels
    base = re.sub(r"\d+$", "", pos)
    return POSITION_WEIGHTS.get(base, DEFAULT_POSITION_WEIGHT)


def _starter_keys(snaps: pd.DataFrame) -> set[tuple[int, int, str, str]]:
    """(season, week, team, normalized_name) for high-snap players in the lookback window."""
    if snaps is None or snaps.empty:
        return set()

    df = snaps.copy()
    off = pd.to_numeric(df.get("offense_pct"), errors="coerce").fillna(0)
    deff = pd.to_numeric(df.get("defense_pct"), errors="coerce").fillna(0)
    df["snap_pct"] = pd.concat([off, deff], axis=1).max(axis=1)
    df["name_key"] = df.get("player", pd.Series("", index=df.index)).map(normalize_name)
    df["team"] = df["team"].astype(str)
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.dropna(subset=["season", "week", "team", "name_key"])
    df = df[df["name_key"] != ""]

    starters: set[tuple[int, int, str, str]] = set()
    grouped = df.groupby(["season", "team"], sort=False)
    for (season, team), group in grouped:
        group = group.sort_values("week")
        weeks = sorted(group["week"].unique())
        for week in weeks:
            lookback = group[
                (group["week"] < week) & (group["week"] >= week - STARTER_LOOKBACK_WEEKS)
            ]
            if lookback.empty:
                lookback = group[group["week"] == week]
            avg = lookback.groupby("name_key")["snap_pct"].mean()
            for name_key, pct in avg.items():
                if pct >= STARTER_SNAP_PCT:
                    starters.add((int(season), int(week), str(team), str(name_key)))
    return starters


def compute_injury_penalties(injuries: pd.DataFrame, snaps: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per team-week with injury_penalty, n_starters_out, qb_out."""
    empty = pd.DataFrame(columns=["season", "week", "team", "injury_penalty", "n_starters_out", "qb_out"])
    if injuries is None or injuries.empty:
        return empty

    df = injuries.copy()
    name_col = "full_name" if "full_name" in df.columns else "player"
    df["name_key"] = df.get(name_col, pd.Series("", index=df.index)).map(normalize_name)
    status_col = "report_status" if "report_status" in df.columns else "injury_status"
    df["status_w"] = df.get(status_col, pd.Series(index=df.index)).map(_status_weight)
    df = df[df["status_w"] > 0]
    if df.empty:
        return empty

    df["pos_w"] = df.get("position", pd.Series(index=df.index)).map(_position_weight)
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["team"] = df["team"].astype(str)
    df = df.dropna(subset=["season", "week", "team"])

    starters = _starter_keys(snaps)
    is_starter = [
        (int(season), int(week), str(team), str(name)) in starters
        or str(pos).strip().upper() == "QB"
        for season, week, team, name, pos in zip(
            df["season"], df["week"], df["team"], df["name_key"], df.get("position", pd.Series("", index=df.index))
        )
    ]
    df["starter_scale"] = [1.0 if flag else NON_STARTER_INJURY_SCALE for flag in is_starter]
    df["player_penalty"] = df["status_w"] * df["pos_w"] * df["starter_scale"]
    df["starter_out"] = [
        float(scale >= 1.0 and status >= 0.75) for scale, status in zip(df["starter_scale"], df["status_w"])
    ]
    df["qb_flag"] = [
        float(str(pos).strip().upper() == "QB" and status >= 0.75)
        for pos, status in zip(df.get("position", pd.Series("", index=df.index)), df["status_w"])
    ]

    out = (
        df.groupby(["season", "week", "team"], as_index=False)
        .agg(
            injury_penalty=("player_penalty", "sum"),
            n_starters_out=("starter_out", "sum"),
            qb_out=("qb_flag", "max"),
        )
    )
    return out
