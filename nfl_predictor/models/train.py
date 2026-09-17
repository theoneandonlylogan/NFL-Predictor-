"""Rolling-week logistic regression and persisted model artifact."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nfl_predictor import numpy_compat as _numpy_compat  # noqa: F401  # patch RNG before sklearn

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_predictor.config import ARTIFACTS, FEATURE_COLUMNS, MIN_TRAIN_GAMES

MODEL_PATH = ARTIFACTS / "nfl_win_model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"


def _pipeline() -> Pipeline:
    try:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        imputer = SimpleImputer(strategy="median")
    return Pipeline(
        steps=[
            ("impute", imputer),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=1000, C=0.8, solver="lbfgs"),
            ),
        ]
    )


def _labeled(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    df = df[df["home_win"].notna()]
    df = df[~df["tie"].fillna(False).astype(bool)]
    df["home_win"] = df["home_win"].astype(int)
    for col in _available_features(df):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLUMNS if c in df.columns]


def rolling_backtest(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Expanding window: train on all prior weeks, predict the next week."""
    df = _labeled(features).sort_values(["season", "week", "game_id"])
    cols = _available_features(df)
    df["season_week"] = df["season"].astype(int) * 100 + df["week"].astype(int)

    preds: list[pd.DataFrame] = []
    for sw in sorted(df["season_week"].unique()):
        train = df[df["season_week"] < sw]
        test = df[df["season_week"] == sw]
        if len(train) < MIN_TRAIN_GAMES or test.empty:
            continue
        pipe = _pipeline()
        pipe.fit(train[cols], train["home_win"])
        chunk = test.copy()
        chunk["pred_proba"] = pipe.predict_proba(chunk[cols])[:, 1]
        chunk["pred_home_win"] = (chunk["pred_proba"] >= 0.5).astype(int)
        preds.append(chunk)

    if not preds:
        raise ValueError("Not enough completed games to backtest.")

    out = pd.concat(preds, ignore_index=True)
    y = out["home_win"].astype(int)
    proba = out["pred_proba"]
    pred = out["pred_home_win"]

    # Baseline: better prior point differential (home_ppg - home_papg vs away).
    home_net = out["home_ppg"] - out["home_papg"]
    away_net = out["away_ppg"] - out["away_papg"]
    baseline = (home_net.fillna(0) >= away_net.fillna(0)).astype(int)

    metrics = {
        "games": int(len(out)),
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y, proba)),
        "baseline_point_diff_accuracy": float(accuracy_score(y, baseline)),
        "home_win_rate": float(y.mean()),
        "seasons": {
            "min": int(out["season"].min()),
            "max": int(out["season"].max()),
        },
    }
    return out, metrics


def coefficient_table(model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    clf: LogisticRegression = model.named_steps["model"]
    coefs = clf.coef_[0]
    table = pd.DataFrame({"feature": feature_cols, "coefficient": coefs})
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False)


def train_final_model(features: pd.DataFrame, save: bool = True) -> dict:
    df = _labeled(features)
    cols = _available_features(df)
    pipe = _pipeline()
    pipe.fit(df[cols], df["home_win"])
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": pipe,
        "feature_columns": cols,
        "n_games": int(len(df)),
        "seasons": [int(df["season"].min()), int(df["season"].max())],
    }
    if save:
        joblib.dump(artifact, MODEL_PATH)
    return artifact


def load_model(path: Path | None = None) -> dict:
    path = path or MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Run train_final_model first.")
    return joblib.load(path)


def ensure_model(features: pd.DataFrame) -> dict:
    if MODEL_PATH.exists():
        return load_model()
    return train_final_model(features, save=True)


def save_metrics(metrics: dict, path: Path | None = None) -> None:
    path = path or METRICS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def reliability_bins(backtest: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    df = backtest.copy()
    df["bin"] = pd.cut(df["pred_proba"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    grouped = df.groupby("bin", observed=False).agg(
        predicted=("pred_proba", "mean"),
        actual=("home_win", "mean"),
        count=("home_win", "size"),
    )
    return grouped.reset_index()
