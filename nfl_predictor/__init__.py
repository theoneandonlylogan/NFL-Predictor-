"""NFL weekly matchup predictor."""

from nfl_predictor import numpy_compat as _numpy_compat  # noqa: F401
from nfl_predictor.config import current_nfl_season, season_years
from nfl_predictor.pipeline import build_feature_table, load_week_matchups

__all__ = [
    "build_feature_table",
    "current_nfl_season",
    "load_week_matchups",
    "season_years",
]
