"""Tunable constants: seasons, style cutoffs, weather bins, injury weights."""

from __future__ import annotations

from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
ARTIFACTS = PROJECT_ROOT / "artifacts"

FIRST_SEASON = 2018
MIN_TRAIN_GAMES = 200
FORM_WINDOW = 4
# Week 2: last 5 prior-season REG games + downweighted week 1.
# Each later week drops one prior-season game until current season stands alone.
PREV_SEASON_BLEND_GAMES = 5
FIRST_CURRENT_GAME_WEIGHT = 0.75
STARTER_SNAP_PCT = 50.0
STARTER_LOOKBACK_WEEKS = 3
LIVE_CACHE_HOURS = 12

# Rush rate vs league: +/- this gap is run-heavy / pass-heavy.
STYLE_RUSH_GAP = 0.05
LEAGUE_RUSH_RATE_DEFAULT = 0.42

# Weather bins (Fahrenheit, mph).
HEAT_TEMP_F = 85
COLD_TEMP_F = 32
WINDY_MPH = 15
SNOW_INCHES = 0.05
RAIN_INCHES = 0.05

STATUS_WEIGHTS = {
    "out": 1.0,
    "injured reserve": 1.0,
    "ir": 1.0,
    "pup": 1.0,
    "physically unable to perform": 1.0,
    "suspended": 0.9,
    "doubtful": 0.75,
    "questionable": 0.35,
    "probable": 0.10,
}

# Relative impact of a starter at each position missing time.
POSITION_WEIGHTS = {
    "QB": 1.00,
    "LT": 0.45,
    "RT": 0.40,
    "OT": 0.42,
    "T": 0.42,
    "LG": 0.30,
    "RG": 0.30,
    "G": 0.30,
    "OG": 0.30,
    "C": 0.40,
    "OL": 0.35,
    "WR": 0.28,
    "TE": 0.22,
    "RB": 0.30,
    "HB": 0.30,
    "FB": 0.10,
    "DE": 0.32,
    "DT": 0.28,
    "NT": 0.25,
    "DL": 0.28,
    "EDGE": 0.35,
    "OLB": 0.28,
    "ILB": 0.25,
    "MLB": 0.25,
    "LB": 0.26,
    "CB": 0.30,
    "S": 0.25,
    "FS": 0.25,
    "SS": 0.25,
    "SAF": 0.25,
    "DB": 0.26,
    "NB": 0.18,
    "K": 0.08,
    "P": 0.05,
    "LS": 0.04,
}

DEFAULT_POSITION_WEIGHT = 0.15
NON_STARTER_INJURY_SCALE = 0.25

PBP_COLUMNS = [
    "game_id",
    "season",
    "week",
    "posteam",
    "defteam",
    "play_type",
    "pass",
    "rush",
    "epa",
    "success",
    "yards_gained",
]

FEATURE_COLUMNS = [
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
    "home_home_pd",
    "away_away_pd",
    "home_form_pd",
    "away_form_pd",
    "home_injury_penalty",
    "away_injury_penalty",
    "home_qb_out",
    "away_qb_out",
    "home_rest",
    "away_rest",
    "rest_diff",
    "is_indoor",
    "temp",
    "wind",
    "is_heat",
    "is_cold",
    "is_snow",
    "is_rain",
    "is_windy",
    "home_pass_weather",
    "away_pass_weather",
    "home_prior_games",
    "away_prior_games",
]

FEATURE_LABELS = {
    "home_off_epa": "Home offense EPA/play",
    "away_off_epa": "Away offense EPA/play",
    "home_off_rush_epa": "Home rush EPA",
    "away_off_rush_epa": "Away rush EPA",
    "home_off_pass_epa": "Home pass EPA",
    "away_off_pass_epa": "Away pass EPA",
    "home_rush_rate": "Home rush rate",
    "away_rush_rate": "Away rush rate",
    "home_def_rush_epa": "Home EPA allowed vs rush",
    "away_def_rush_epa": "Away EPA allowed vs rush",
    "home_def_pass_epa": "Home EPA allowed vs pass",
    "away_def_pass_epa": "Away EPA allowed vs pass",
    "home_ppg": "Home points per game",
    "away_ppg": "Away points per game",
    "home_papg": "Home points allowed",
    "away_papg": "Away points allowed",
    "rush_edge_net": "Net rush matchup edge (home)",
    "pass_edge_net": "Net pass matchup edge (home)",
    "home_home_pd": "Home team home-field point diff",
    "away_away_pd": "Away team road point diff",
    "home_form_pd": "Home blended form point diff",
    "away_form_pd": "Away blended form point diff",
    "home_injury_penalty": "Home injury penalty",
    "away_injury_penalty": "Away injury penalty",
    "home_qb_out": "Home QB out/doubtful",
    "away_qb_out": "Away QB out/doubtful",
    "home_rest": "Home rest days",
    "away_rest": "Away rest days",
    "rest_diff": "Rest advantage (home)",
    "is_indoor": "Indoor / dome",
    "temp": "Temperature (F)",
    "wind": "Wind (mph)",
    "is_heat": "Hot weather",
    "is_cold": "Cold weather",
    "is_snow": "Snow",
    "is_rain": "Rain",
    "is_windy": "Windy",
    "home_pass_weather": "Home pass-heavy in bad weather",
    "away_pass_weather": "Away pass-heavy in bad weather",
    "home_prior_games": "Home games in sample",
    "away_prior_games": "Away games in sample",
}

# Home stadiums used for upcoming-game weather forecasts.
TEAM_STADIUMS: dict[str, dict] = {
    "ARI": {"name": "State Farm Stadium", "lat": 33.5276, "lon": -112.2626, "tz": "America/Phoenix", "roof": "retractable"},
    "ATL": {"name": "Mercedes-Benz Stadium", "lat": 33.7553, "lon": -84.4006, "tz": "America/New_York", "roof": "retractable"},
    "BAL": {"name": "M&T Bank Stadium", "lat": 39.2780, "lon": -76.6227, "tz": "America/New_York", "roof": "outdoor"},
    "BUF": {"name": "Highmark Stadium", "lat": 42.7738, "lon": -78.7870, "tz": "America/New_York", "roof": "outdoor"},
    "CAR": {"name": "Bank of America Stadium", "lat": 35.2258, "lon": -80.8528, "tz": "America/New_York", "roof": "outdoor"},
    "CHI": {"name": "Soldier Field", "lat": 41.8623, "lon": -87.6167, "tz": "America/Chicago", "roof": "outdoor"},
    "CIN": {"name": "Paycor Stadium", "lat": 39.0954, "lon": -84.5160, "tz": "America/New_York", "roof": "outdoor"},
    "CLE": {"name": "Huntington Bank Field", "lat": 41.5061, "lon": -81.6995, "tz": "America/New_York", "roof": "outdoor"},
    "DAL": {"name": "AT&T Stadium", "lat": 32.7478, "lon": -97.0928, "tz": "America/Chicago", "roof": "retractable"},
    "DEN": {"name": "Empower Field at Mile High", "lat": 39.7439, "lon": -105.0201, "tz": "America/Denver", "roof": "outdoor"},
    "DET": {"name": "Ford Field", "lat": 42.3400, "lon": -83.0456, "tz": "America/New_York", "roof": "dome"},
    "GB": {"name": "Lambeau Field", "lat": 44.5013, "lon": -88.0622, "tz": "America/Chicago", "roof": "outdoor"},
    "HOU": {"name": "NRG Stadium", "lat": 29.6847, "lon": -95.4107, "tz": "America/Chicago", "roof": "retractable"},
    "IND": {"name": "Lucas Oil Stadium", "lat": 39.7601, "lon": -86.1639, "tz": "America/Indiana/Indianapolis", "roof": "retractable"},
    "JAX": {"name": "EverBank Stadium", "lat": 30.3239, "lon": -81.6373, "tz": "America/New_York", "roof": "outdoor"},
    "KC": {"name": "GEHA Field at Arrowhead", "lat": 39.0489, "lon": -94.4839, "tz": "America/Chicago", "roof": "outdoor"},
    "LAC": {"name": "SoFi Stadium", "lat": 33.9535, "lon": -118.3392, "tz": "America/Los_Angeles", "roof": "dome"},
    "LAR": {"name": "SoFi Stadium", "lat": 33.9535, "lon": -118.3392, "tz": "America/Los_Angeles", "roof": "dome"},
    "LV": {"name": "Allegiant Stadium", "lat": 36.0909, "lon": -115.1833, "tz": "America/Los_Angeles", "roof": "dome"},
    "MIA": {"name": "Hard Rock Stadium", "lat": 25.9580, "lon": -80.2389, "tz": "America/New_York", "roof": "outdoor"},
    "MIN": {"name": "U.S. Bank Stadium", "lat": 44.9738, "lon": -93.2575, "tz": "America/Chicago", "roof": "dome"},
    "NE": {"name": "Gillette Stadium", "lat": 42.0909, "lon": -71.2643, "tz": "America/New_York", "roof": "outdoor"},
    "NO": {"name": "Caesars Superdome", "lat": 29.9511, "lon": -90.0812, "tz": "America/Chicago", "roof": "dome"},
    "NYG": {"name": "MetLife Stadium", "lat": 40.8135, "lon": -74.0745, "tz": "America/New_York", "roof": "outdoor"},
    "NYJ": {"name": "MetLife Stadium", "lat": 40.8135, "lon": -74.0745, "tz": "America/New_York", "roof": "outdoor"},
    "PHI": {"name": "Lincoln Financial Field", "lat": 39.9008, "lon": -75.1675, "tz": "America/New_York", "roof": "outdoor"},
    "PIT": {"name": "Acrisure Stadium", "lat": 40.4468, "lon": -80.0158, "tz": "America/New_York", "roof": "outdoor"},
    "SEA": {"name": "Lumen Field", "lat": 47.5952, "lon": -122.3316, "tz": "America/Los_Angeles", "roof": "outdoor"},
    "SF": {"name": "Levi's Stadium", "lat": 37.4033, "lon": -121.9694, "tz": "America/Los_Angeles", "roof": "outdoor"},
    "TB": {"name": "Raymond James Stadium", "lat": 27.9759, "lon": -82.5033, "tz": "America/New_York", "roof": "outdoor"},
    "TEN": {"name": "Nissan Stadium", "lat": 36.1665, "lon": -86.7713, "tz": "America/Chicago", "roof": "outdoor"},
    "WAS": {"name": "Northwest Stadium", "lat": 38.9078, "lon": -76.8645, "tz": "America/New_York", "roof": "outdoor"},
}


def current_nfl_season(today: date | None = None) -> int:
    """Season year (the calendar year the regular season starts in)."""
    today = today or date.today()
    if today.month >= 8:
        return today.year
    if today.month <= 2:
        return today.year - 1
    return today.year


def season_years(end_season: int | None = None, first: int = FIRST_SEASON) -> list[int]:
    end_season = end_season or current_nfl_season()
    return list(range(first, end_season + 1))


def classify_style(rush_rate: float, league_rush_rate: float = LEAGUE_RUSH_RATE_DEFAULT) -> str:
    if rush_rate >= league_rush_rate + STYLE_RUSH_GAP:
        return "run-heavy"
    if rush_rate <= league_rush_rate - STYLE_RUSH_GAP:
        return "pass-heavy"
    return "balanced"
