# NFL Weekly Matchup Predictor

Predict NFL week-to-week winners from team style (run vs pass), defensive matchups, home/away form, weather, and starter injuries.

This is a practice modeling project. NFL games are noisy; the train notebook reports honest backtest accuracy rather than promising a fixed win rate.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install -r requirements.txt
# macOS / Linux
# source .venv/bin/activate
# python -m pip install -r requirements.txt
```

Then open the notebooks in Jupyter:

```bash
python -m jupyter notebook notebooks
```

## Notebooks

1. **`01_team_profiles.ipynb`** — run/pass mix, defense vs run/pass, home/away splits, injury penalty
2. **`02_train_and_backtest.ipynb`** — rolling-week logistic regression, accuracy, which features matter
3. **`03_predict_week.ipynb`** — pick a season and week, get win probabilities and the main reasons

First data pull downloads nflverse play-by-play and can take several minutes. Later runs use `data/raw/`.

## How a prediction is built

Each game becomes one row of **pre-kickoff** features (prior weeks only):

- Offensive style: rush rate, rush/pass EPA, scoring
- Defense vs rush and vs pass
- Matchup edges (your rush offense vs their rush defense, and the reverse)
- Home/away point differential and rest
- Weather bins (heat, cold/snow, rain, wind, indoor) plus a pass-heavy-in-bad-weather term
- Starter-weighted injury penalty (QB counts much more than a backup)

A logistic regression maps those features to **P(home team wins)**.

## Data sources

- nflverse GitHub releases (schedules, play-by-play, injuries, snap counts)
- [Open-Meteo](https://open-meteo.com/) forecasts for upcoming outdoor games (no API key)

## Project layout

```
nfl_predictor/     reusable ingest, features, and model code
notebooks/         weekly UI
data/raw/          local cache (gitignored)
artifacts/         saved model
```
