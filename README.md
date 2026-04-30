# NBA Betting Model

Automated NBA moneyline edge-finding model. Retrains daily via GitHub Actions
and commits fresh picks to `predictions.json`.

## What it does

- Pulls 4 seasons of NBA game data (~4,400 games) from BallDontLie
- Builds walk-forward rolling features (no data leakage)
- Computes per-game offensive/defensive/net rating from scores
- Trains a Logistic Regression + XGBoost ensemble on game-only features
- Augments current predictions with player stats and injury data
- Compares model probability to Pinnacle (sharpest book) no-vig line
- Flags bets where model edge > 3% and grades them A/B/C
- Tracks Closing Line Value (CLV) for real edge validation

## Stack

- **Game data**: BallDontLie API (multi-season)
- **Player data**: api.server.nbaapi.com (free, current season)
- **Injuries**: BallDontLie injury endpoint
- **Odds**: The Odds API + OpticOdds (OddsJam) — multi-book, Pinnacle no-vig
- **Models**: Logistic Regression + XGBoost (moneyline)
- **Sizing**: Quarter-Kelly criterion

## Required GitHub Secrets

| Secret | Where to get it |
|---|---|
| `BALLDONTLIE_API_KEY` | balldontlie.io — upgrade to paid for multi-season history |
| `ODDS_API_KEY` | the-odds-api.com |
| `OPTICODDS_KEY` | opticodds.com (formerly OddsJam) |

## Setup

```bash
pip install -r requirements.txt
python nba_betting_model.py
```

## Automation

GitHub Actions runs daily at 12:00 UTC (`retrain.yml`).
Manual trigger: **Actions → Retrain NBA Model → Run workflow**.

## Output

`predictions.json` — one entry per upcoming game:

```json
{
  "matchup": "BOS vs NYK",
  "home_win_prob": 0.617,
  "away_win_prob": 0.383,
  "has_live_odds": true,
  "has_sharp_novig": true,
  "recommendations": [
    {
      "type": "ML",
      "side": "HOME",
      "team": "BOS",
      "odds": -148,
      "best_book": "draftkings",
      "model_prob": 0.617,
      "novig_prob": 0.571,
      "edge": 0.046,
      "kelly_pct": 2.1,
      "grade": "B",
      "label": "BOS ML"
    }
  ]
}
```

`backtest.json` — walk-forward backtest results (honest, no lookahead):

```json
{
  "test_games": 1100,
  "accuracy": 0.672,
  "auc": 0.718,
  "brier": 0.231,
  "flat_bet": { "bets_placed": 380, "win_rate": 0.571, "roi_pct": 3.2 }
}
```

`bet_log.json` — every flagged bet with CLV tracking fields.

## Key Config

| Variable | Default | Notes |
|---|---|---|
| `TRAIN_SEASONS` | `[2021,2022,2023,2024]` | Requires paid BDL plan for history |
| `MIN_EDGE` | `0.03` | Raise to 0.05+ to be more selective |
| `KELLY_FRACTION` | `0.25` | Never go above 0.5 |
| `SHARP_BOOKS` | `["pinnacle",...]` | Priority order for no-vig baseline |

## Edge Grading

| Grade | Edge vs no-vig |
|---|---|
| A | > 6% |
| B | 4 – 6% |
| C | 3 – 4% |

## CLV Tracking

After each game resolves, manually update `bet_log.json`:

```json
{
  "result": "W",
  "closing_odds": -160,
  "clv": 12
}
```

Positive CLV means you got better odds than the market closed at — the
most reliable long-term signal that your model has a genuine edge.
