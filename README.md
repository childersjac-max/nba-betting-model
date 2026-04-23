# NBA Betting Model

Automated NBA betting model covering moneyline, spread, totals, and player props.  
Retrains daily via GitHub Actions and commits fresh predictions to `predictions.json`.

## Stack
- **Data**: `sportsreference` (basketball-reference.com)
- **Models**: Logistic Regression + XGBoost (moneyline), Ridge Regression (spread/totals)
- **Sizing**: Fractional Kelly criterion (quarter-Kelly)

## Setup

```bash
pip install -r requirements.txt
python nba_betting_model.py
```

## Updating upcoming games

Edit the `UPCOMING_GAMES` list in `nba_betting_model.py`:

```python
UPCOMING_GAMES = [
    ("BOS", "NYK", -155, -4.5, 224.5),  # home, away, ML odds, spread, total
    ("OKC", "DEN", -180, -5.5, 221.0),
]
```

Use 3-letter sportsreference abbreviations (same as basketball-reference.com).

## Automation

The GitHub Actions workflow (`.github/workflows/retrain.yml`) runs daily at 12:00 UTC.  
Trigger a manual run anytime: **Actions → Retrain NBA Model → Run workflow**.

## Output

`predictions.json` is written to the repo root after every run. Example:

```json
{
  "season": 2025,
  "generated_at": "2025-04-23T12:00:00Z",
  "games": [
    {
      "matchup": "BOS vs NYK",
      "moneyline": { "home_win_prob": 0.612, "home_bet": { "edge": 0.047, "bet": true } },
      "spread":    { "predicted_margin": 4.8 },
      "totals":    { "predicted_total": 223.1, "over_bet": { "edge": -0.01, "bet": false } }
    }
  ]
}
```

## Tuning

| Config var | Location | Default | Notes |
|---|---|---|---|
| `MIN_EDGE` | top of script | `0.03` | Raise to 0.05+ to be more selective |
| `KELLY_FRACTION` | top of script | `0.25` | Never go above 0.5 |
| `SEASON` | top of script | `2025` | Year the season ends in |
