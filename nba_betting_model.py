"""
NBA Betting Model
=================
Covers: Moneyline · Spread · Over/Under · Player Props
Data:   sportsreference (sports-reference.com)
Models: Logistic Regression, Ridge Regression, XGBoost

Designed to run headlessly via GitHub Actions.
Outputs predictions.json to repo root on each run.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

# sportsreference imports
from sportsreference.nba.teams import Teams
from sportsreference.nba.schedule import Schedule
from sportsreference.nba.boxscore import Boxscores
from sportsreference.nba.roster import Player

# sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# XGBoost
import xgboost as xgb


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these to control which games get predicted each run
# ─────────────────────────────────────────────────────────────────────────────

SEASON = 2025  # year the season ends in (2024-25 → 2025)

# Games to predict each run — update this list as the schedule changes.
# Format: (home_abbr, away_abbr, ml_odds_home, spread_line, total_line)
UPCOMING_GAMES = [
    ("BOS", "NYK", -155, -4.5, 224.5),
    ("OKC", "DEN", -180, -5.5, 221.0),
    ("SAS", "HOU", -130, -3.0, 218.5),
]

# Minimum model edge required to flag a bet (3% = 0.03)
MIN_EDGE = 0.03

# Kelly fraction — 0.25 = quarter Kelly (conservative, recommended)
KELLY_FRACTION = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def fetch_team_game_logs(season: int = SEASON) -> pd.DataFrame:
    print(f"[Data] Fetching team schedules for {season - 1}-{str(season)[-2:]} season...")
    rows = []
    for team in Teams(season):
        abbr = team.abbreviation
        try:
            schedule = Schedule(abbr, year=season)
            for game in schedule:
                row = {
                    "team":       abbr,
                    "date":       game.date,
                    "opponent":   game.opponent_abbr,
                    "home":       1 if game.location == "Home" else 0,
                    "pts_for":    game.points,
                    "pts_against":game.opp_points,
                    "fg_pct":     game.fg_percentage,
                    "fg3_pct":    game.three_point_field_goal_percentage,
                    "ft_pct":     game.free_throw_percentage,
                    "oreb":       game.offensive_rebounds,
                    "dreb":       game.defensive_rebounds,
                    "ast":        game.assists,
                    "tov":        game.turnovers,
                    "stl":        game.steals,
                    "blk":        game.blocks,
                    "pace":       game.pace,
                    "off_rtg":    game.offensive_rating,
                    "def_rtg":    game.defensive_rating,
                    "result":     1 if game.result == "W" else 0,
                }
                rows.append(row)
        except Exception as e:
            print(f"  Warning: could not fetch {abbr}: {e}")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["team", "date"]).reset_index(drop=True)
    df["point_diff"] = df["pts_for"] - df["pts_against"]
    df["total_pts"]  = df["pts_for"] + df["pts_against"]
    return df


def compute_rolling_features(df: pd.DataFrame, windows: list = [5, 10]) -> pd.DataFrame:
    stat_cols = [
        "pts_for", "pts_against", "point_diff", "total_pts",
        "fg_pct", "fg3_pct", "ft_pct",
        "oreb", "dreb", "ast", "tov",
        "off_rtg", "def_rtg", "pace",
    ]
    all_frames = []
    for team, grp in df.groupby("team"):
        grp = grp.copy().sort_values("date")
        for w in windows:
            for col in stat_cols:
                grp[f"{col}_roll{w}"] = (
                    grp[col].shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
                )
        all_frames.append(grp)
    return pd.concat(all_frames).sort_values(["date", "team"]).reset_index(drop=True)


def build_game_level_features(df: pd.DataFrame) -> pd.DataFrame:
    home = df[df["home"] == 1].copy()
    away = df[df["home"] == 0].copy()
    roll_cols = [c for c in df.columns if "_roll" in c]
    base_cols = ["date", "team", "opponent", "point_diff", "total_pts", "result"]

    home_feat = home[base_cols + roll_cols].copy()
    home_feat.columns = (
        ["date", "home_team", "away_team", "point_diff", "total_pts", "home_win"]
        + [f"home_{c}" for c in roll_cols]
    )
    away_feat = away[["date", "team"] + roll_cols].copy()
    away_feat.columns = ["date", "away_team_check"] + [f"away_{c}" for c in roll_cols]

    games = home_feat.merge(
        away_feat,
        left_on=["date", "away_team"],
        right_on=["date", "away_team_check"],
        how="inner",
    ).drop(columns=["away_team_check"])

    for col in roll_cols:
        games[f"diff_{col}"] = games[f"home_{col}"] - games[f"away_{col}"]

    return games.dropna().reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ELO
# ─────────────────────────────────────────────────────────────────────────────

def compute_elo(df: pd.DataFrame, k: int = 20, home_adv: float = 100) -> pd.DataFrame:
    elo = {team: 1500.0 for team in
           pd.concat([df["home_team"], df["away_team"]]).unique()}
    home_elos, away_elos = [], []

    for _, row in df.sort_values("date").iterrows():
        h, a = row["home_team"], row["away_team"]
        exp_h = 1 / (1 + 10 ** ((elo[a] - (elo[h] + home_adv)) / 400))
        actual_h = row["home_win"]
        home_elos.append(elo[h])
        away_elos.append(elo[a])
        margin = abs(row["point_diff"])
        k_margin = k * np.log1p(margin) * (2.2 / ((margin * 0.001) + 2.2))
        elo[h] += k_margin * (actual_h - exp_h)
        elo[a] += k_margin * (exp_h - actual_h)

    df = df.sort_values("date").copy()
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["elo_win_prob"] = 1 / (1 + 10 ** (-df["elo_diff"] / 400))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE SETS
# ─────────────────────────────────────────────────────────────────────────────

MONEYLINE_FEATURES = [
    "elo_diff", "elo_win_prob",
    "diff_off_rtg_roll10", "diff_def_rtg_roll10",
    "diff_point_diff_roll10", "diff_pace_roll10",
    "diff_fg_pct_roll5", "diff_fg3_pct_roll5",
    "diff_ast_roll5", "diff_tov_roll5",
]

SPREAD_FEATURES = [
    "elo_diff",
    "diff_off_rtg_roll10", "diff_def_rtg_roll10",
    "diff_point_diff_roll10", "diff_pace_roll10",
    "diff_fg_pct_roll5", "diff_fg3_pct_roll5",
    "diff_oreb_roll5", "diff_tov_roll5",
    "home_off_rtg_roll10", "home_def_rtg_roll10",
    "away_off_rtg_roll10", "away_def_rtg_roll10",
]

TOTALS_FEATURES = [
    "home_off_rtg_roll10", "home_def_rtg_roll10", "home_pace_roll10",
    "away_off_rtg_roll10", "away_def_rtg_roll10", "away_pace_roll10",
    "home_total_pts_roll10", "away_total_pts_roll10",
    "diff_pace_roll10",
    "home_fg_pct_roll5", "away_fg_pct_roll5",
    "home_fg3_pct_roll5", "away_fg3_pct_roll5",
]


def filter_available(df, cols):
    return [c for c in cols if c in df.columns]


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODELS
# ─────────────────────────────────────────────────────────────────────────────

def train_moneyline_model(games):
    feats = filter_available(games, MONEYLINE_FEATURES)
    X, y = games[feats].values, games["home_win"].values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=1000), cv=5, method="sigmoid"
        )),
    ])
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", use_label_encoder=False, random_state=42,
    )
    pipe.fit(X, y)
    xgb_clf.fit(X, y)
    tscv = TimeSeriesSplit(n_splits=5)
    lr_auc  = cross_val_score(pipe,    X, y, cv=tscv, scoring="roc_auc").mean()
    xgb_auc = cross_val_score(xgb_clf, X, y, cv=tscv, scoring="roc_auc").mean()
    print(f"[Moneyline] LR ROC-AUC={lr_auc:.3f}  XGB ROC-AUC={xgb_auc:.3f}")
    return {"lr": pipe, "xgb": xgb_clf, "features": feats}


def train_spread_model(games):
    feats = filter_available(games, SPREAD_FEATURES)
    X, y = games[feats].values, games["point_diff"].values
    pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0))])
    pipe.fit(X, y)
    mae = -cross_val_score(pipe, X, y, cv=TimeSeriesSplit(5),
                           scoring="neg_mean_absolute_error").mean()
    print(f"[Spread]    Ridge MAE={mae:.2f} pts")
    return {"model": pipe, "features": feats}


def train_totals_model(games):
    feats = filter_available(games, TOTALS_FEATURES)
    X, y = games[feats].values, games["total_pts"].values
    pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0))])
    pipe.fit(X, y)
    mae = -cross_val_score(pipe, X, y, cv=TimeSeriesSplit(5),
                           scoring="neg_mean_absolute_error").mean()
    print(f"[Totals]    Ridge MAE={mae:.2f} pts")
    return {"model": pipe, "features": feats}


# ─────────────────────────────────────────────────────────────────────────────
# 5. BETTING EDGE
# ─────────────────────────────────────────────────────────────────────────────

def american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def kelly_fraction(win_prob: float, odds: int) -> float:
    b = odds / 100 if odds > 0 else 100 / abs(odds)
    q = 1 - win_prob
    return max(0.0, ((b * win_prob - q) / b) * KELLY_FRACTION)


def evaluate_bet(model_prob: float, odds: int) -> dict:
    implied = american_to_prob(odds)
    edge    = model_prob - implied
    kelly   = kelly_fraction(model_prob, odds) if edge > MIN_EDGE else 0.0
    return {
        "model_prob":   round(model_prob, 3),
        "implied_prob": round(implied, 3),
        "edge":         round(edge, 3),
        "kelly_pct":    round(kelly * 100, 1),
        "bet":          edge > MIN_EDGE,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class NBABettingModel:

    def __init__(self, season: int = SEASON):
        self.season = season
        self.models = {}
        self.games  = None
        self.raw    = None

    def fit(self):
        print("=" * 55)
        print("NBA Betting Model — Training")
        print("=" * 55)
        raw    = fetch_team_game_logs(self.season)
        rolled = compute_rolling_features(raw)
        games  = build_game_level_features(rolled)
        games  = compute_elo(games)
        self.games = games
        self.raw   = rolled
        self.models["moneyline"] = train_moneyline_model(games)
        self.models["spread"]    = train_spread_model(games)
        self.models["totals"]    = train_totals_model(games)
        print("✓ Models trained.\n")

    def _team_feats(self, team: str, is_home: bool) -> dict:
        prefix  = "home" if is_home else "away"
        team_df = self.raw[self.raw["team"] == team].sort_values("date")
        if team_df.empty:
            raise ValueError(f"Team '{team}' not found.")
        latest = team_df.iloc[-1]
        return {f"{prefix}_{c}": latest[c]
                for c in team_df.columns if "_roll" in c}

    def predict_game(self, home: str, away: str,
                     ml_odds_home: int = None,
                     spread_line: float = None, spread_odds: int = -110,
                     total_line: float = None,  total_odds: int = -110) -> dict:

        feat_row = {**self._team_feats(home, True), **self._team_feats(away, False)}

        h_elo = self.games[self.games["home_team"] == home].sort_values("date").iloc[-1]["home_elo"]
        a_elo = self.games[self.games["away_team"] == away].sort_values("date").iloc[-1]["away_elo"]
        feat_row["elo_diff"]     = h_elo - a_elo
        feat_row["elo_win_prob"] = 1 / (1 + 10 ** (-feat_row["elo_diff"] / 400))

        roll_suffixes = {k[len("home_"):] for k in feat_row if k.startswith("home_") and "_roll" in k}
        for suf in roll_suffixes:
            feat_row[f"diff_{suf}"] = feat_row.get(f"home_{suf}", 0) - feat_row.get(f"away_{suf}", 0)

        result = {"matchup": f"{home} vs {away}", "generated_at": datetime.utcnow().isoformat() + "Z"}

        # Moneyline
        ml   = self.models["moneyline"]
        ml_x = np.array([[feat_row.get(f, 0) for f in ml["features"]]])
        hwp  = float(ml["lr"].predict_proba(ml_x)[0][1])
        result["moneyline"] = {
            "home_win_prob": round(hwp, 3),
            "away_win_prob": round(1 - hwp, 3),
        }
        if ml_odds_home is not None:
            result["moneyline"]["home_bet"] = evaluate_bet(hwp,       ml_odds_home)
            result["moneyline"]["away_bet"] = evaluate_bet(1 - hwp,  -ml_odds_home)

        # Spread
        sp   = self.models["spread"]
        sp_x = np.array([[feat_row.get(f, 0) for f in sp["features"]]])
        pred_diff = float(sp["model"].predict(sp_x)[0])
        result["spread"] = {"predicted_margin": round(pred_diff, 2)}
        if spread_line is not None:
            cover_prob = min(0.65, max(0.35, 0.5 + (pred_diff - spread_line) * 0.02))
            result["spread"]["home_covers_prob"] = round(cover_prob, 3)
            result["spread"]["home_bet"] = evaluate_bet(cover_prob, spread_odds)

        # Totals
        tot   = self.models["totals"]
        tot_x = np.array([[feat_row.get(f, 0) for f in tot["features"]]])
        pred_total = float(tot["model"].predict(tot_x)[0])
        result["totals"] = {"predicted_total": round(pred_total, 1)}
        if total_line is not None:
            over_prob = min(0.65, max(0.35, 0.5 + (pred_total - total_line) * 0.02))
            result["totals"]["over_prob"]  = round(over_prob, 3)
            result["totals"]["over_bet"]   = evaluate_bet(over_prob,     total_odds)
            result["totals"]["under_bet"]  = evaluate_bet(1 - over_prob, total_odds)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = NBABettingModel(season=SEASON)
    model.fit()

    all_predictions = {
        "season":       SEASON,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "games":        [],
    }

    for home, away, ml_odds, spread, total in UPCOMING_GAMES:
        try:
            pred = model.predict_game(
                home=home, away=away,
                ml_odds_home=ml_odds,
                spread_line=spread,
                total_line=total,
            )
            all_predictions["games"].append(pred)

            # Console summary
            ml  = pred["moneyline"]
            sp  = pred["spread"]
            tot = pred["totals"]
            print(f"\n{'─'*50}")
            print(f"  {pred['matchup']}")
            print(f"  ML   : home {ml['home_win_prob']:.1%} | away {ml['away_win_prob']:.1%}")
            if "home_bet" in ml:
                b = ml["home_bet"]
                print(f"  ML bet (home): {'✅ BET' if b['bet'] else '❌ skip'} edge={b['edge']:+.1%} Kelly={b['kelly_pct']:.1f}%")
            print(f"  Spread: predicted margin {sp['predicted_margin']:+.1f} pts")
            print(f"  Total : predicted {tot['predicted_total']:.1f} pts")
            if "over_bet" in tot:
                ob = tot["over_bet"]
                print(f"  Total bet (over): {'✅ BET' if ob['bet'] else '❌ skip'} edge={ob['edge']:+.1%}")

        except Exception as e:
            print(f"  Error predicting {home} vs {away}: {e}")
            all_predictions["games"].append({"matchup": f"{home} vs {away}", "error": str(e)})

    # Write predictions.json — committed back to repo by GitHub Actions
    output_path = os.path.join(os.path.dirname(__file__), "predictions.json")
    with open(output_path, "w") as f:
        json.dump(all_predictions, f, indent=2)

    print(f"\n✓ predictions.json written ({len(all_predictions['games'])} games)")
