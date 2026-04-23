"""
NBA Betting Model
=================
Covers: Moneyline · Spread · Over/Under · Player Props
Data:   nba_api (official NBA stats API — works in CI/GitHub Actions)
Models: Logistic Regression, Ridge Regression, XGBoost

Outputs predictions.json to repo root on each run.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

# nba_api
from nba_api.stats.endpoints import leaguegamelog, playergamelog, leaguedashteamstats
from nba_api.stats.static import teams as nba_teams_static

# sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# XGBoost
import xgboost as xgb


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SEASON = "2024-25"

# Games to predict — update before each run
# Format: (home_abbr, away_abbr, ml_odds_home, spread_line, total_line)
UPCOMING_GAMES = [
    ("BOS", "NYK", -155, -4.5, 224.5),
    ("OKC", "DEN", -180, -5.5, 221.0),
    ("SAS", "HOU", -130, -3.0, 218.5),
]

MIN_EDGE       = 0.03
KELLY_FRACTION = 0.25
API_DELAY      = 0.6


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def fetch_team_game_logs(season: str = SEASON) -> pd.DataFrame:
    print(f"[Data] Fetching game logs for {season} via nba_api...")
    time.sleep(API_DELAY)
    logs = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        direction="ASC",
    )
    df = logs.get_data_frames()[0]
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["game_date"])
    df["home"] = df["matchup"].apply(lambda x: 0 if "@" in x else 1)
    df = df.rename(columns={"team_abbreviation": "team", "pts": "pts_for", "wl": "result_str"})
    df["result"]      = (df["result_str"] == "W").astype(int)
    df["pts_against"] = df["pts_for"] - df["plus_minus"]
    df["point_diff"]  = df["pts_for"] - df["pts_against"]
    df["total_pts"]   = df["pts_for"] + df["pts_against"]
    df["opponent"]    = df["matchup"].apply(
        lambda x: x.split(" @ ")[-1] if "@" in x else x.split(" vs. ")[-1]
    )
    df = df.sort_values(["team", "date"]).reset_index(drop=True)
    print(f"[Data] {len(df)} team-game rows across {df['team'].nunique()} teams.")
    return df


def compute_rolling_features(df: pd.DataFrame, windows: list = [5, 10]) -> pd.DataFrame:
    stat_cols = [c for c in [
        "pts_for", "pts_against", "point_diff", "total_pts",
        "fg_pct", "fg3_pct", "ft_pct",
        "oreb", "dreb", "ast", "tov", "stl", "blk", "plus_minus",
    ] if c in df.columns]

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
    base_cols = [c for c in ["date", "team", "opponent", "point_diff",
                              "total_pts", "result", "game_id"] if c in df.columns]

    home_feat = home[base_cols + roll_cols].copy()
    home_feat = home_feat.rename(columns={
        **{c: f"home_{c}" for c in roll_cols},
        "team": "home_team", "opponent": "away_team", "result": "home_win",
    })

    away_feat = away[["date", "team"] + roll_cols].copy()
    away_feat = away_feat.rename(columns={
        **{c: f"away_{c}" for c in roll_cols},
        "team": "away_team_check",
    })

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
    elo = {t: 1500.0 for t in
           pd.concat([df["home_team"], df["away_team"]]).unique()}
    home_elos, away_elos = [], []

    for _, row in df.sort_values("date").iterrows():
        h, a   = row["home_team"], row["away_team"]
        exp_h  = 1 / (1 + 10 ** ((elo[a] - (elo[h] + home_adv)) / 400))
        actual = row["home_win"]
        home_elos.append(elo[h])
        away_elos.append(elo[a])
        margin   = abs(row["point_diff"])
        k_margin = k * np.log1p(margin) * (2.2 / ((margin * 0.001) + 2.2))
        elo[h]  += k_margin * (actual - exp_h)
        elo[a]  += k_margin * (exp_h - actual)

    df = df.sort_values("date").copy()
    df["home_elo"]     = home_elos
    df["away_elo"]     = away_elos
    df["elo_diff"]     = df["home_elo"] - df["away_elo"]
    df["elo_win_prob"] = 1 / (1 + 10 ** (-df["elo_diff"] / 400))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE SETS
# ─────────────────────────────────────────────────────────────────────────────

MONEYLINE_FEATURES = [
    "elo_diff", "elo_win_prob",
    "diff_pts_for_roll10", "diff_pts_against_roll10",
    "diff_point_diff_roll10", "diff_plus_minus_roll10",
    "diff_fg_pct_roll5", "diff_fg3_pct_roll5",
    "diff_ast_roll5", "diff_tov_roll5",
]

SPREAD_FEATURES = [
    "elo_diff",
    "diff_pts_for_roll10", "diff_pts_against_roll10",
    "diff_point_diff_roll10", "diff_plus_minus_roll10",
    "diff_fg_pct_roll5", "diff_fg3_pct_roll5",
    "diff_oreb_roll5", "diff_tov_roll5",
    "home_pts_for_roll10", "home_pts_against_roll10",
    "away_pts_for_roll10", "away_pts_against_roll10",
]

TOTALS_FEATURES = [
    "home_pts_for_roll10", "home_pts_against_roll10",
    "away_pts_for_roll10", "away_pts_against_roll10",
    "home_total_pts_roll10", "away_total_pts_roll10",
    "home_fg_pct_roll5", "away_fg_pct_roll5",
    "home_fg3_pct_roll5", "away_fg3_pct_roll5",
]


def filter_available(df, cols):
    keys = df.columns if isinstance(df, pd.DataFrame) else df
    return [c for c in cols if c in keys]


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODELS
# ─────────────────────────────────────────────────────────────────────────────

def train_moneyline_model(games):
    feats = filter_available(games, MONEYLINE_FEATURES)
    X, y  = games[feats].values, games["home_win"].values
    pipe  = Pipeline([
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
    tscv    = TimeSeriesSplit(n_splits=5)
    lr_auc  = cross_val_score(pipe,    X, y, cv=tscv, scoring="roc_auc").mean()
    xgb_auc = cross_val_score(xgb_clf, X, y, cv=tscv, scoring="roc_auc").mean()
    print(f"[Moneyline] LR ROC-AUC={lr_auc:.3f}  XGB ROC-AUC={xgb_auc:.3f}")
    return {"lr": pipe, "xgb": xgb_clf, "features": feats}


def train_spread_model(games):
    feats = filter_available(games, SPREAD_FEATURES)
    X, y  = games[feats].values, games["point_diff"].values
    pipe  = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0))])
    pipe.fit(X, y)
    mae = -cross_val_score(pipe, X, y, cv=TimeSeriesSplit(5),
                           scoring="neg_mean_absolute_error").mean()
    print(f"[Spread]    Ridge MAE={mae:.2f} pts")
    return {"model": pipe, "features": feats}


def train_totals_model(games):
    feats = filter_available(games, TOTALS_FEATURES)
    X, y  = games[feats].values, games["total_pts"].values
    pipe  = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0))])
    pipe.fit(X, y)
    mae = -cross_val_score(pipe, X, y, cv=TimeSeriesSplit(5),
                           scoring="neg_mean_absolute_error").mean()
    print(f"[Totals]    Ridge MAE={mae:.2f} pts")
    return {"model": pipe, "features": feats}


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLAYER PROPS
# ─────────────────────────────────────────────────────────────────────────────

def run_player_prop(player_id: int, stat: str, line: float,
                    odds: int = -110, season: str = SEASON):
    """
    End-to-end player prop evaluation.
    player_id: nba_api numeric ID (find at nba_api.stats.static.players)
    stat: nba_api column name e.g. 'pts', 'ast', 'reb', 'fg3m'
    """
    time.sleep(API_DELAY)
    logs = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df   = logs.get_data_frames()[0]
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("date").reset_index(drop=True)

    if stat not in df.columns:
        print(f"Stat '{stat}' not found. Available: {list(df.columns)}")
        return

    for w in [5, 10, 20]:
        df[f"{stat}_roll{w}"] = df[stat].shift(1).rolling(w, min_periods=2).mean()
        df[f"{stat}_std{w}"]  = df[stat].shift(1).rolling(w, min_periods=2).std()
    df["home"] = df["matchup"].apply(lambda x: 0 if "@" in x else 1)
    df = df.dropna()

    feat_cols = [c for c in df.columns if "_roll" in c or "_std" in c or c == "home"]
    X, y = df[feat_cols].values, df[stat].values

    model = xgb.XGBRegressor(n_estimators=200, max_depth=3, random_state=42)
    model.fit(X, y)
    pred   = float(model.predict(df[feat_cols].iloc[[-1]].values)[0])
    over_p = float(np.mean(df[stat] > line))
    bet    = evaluate_bet(over_p, odds)

    print(f"\n[Props] player_id={player_id}  stat={stat}  line={line}")
    print(f"  Model prediction : {pred:.1f}")
    print(f"  Historical over% : {over_p:.1%}")
    print(f"  {'✅ BET OVER' if bet['bet'] else '❌ skip'}  "
          f"edge={bet['edge']:+.1%}  Kelly={bet['kelly_pct']:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 6. BETTING EDGE
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
# 7. MAIN MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class NBABettingModel:

    def __init__(self, season: str = SEASON):
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
        return {f"{prefix}_{c}": latest[c] for c in team_df.columns if "_roll" in c}

    def predict_game(self, home: str, away: str,
                     ml_odds_home: int = None,
                     spread_line: float = None, spread_odds: int = -110,
                     total_line: float = None, total_odds: int = -110) -> dict:

        feat_row = {**self._team_feats(home, True), **self._team_feats(away, False)}

        h_elo = self.games[self.games["home_team"] == home].sort_values("date").iloc[-1]["home_elo"]
        a_elo = self.games[self.games["away_team"] == away].sort_values("date").iloc[-1]["away_elo"]
        feat_row["elo_diff"]     = h_elo - a_elo
        feat_row["elo_win_prob"] = 1 / (1 + 10 ** (-feat_row["elo_diff"] / 400))

        for suf in {k[5:] for k in feat_row if k.startswith("home_") and "_roll" in k}:
            feat_row[f"diff_{suf}"] = (feat_row.get(f"home_{suf}", 0)
                                       - feat_row.get(f"away_{suf}", 0))

        result = {"matchup": f"{home} vs {away}",
                  "generated_at": datetime.utcnow().isoformat() + "Z"}

        ml   = self.models["moneyline"]
        ml_x = np.array([[feat_row.get(f, 0) for f in ml["features"]]])
        hwp  = float(ml["lr"].predict_proba(ml_x)[0][1])
        result["moneyline"] = {"home_win_prob": round(hwp, 3),
                               "away_win_prob": round(1 - hwp, 3)}
        if ml_odds_home:
            result["moneyline"]["home_bet"] = evaluate_bet(hwp,      ml_odds_home)
            result["moneyline"]["away_bet"] = evaluate_bet(1 - hwp, -ml_odds_home)

        sp        = self.models["spread"]
        pred_diff = float(sp["model"].predict(
            np.array([[feat_row.get(f, 0) for f in sp["features"]]]))[0])
        result["spread"] = {"predicted_margin": round(pred_diff, 2)}
        if spread_line is not None:
            cp = min(0.65, max(0.35, 0.5 + (pred_diff - spread_line) * 0.02))
            result["spread"]["home_covers_prob"] = round(cp, 3)
            result["spread"]["home_bet"] = evaluate_bet(cp, spread_odds)

        tot        = self.models["totals"]
        pred_total = float(tot["model"].predict(
            np.array([[feat_row.get(f, 0) for f in tot["features"]]]))[0])
        result["totals"] = {"predicted_total": round(pred_total, 1)}
        if total_line is not None:
            op = min(0.65, max(0.35, 0.5 + (pred_total - total_line) * 0.02))
            result["totals"]["over_prob"]  = round(op, 3)
            result["totals"]["over_bet"]   = evaluate_bet(op,     total_odds)
            result["totals"]["under_bet"]  = evaluate_bet(1 - op, total_odds)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
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
            ml  = pred["moneyline"]
            sp  = pred["spread"]
            tot = pred["totals"]
            print(f"\n{'─'*50}")
            print(f"  {pred['matchup']}")
            print(f"  ML   : home {ml['home_win_prob']:.1%} | away {ml['away_win_prob']:.1%}")
            if "home_bet" in ml:
                b = ml["home_bet"]
                print(f"  ML bet: {'✅ BET' if b['bet'] else '❌ skip'} "
                      f"edge={b['edge']:+.1%} Kelly={b['kelly_pct']:.1f}%")
            print(f"  Spread: {sp['predicted_margin']:+.1f} pts  |  "
                  f"Total: {tot['predicted_total']:.1f} pts")
        except Exception as e:
            print(f"  Error: {home} vs {away}: {e}")
            all_predictions["games"].append({"matchup": f"{home} vs {away}", "error": str(e)})

    output_path = os.path.join(os.path.dirname(__file__), "predictions.json")
    with open(output_path, "w") as f:
        json.dump(all_predictions, f, indent=2)
    print(f"\n✓ predictions.json written ({len(all_predictions['games'])} games)")
