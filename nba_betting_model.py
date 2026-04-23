"""
NBA Betting Model
=================
Data:   balldontlie API (works in GitHub Actions)
Covers: Moneyline · Spread · Over/Under · Player Props
Models: Logistic Regression, Ridge Regression, XGBoost
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

API_KEY  = os.environ["BALLDONTLIE_API_KEY"]
BASE_URL = "https://api.balldontlie.io/nba/v1"
HEADERS  = {"Authorization": API_KEY}
SEASON   = 2024   # 2024 = 2024-25 season

UPCOMING_GAMES = [
    ("BOS", "NYK", -155, -4.5, 224.5),
    ("OKC", "DEN", -180, -5.5, 221.0),
    ("SAS", "HOU", -130, -3.0, 218.5),
]

MIN_EDGE       = 0.03
KELLY_FRACTION = 0.25
API_DELAY      = 2.0   # seconds between requests (free tier safe)
MAX_RETRIES    = 5     # retries on 429


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def bdl_get(endpoint: str, params: dict = {}) -> list:
    """Paginate through all results with retry on 429."""
    results = []
    params  = {**params, "per_page": 100}
    cursor  = None

    while True:
        if cursor:
            params["cursor"] = cursor

        for attempt in range(MAX_RETRIES):
            time.sleep(API_DELAY)
            try:
                resp = requests.get(
                    f"{BASE_URL}/{endpoint}",
                    headers=HEADERS,
                    params=params,
                    timeout=30,
                )
                if resp.status_code == 429:
                    wait = 10 * (attempt + 1)
                    print(f"  Rate limited — waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(10)
        else:
            raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {endpoint}")

        data = resp.json()
        results.extend(data["data"])
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        print(f"  Fetched {len(results)} records so far...")

    return results


def fetch_games(season: int = SEASON) -> pd.DataFrame:
    print(f"[Data] Fetching {season}-{str(season+1)[-2:]} games...")
    games = bdl_get("games", {"seasons[]": season, "postseason": "false"})
    rows  = []
    for g in games:
        if g["status"] != "Final":
            continue
        rows.append({
            "game_id":    g["id"],
            "date":       g["date"][:10],
            "home_team":  g["home_team"]["abbreviation"],
            "away_team":  g["visitor_team"]["abbreviation"],
            "home_score": g["home_team_score"],
            "away_score": g["visitor_team_score"],
        })
    df = pd.DataFrame(rows)
    df["date"]       = pd.to_datetime(df["date"])
    df["point_diff"] = df["home_score"] - df["away_score"]
    df["total_pts"]  = df["home_score"] + df["away_score"]
    df["home_win"]   = (df["point_diff"] > 0).astype(int)
    print(f"[Data] {len(df)} completed games.")
    return df.sort_values("date").reset_index(drop=True)


def fetch_team_stats(season: int = SEASON) -> pd.DataFrame:
    print("[Data] Fetching team box scores (may take a few minutes on free tier)...")
    stats = bdl_get("stats", {"seasons[]": season, "postseason": "false"})
    rows  = []
    for s in stats:
        rows.append({
            "game_id": s["game"]["id"],
            "team":    s["team"]["abbreviation"],
            "pts":     s.get("pts") or 0,
            "fgm":     s.get("fgm") or 0,
            "fga":     s.get("fga") or 0,
            "fg3m":    s.get("fg3m") or 0,
            "fg3a":    s.get("fg3a") or 0,
            "ftm":     s.get("ftm") or 0,
            "fta":     s.get("fta") or 0,
            "oreb":    s.get("oreb") or 0,
            "dreb":    s.get("dreb") or 0,
            "ast":     s.get("ast") or 0,
            "tov":     s.get("turnover") or 0,
            "stl":     s.get("stl") or 0,
            "blk":     s.get("blk") or 0,
        })
    df = pd.DataFrame(rows)
    tg = df.groupby(["game_id", "team"]).sum().reset_index()
    tg["fg_pct"]  = tg["fgm"]  / tg["fga"].replace(0, np.nan)
    tg["fg3_pct"] = tg["fg3m"] / tg["fg3a"].replace(0, np.nan)
    tg["ft_pct"]  = tg["ftm"]  / tg["fta"].replace(0, np.nan)
    tg = tg.fillna(0)
    print(f"[Data] {len(tg)} team-game rows.")
    return tg


def build_full_logs(games: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    home = games[["game_id","date","home_team","away_team",
                  "home_score","away_score","point_diff","total_pts","home_win"]].copy()
    home = home.rename(columns={"home_team":"team","away_team":"opponent",
                                 "home_score":"pts_for","away_score":"pts_against"})
    home["home"] = 1; home["result"] = home["home_win"]

    away = games[["game_id","date","away_team","home_team",
                  "home_score","away_score","point_diff","total_pts","home_win"]].copy()
    away = away.rename(columns={"away_team":"team","home_team":"opponent",
                                 "away_score":"pts_for","home_score":"pts_against"})
    away["home"]       = 0
    away["point_diff"] = -away["point_diff"]
    away["result"]     = (1 - away["home_win"]).astype(int)

    logs = pd.concat([home, away], ignore_index=True)
    logs = logs.merge(team_stats, on=["game_id","team"], how="left")
    return logs.sort_values(["team","date"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_features(df: pd.DataFrame, windows=[5, 10]) -> pd.DataFrame:
    stat_cols = [c for c in ["pts_for","pts_against","point_diff","total_pts",
                              "fg_pct","fg3_pct","ft_pct","oreb","dreb",
                              "ast","tov","stl","blk"] if c in df.columns]
    frames = []
    for team, grp in df.groupby("team"):
        grp = grp.copy().sort_values("date")
        for w in windows:
            for col in stat_cols:
                grp[f"{col}_roll{w}"] = (
                    grp[col].shift(1).rolling(w, min_periods=max(1, w//2)).mean()
                )
        frames.append(grp)
    return pd.concat(frames).sort_values(["date","team"]).reset_index(drop=True)


def build_game_level_features(df: pd.DataFrame) -> pd.DataFrame:
    home      = df[df["home"] == 1].copy()
    away      = df[df["home"] == 0].copy()
    roll_cols = [c for c in df.columns if "_roll" in c]
    base      = [c for c in ["game_id","date","team","opponent",
                              "point_diff","total_pts","home_win"] if c in df.columns]
    hf = home[base + roll_cols].rename(columns={
        **{c: f"home_{c}" for c in roll_cols},
        "team":"home_team","opponent":"away_team"})
    af = away[["date","team"] + roll_cols].rename(columns={
        **{c: f"away_{c}" for c in roll_cols}, "team":"away_team_check"})
    games = hf.merge(af, left_on=["date","away_team"],
                     right_on=["date","away_team_check"],
                     how="inner").drop(columns=["away_team_check"])
    for col in roll_cols:
        games[f"diff_{col}"] = games[f"home_{col}"] - games[f"away_{col}"]
    return games.dropna().reset_index(drop=True)


def compute_elo(df: pd.DataFrame, k=20, home_adv=100) -> pd.DataFrame:
    elo = {t: 1500.0 for t in pd.concat([df["home_team"],df["away_team"]]).unique()}
    h_elos, a_elos = [], []
    for _, row in df.sort_values("date").iterrows():
        h, a   = row["home_team"], row["away_team"]
        exp_h  = 1/(1+10**((elo[a]-(elo[h]+home_adv))/400))
        actual = row["home_win"]
        h_elos.append(elo[h]); a_elos.append(elo[a])
        margin = abs(row["point_diff"])
        k_m    = k * np.log1p(margin) * (2.2/((margin*0.001)+2.2))
        elo[h] += k_m*(actual-exp_h); elo[a] += k_m*(exp_h-actual)
    df = df.sort_values("date").copy()
    df["home_elo"]     = h_elos;  df["away_elo"]     = a_elos
    df["elo_diff"]     = df["home_elo"] - df["away_elo"]
    df["elo_win_prob"] = 1/(1+10**(-df["elo_diff"]/400))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE SETS & MODELS
# ─────────────────────────────────────────────────────────────────────────────

ML_FEATS  = ["elo_diff","elo_win_prob","diff_pts_for_roll10","diff_pts_against_roll10",
             "diff_point_diff_roll10","diff_fg_pct_roll5","diff_fg3_pct_roll5",
             "diff_ast_roll5","diff_tov_roll5"]
SP_FEATS  = ["elo_diff","diff_pts_for_roll10","diff_pts_against_roll10",
             "diff_point_diff_roll10","diff_fg_pct_roll5","diff_oreb_roll5",
             "diff_tov_roll5","home_pts_for_roll10","home_pts_against_roll10",
             "away_pts_for_roll10","away_pts_against_roll10"]
TOT_FEATS = ["home_pts_for_roll10","home_pts_against_roll10","away_pts_for_roll10",
             "away_pts_against_roll10","home_total_pts_roll10","away_total_pts_roll10",
             "home_fg_pct_roll5","away_fg_pct_roll5","home_fg3_pct_roll5","away_fg3_pct_roll5"]

def avail(df, cols): return [c for c in cols if c in df.columns]

def train_moneyline(games):
    f = avail(games, ML_FEATS); X, y = games[f].values, games["home_win"].values
    pipe = Pipeline([("sc",StandardScaler()),
                     ("cl",CalibratedClassifierCV(LogisticRegression(C=1.0,max_iter=1000),cv=5))])
    xgbc = xgb.XGBClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,
                              subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",
                              use_label_encoder=False,random_state=42)
    pipe.fit(X,y); xgbc.fit(X,y)
    tscv = TimeSeriesSplit(5)
    print(f"[ML]  LR AUC={cross_val_score(pipe,X,y,cv=tscv,scoring='roc_auc').mean():.3f}  "
          f"XGB AUC={cross_val_score(xgbc,X,y,cv=tscv,scoring='roc_auc').mean():.3f}")
    return {"lr":pipe,"xgb":xgbc,"features":f}

def train_spread(games):
    f = avail(games, SP_FEATS); X, y = games[f].values, games["point_diff"].values
    pipe = Pipeline([("sc",StandardScaler()),("rg",Ridge(alpha=10.0))])
    pipe.fit(X,y)
    print(f"[SP]  MAE={-cross_val_score(pipe,X,y,cv=TimeSeriesSplit(5),scoring='neg_mean_absolute_error').mean():.2f} pts")
    return {"model":pipe,"features":f}

def train_totals(games):
    f = avail(games, TOT_FEATS); X, y = games[f].values, games["total_pts"].values
    pipe = Pipeline([("sc",StandardScaler()),("rg",Ridge(alpha=10.0))])
    pipe.fit(X,y)
    print(f"[TOT] MAE={-cross_val_score(pipe,X,y,cv=TimeSeriesSplit(5),scoring='neg_mean_absolute_error').mean():.2f} pts")
    return {"model":pipe,"features":f}


# ─────────────────────────────────────────────────────────────────────────────
# 4. BETTING EDGE
# ─────────────────────────────────────────────────────────────────────────────

def to_prob(o): return 100/(o+100) if o>0 else abs(o)/(abs(o)+100)
def kelly(p,o):
    b = o/100 if o>0 else 100/abs(o)
    return max(0.0,((b*p-(1-p))/b)*KELLY_FRACTION)
def ev(p,o):
    imp=to_prob(o); e=p-imp
    return {"model_prob":round(p,3),"implied_prob":round(imp,3),"edge":round(e,3),
            "kelly_pct":round(kelly(p,o)*100 if e>MIN_EDGE else 0,1),"bet":e>MIN_EDGE}


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class NBABettingModel:
    def __init__(self, season=SEASON):
        self.season=season; self.models={}; self.games=None; self.raw=None

    def fit(self):
        print("="*55+"\nNBA Betting Model — Training\n"+"="*55)
        games      = fetch_games(self.season)
        team_stats = fetch_team_stats(self.season)
        logs       = build_full_logs(games, team_stats)
        rolled     = compute_rolling_features(logs)
        gf         = build_game_level_features(rolled)
        gf         = compute_elo(gf)
        self.games = gf; self.raw = rolled
        self.models["ml"]  = train_moneyline(gf)
        self.models["sp"]  = train_spread(gf)
        self.models["tot"] = train_totals(gf)
        print("✓ Done.\n")

    def _feats(self, team, is_home):
        prefix = "home" if is_home else "away"
        tdf = self.raw[self.raw["team"]==team].sort_values("date")
        if tdf.empty: raise ValueError(f"Team '{team}' not found.")
        latest = tdf.iloc[-1]
        return {f"{prefix}_{c}":latest[c] for c in tdf.columns if "_roll" in c}

    def predict(self, home, away, ml_odds=None, spread_line=None,
                spread_odds=-110, total_line=None, total_odds=-110):
        fr = {**self._feats(home,True), **self._feats(away,False)}
        h_elo = self.games[self.games["home_team"]==home].sort_values("date").iloc[-1]["home_elo"]
        a_elo = self.games[self.games["away_team"]==away].sort_values("date").iloc[-1]["away_elo"]
        fr["elo_diff"]     = h_elo - a_elo
        fr["elo_win_prob"] = 1/(1+10**(-fr["elo_diff"]/400))
        for s in {k[5:] for k in fr if k.startswith("home_") and "_roll" in k}:
            fr[f"diff_{s}"] = fr.get(f"home_{s}",0)-fr.get(f"away_{s}",0)

        res = {"matchup":f"{home} vs {away}","generated_at":datetime.utcnow().isoformat()+"Z"}

        m   = self.models["ml"]
        hwp = float(m["lr"].predict_proba(np.array([[fr.get(f,0) for f in m["features"]]]))[0][1])
        res["moneyline"] = {"home_win_prob":round(hwp,3),"away_win_prob":round(1-hwp,3)}
        if ml_odds:
            res["moneyline"]["home_bet"] = ev(hwp,ml_odds)
            res["moneyline"]["away_bet"] = ev(1-hwp,-ml_odds)

        sp  = self.models["sp"]
        pd_ = float(sp["model"].predict(np.array([[fr.get(f,0) for f in sp["features"]]]))[0])
        res["spread"] = {"predicted_margin":round(pd_,2)}
        if spread_line is not None:
            cp = min(0.65,max(0.35,0.5+(pd_-spread_line)*0.02))
            res["spread"]["home_covers_prob"]=round(cp,3)
            res["spread"]["home_bet"]=ev(cp,spread_odds)

        tot = self.models["tot"]
        pt  = float(tot["model"].predict(np.array([[fr.get(f,0) for f in tot["features"]]]))[0])
        res["totals"] = {"predicted_total":round(pt,1)}
        if total_line is not None:
            op = min(0.65,max(0.35,0.5+(pt-total_line)*0.02))
            res["totals"]["over_prob"]=round(op,3)
            res["totals"]["over_bet"]=ev(op,total_odds)
            res["totals"]["under_bet"]=ev(1-op,total_odds)

        return res


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = NBABettingModel(season=SEASON)
    model.fit()

    out = {"generated_at":datetime.utcnow().isoformat()+"Z","games":[]}

    for home, away, ml_odds, spread, total in UPCOMING_GAMES:
        try:
            pred = model.predict(home,away,ml_odds,spread,-110,total,-110)
            out["games"].append(pred)
            ml=pred["moneyline"]; sp=pred["spread"]; tot=pred["totals"]
            print(f"\n{'─'*48}\n  {pred['matchup']}")
            print(f"  ML : home {ml['home_win_prob']:.1%} | away {ml['away_win_prob']:.1%}")
            if "home_bet" in ml:
                b=ml["home_bet"]
                print(f"  ML bet: {'✅ BET' if b['bet'] else '❌ skip'} "
                      f"edge={b['edge']:+.1%} Kelly={b['kelly_pct']:.1f}%")
            print(f"  Spread: {sp['predicted_margin']:+.1f} pts | "
                  f"Total: {tot['predicted_total']:.1f} pts")
        except Exception as e:
            print(f"  Error {home} vs {away}: {e}")
            out["games"].append({"matchup":f"{home} vs {away}","error":str(e)})

    path = os.path.join(os.path.dirname(__file__),"predictions.json")
    with open(path,"w") as f: json.dump(out,f,indent=2)
    print(f"\n✓ predictions.json written ({len(out['games'])} games)")
