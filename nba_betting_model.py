"""
NBA Betting Model - Edge Edition
Auto-fetches tonight's games from balldontlie API.
No manual UPCOMING_GAMES update needed.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

API_KEY  = os.environ["BALLDONTLIE_API_KEY"]
BASE_URL = "https://api.balldontlie.io/nba/v1"
HEADERS  = {"Authorization": API_KEY}
SEASON   = 2024

MIN_EDGE       = 0.03
KELLY_FRACTION = 0.25
API_DELAY      = 1.5
MAX_RETRIES    = 6

BET_LOG_PATH = os.path.join(os.path.dirname(__file__), "bet_log.json")


# -----------------------------------------------------------------------------
# 1. DATA COLLECTION
# -----------------------------------------------------------------------------

def bdl_get(endpoint, params={}):
    results = []
    params  = {**params, "per_page": 100}
    cursor  = None
    while True:
        if cursor:
            params["cursor"] = cursor
        for attempt in range(MAX_RETRIES):
            time.sleep(API_DELAY)
            try:
                resp = requests.get(f"{BASE_URL}/{endpoint}",
                                    headers=HEADERS, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = 15 * (attempt + 1)
                    print(f"  Rate limited - waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(15)
        data   = resp.json()
        results.extend(data["data"])
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        print(f"  {len(results)} records fetched...")
    return results


def fetch_games(season=SEASON):
    print(f"[Data] Fetching {season}-{str(season+1)[-2:]} game scores...")
    raw  = bdl_get("games", {"seasons[]": season, "postseason": "false"})
    rows = []
    for g in raw:
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
    print(f"[Data] {len(df)} completed games loaded.")
    return df.sort_values("date").reset_index(drop=True)


def fetch_upcoming_games(season=SEASON):
    """
    Fetch today's and tomorrow's scheduled games automatically.
    Returns list of (home_abbr, away_abbr) tuples.
    """
    print("[Data] Fetching upcoming scheduled games...")
    today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw      = bdl_get("games", {
        "seasons[]":    season,
        "postseason":   "false",
        "start_date":   today,
        "per_page":     100,
    })
    upcoming = []
    for g in raw:
        if g["status"] == "Final":
            continue
        upcoming.append({
            "home": g["home_team"]["abbreviation"],
            "away": g["visitor_team"]["abbreviation"],
            "date": g["date"][:10],
            "time": g.get("status", "TBD"),
        })
    print(f"[Data] {len(upcoming)} upcoming games found.")
    return upcoming


def build_team_logs(games):
    home = games[["game_id","date","home_team","away_team",
                  "home_score","away_score","point_diff","total_pts","home_win"]].copy()
    home = home.rename(columns={"home_team":"team","away_team":"opponent",
                                 "home_score":"pts_for","away_score":"pts_against"})
    home["home"] = 1; home["result"] = home["home_win"]

    away = games[["game_id","date","away_team","home_team",
                  "away_score","home_score","point_diff","total_pts","home_win"]].copy()
    away = away.rename(columns={"away_team":"team","home_team":"opponent",
                                 "away_score":"pts_for","home_score":"pts_against"})
    away["home"] = 0
    away["point_diff"] = -away["point_diff"]
    away["result"] = (1 - away["home_win"]).astype(int)

    logs = pd.concat([home, away], ignore_index=True)
    return logs.sort_values(["team","date"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# -----------------------------------------------------------------------------

def build_all_features(logs, windows=[5, 10, 20]):
    frames = []
    for team, grp in logs.groupby("team"):
        grp = grp.copy().sort_values("date")

        # Base rolling
        for w in windows:
            for col in ["pts_for","pts_against","point_diff","total_pts"]:
                grp[f"{col}_roll{w}"] = grp[col].shift(1).rolling(w, min_periods=max(2,w//2)).mean()
            grp[f"win_rate{w}"] = grp["result"].shift(1).rolling(w, min_periods=max(2,w//2)).mean()

        # Rest & fatigue
        grp["days_rest"] = grp["date"].diff().dt.days.fillna(7).clip(0, 14)
        grp["is_b2b"]    = (grp["days_rest"] == 1).astype(int)

        # Momentum (EWM)
        grp["pts_for_ewm"]     = grp["pts_for"].shift(1).ewm(span=5).mean()
        grp["pts_against_ewm"] = grp["pts_against"].shift(1).ewm(span=5).mean()
        grp["point_diff_ewm"]  = grp["point_diff"].shift(1).ewm(span=5).mean()
        grp["win_ewm"]         = grp["result"].shift(1).ewm(span=5).mean()
        grp["win_rate3"]       = grp["result"].shift(1).rolling(3, min_periods=1).mean()
        grp["momentum"]        = grp["win_rate3"] - grp["win_rate10"] if "win_rate10" in grp.columns else 0

        # Pace proxy
        grp["pace_proxy"]  = grp["pts_for"] + grp["pts_against"]
        grp["pace_roll5"]  = grp["pace_proxy"].shift(1).rolling(5,  min_periods=2).mean()
        grp["pace_roll10"] = grp["pace_proxy"].shift(1).rolling(10, min_periods=3).mean()
        grp["scoring_var10"] = grp["pts_for"].shift(1).rolling(10, min_periods=3).std()

        # Injury proxy
        roll_avg = grp["pts_for"].shift(1).rolling(10, min_periods=3).mean()
        grp["scoring_drop"]  = (roll_avg - grp["pts_for"].shift(1)).clip(lower=0)
        grp["injury_signal"] = grp["scoring_drop"].shift(1).rolling(5, min_periods=1).mean()

        # Home/away splits
        home_pts = grp[grp["home"]==1]["pts_for"].shift(1).rolling(10, min_periods=2).mean()
        away_pts = grp[grp["home"]==0]["pts_for"].shift(1).rolling(10, min_periods=2).mean()
        grp["home_scoring_avg"] = home_pts.reindex(grp.index).ffill()
        grp["away_scoring_avg"] = away_pts.reindex(grp.index).ffill()

        frames.append(grp)
    return pd.concat(frames).sort_values(["date","team"]).reset_index(drop=True)


def build_game_level_features(df):
    home = df[df["home"] == 1].copy()
    away = df[df["home"] == 0].copy()
    feat_cols = [c for c in df.columns if any(x in c for x in [
        "_roll","win_rate","ewm","momentum","pace","scoring",
        "injury","home_scoring","away_scoring","days_rest","is_b2b",
    ])]
    base = [c for c in ["game_id","date","team","opponent",
                         "point_diff","total_pts","home_win"] if c in df.columns]
    hf = home[base + feat_cols].rename(columns={
        **{c: f"home_{c}" for c in feat_cols},
        "team":"home_team","opponent":"away_team"})
    af = away[["date","team"] + feat_cols].rename(columns={
        **{c: f"away_{c}" for c in feat_cols},
        "team":"away_team_check"})
    games = hf.merge(af, left_on=["date","away_team"],
                     right_on=["date","away_team_check"],
                     how="inner").drop(columns=["away_team_check"])
    for col in feat_cols:
        games[f"diff_{col}"] = games[f"home_{col}"] - games[f"away_{col}"]
    return games.dropna().reset_index(drop=True)


def compute_elo(df, k=20, home_adv=100):
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


# -----------------------------------------------------------------------------
# 3. FEATURE SETS
# -----------------------------------------------------------------------------

ML_FEATS = [
    "elo_diff","elo_win_prob",
    "diff_pts_for_roll10","diff_pts_against_roll10","diff_point_diff_roll10",
    "diff_pts_for_roll5","diff_pts_against_roll5",
    "diff_win_rate10","diff_win_rate5","home_win_rate10","away_win_rate10",
    "diff_momentum","diff_win_ewm","diff_point_diff_ewm",
    "diff_days_rest","home_is_b2b","away_is_b2b",
    "diff_injury_signal","home_injury_signal","away_injury_signal",
]
SP_FEATS = [
    "elo_diff",
    "diff_pts_for_roll10","diff_pts_against_roll10","diff_point_diff_roll10",
    "diff_pts_for_roll5","diff_pts_against_roll5",
    "diff_momentum","diff_point_diff_ewm",
    "home_pts_for_roll10","home_pts_against_roll10",
    "away_pts_for_roll10","away_pts_against_roll10",
    "diff_days_rest","home_is_b2b","away_is_b2b",
    "diff_injury_signal",
    "home_home_scoring_avg","away_away_scoring_avg",
]
TOT_FEATS = [
    "home_pts_for_roll10","home_pts_against_roll10",
    "away_pts_for_roll10","away_pts_against_roll10",
    "home_total_pts_roll10","away_total_pts_roll10",
    "home_total_pts_roll5","away_total_pts_roll5",
    "home_pts_for_roll20","away_pts_for_roll20",
    "home_pace_roll10","away_pace_roll10","diff_pace_roll10",
    "home_pace_roll5","away_pace_roll5",
    "home_scoring_var10","away_scoring_var10",
    "diff_days_rest","home_is_b2b","away_is_b2b",
]

def avail(df, cols): return [c for c in cols if c in df.columns]


# -----------------------------------------------------------------------------
# 4. MODELS
# -----------------------------------------------------------------------------

def train_moneyline(games):
    f = avail(games, ML_FEATS); X, y = games[f].values, games["home_win"].values
    pipe = Pipeline([("sc",StandardScaler()),
                     ("cl",CalibratedClassifierCV(
                         LogisticRegression(C=0.5,max_iter=1000),cv=5))])
    xgbc = xgb.XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.04,
                              subsample=0.8,colsample_bytree=0.8,min_child_weight=3,
                              eval_metric="logloss",use_label_encoder=False,random_state=42)
    pipe.fit(X,y); xgbc.fit(X,y)
    tscv    = TimeSeriesSplit(5)
    lr_auc  = cross_val_score(pipe, X,y,cv=tscv,scoring="roc_auc").mean()
    xgb_auc = cross_val_score(xgbc,X,y,cv=tscv,scoring="roc_auc").mean()
    print(f"[ML]  LR AUC={lr_auc:.3f}  XGB AUC={xgb_auc:.3f}")
    return {"lr":pipe,"xgb":xgbc,"features":f,"lr_auc":round(lr_auc,3),"xgb_auc":round(xgb_auc,3)}

def train_spread(games):
    f = avail(games, SP_FEATS); X, y = games[f].values, games["point_diff"].values
    pipe = Pipeline([("sc",StandardScaler()),("rg",Ridge(alpha=10.0))])
    pipe.fit(X,y)
    mae = -cross_val_score(pipe,X,y,cv=TimeSeriesSplit(5),scoring="neg_mean_absolute_error").mean()
    print(f"[SP]  Ridge MAE={mae:.2f} pts")
    return {"model":pipe,"features":f,"mae":round(mae,2)}

def train_totals(games):
    f = avail(games, TOT_FEATS); X, y = games[f].values, games["total_pts"].values
    pipe = Pipeline([("sc",StandardScaler()),("rg",Ridge(alpha=10.0))])
    pipe.fit(X,y)
    mae = -cross_val_score(pipe,X,y,cv=TimeSeriesSplit(5),scoring="neg_mean_absolute_error").mean()
    print(f"[TOT] Ridge MAE={mae:.2f} pts")
    return {"model":pipe,"features":f,"mae":round(mae,2)}


# -----------------------------------------------------------------------------
# 5. BETTING EDGE
# -----------------------------------------------------------------------------

def to_prob(o): return 100/(o+100) if o>0 else abs(o)/(abs(o)+100)

def kelly(p,o):
    b = o/100 if o>0 else 100/abs(o)
    return max(0.0,((b*p-(1-p))/b)*KELLY_FRACTION)

def ev(p,o,label=""):
    imp = to_prob(o); e = p-imp
    return {"label":label,"model_prob":round(p,3),"implied_prob":round(imp,3),
            "edge":round(e,3),"kelly_pct":round(kelly(p,o)*100 if e>MIN_EDGE else 0,1),
            "bet":e>MIN_EDGE}

def ensemble_prob(lr_model, xgb_model, features, feat_row):
    x     = np.array([[feat_row.get(f,0) for f in features]])
    lr_p  = float(lr_model.predict_proba(x)[0][1])
    xgb_p = float(xgb_model.predict_proba(x)[0][1])
    return 0.45*lr_p + 0.55*xgb_p, lr_p, xgb_p


# -----------------------------------------------------------------------------
# 6. BET LOG
# -----------------------------------------------------------------------------

def load_bet_log():
    if os.path.exists(BET_LOG_PATH):
        with open(BET_LOG_PATH) as f:
            return json.load(f)
    return {"bets":[],"summary":{}}

def save_bet_log(log):
    with open(BET_LOG_PATH,"w") as f:
        json.dump(log,f,indent=2)

def log_bet(matchup,bet_type,side,model_prob,implied_prob,edge,odds,kelly_pct,date_str):
    log = load_bet_log()
    entry = {"id":f"{date_str}_{matchup}_{bet_type}_{side}".replace(" ","_"),
             "date":date_str,"matchup":matchup,"bet_type":bet_type,"side":side,
             "model_prob":model_prob,"implied_prob":implied_prob,"edge":edge,
             "odds":odds,"kelly_pct":kelly_pct,"result":"PENDING",
             "closing_odds":None,"clv":None}
    existing_ids = {b["id"] for b in log["bets"]}
    if entry["id"] not in existing_ids:
        log["bets"].append(entry)
    graded = [b for b in log["bets"] if b["result"] in ("W","L")]
    if graded:
        wins = sum(1 for b in graded if b["result"]=="W")
        log["summary"] = {"total_bets":len(graded),"wins":wins,
                          "losses":len(graded)-wins,"win_rate":round(wins/len(graded),3),
                          "avg_edge":round(np.mean([b["edge"] for b in graded]),3),
                          "pending_bets":sum(1 for b in log["bets"] if b["result"]=="PENDING")}
    save_bet_log(log)
    return entry


# -----------------------------------------------------------------------------
# 7. MAIN MODEL CLASS
# -----------------------------------------------------------------------------

class NBABettingModel:
    def __init__(self,season=SEASON):
        self.season=season; self.models={}; self.games=None; self.raw=None

    def fit(self):
        print("="*55+"\nNBA Betting Model - Edge Edition\n"+"="*55)
        games  = fetch_games(self.season)
        logs   = build_team_logs(games)
        rolled = build_all_features(logs)
        gf     = build_game_level_features(rolled)
        gf     = compute_elo(gf)
        self.games=gf; self.raw=rolled
        self.models["ml"]  = train_moneyline(gf)
        self.models["sp"]  = train_spread(gf)
        self.models["tot"] = train_totals(gf)
        print("Done.\n")

    def _feats(self,team,is_home):
        prefix = "home" if is_home else "away"
        tdf = self.raw[self.raw["team"]==team].sort_values("date")
        if tdf.empty: raise ValueError(f"Team '{team}' not found.")
        latest = tdf.iloc[-1]
        feat_keys = [c for c in tdf.columns if any(x in c for x in [
            "_roll","win_rate","ewm","momentum","pace","scoring",
            "injury","home_scoring","away_scoring","days_rest","is_b2b",
        ])]
        return {f"{prefix}_{c}":latest[c] for c in feat_keys}

    def predict(self,home,away,log_bets=True):
        fr = {**self._feats(home,True),**self._feats(away,False)}
        h_elo = self.games[self.games["home_team"]==home].sort_values("date").iloc[-1]["home_elo"]
        a_elo = self.games[self.games["away_team"]==away].sort_values("date").iloc[-1]["away_elo"]
        fr["elo_diff"]     = h_elo-a_elo
        fr["elo_win_prob"] = 1/(1+10**(-fr["elo_diff"]/400))
        for s in {k[5:] for k in fr if k.startswith("home_")}:
            fr[f"diff_{s}"] = fr.get(f"home_{s}",0)-fr.get(f"away_{s}",0)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        res   = {"matchup":f"{home} vs {away}",
                 "home_team":home,"away_team":away,
                 "generated_at":datetime.utcnow().isoformat()+"Z",
                 "bets_flagged":[]}

        # Moneyline
        m = self.models["ml"]
        hwp,lr_p,xgb_p = ensemble_prob(m["lr"],m["xgb"],m["features"],fr)
        awp = 1-hwp
        res["moneyline"] = {
            "home_win_prob":round(hwp,3),"away_win_prob":round(awp,3),
            "lr_prob":round(lr_p,3),"xgb_prob":round(xgb_p,3),
            "elo_prob":round(fr["elo_win_prob"],3),
            "model_consensus":abs(lr_p-xgb_p)<0.05,
        }

        # Spread
        sp  = self.models["sp"]
        pd_ = float(sp["model"].predict(np.array([[fr.get(f,0) for f in sp["features"]]]))[0])
        res["spread"] = {"predicted_margin":round(pd_,2),"model_mae":sp["mae"]}

        # Totals
        tot = self.models["tot"]
        pt  = float(tot["model"].predict(np.array([[fr.get(f,0) for f in tot["features"]]]))[0])
        res["totals"] = {"predicted_total":round(pt,1),"model_mae":tot["mae"]}

        # B2B fatigue alerts
        res["fatigue_alert"] = []
        if fr.get("home_is_b2b",0): res["fatigue_alert"].append(f"{home} on B2B")
        if fr.get("away_is_b2b",0): res["fatigue_alert"].append(f"{away} on B2B")

        # ELO ratings for display
        res["elo"] = {"home":round(h_elo,0),"away":round(a_elo,0),
                      "diff":round(h_elo-a_elo,0)}

        return res


# -----------------------------------------------------------------------------
# 8. ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    model = NBABettingModel(season=SEASON)
    model.fit()

    # Auto-fetch tonight's games
    upcoming = fetch_upcoming_games(SEASON)

    out = {
        "generated_at":  datetime.utcnow().isoformat()+"Z",
        "model_version": "edge-v2",
        "model_stats": {
            "ml_lr_auc":  model.models["ml"]["lr_auc"],
            "ml_xgb_auc": model.models["ml"]["xgb_auc"],
            "spread_mae": model.models["sp"]["mae"],
            "totals_mae": model.models["tot"]["mae"],
        },
        "games": [],
    }

    if not upcoming:
        print("No games scheduled today.")
    else:
        for g in upcoming:
            home, away = g["home"], g["away"]
            try:
                pred = model.predict(home, away, log_bets=True)
                pred["game_time"] = g.get("time","TBD")
                out["games"].append(pred)

                ml  = pred["moneyline"]
                sp  = pred["spread"]
                tot = pred["totals"]
                print(f"\n{'--'*24}\n  {pred['matchup']}  ({g.get('time','TBD')})")
                print(f"  ML : home {ml['home_win_prob']:.1%} | away {ml['away_win_prob']:.1%}")
                print(f"       LR={ml['lr_prob']:.1%} XGB={ml['xgb_prob']:.1%} "
                      f"ELO={ml['elo_prob']:.1%} "
                      f"{'[AGREE]' if ml['model_consensus'] else '[SPLIT]'}")
                print(f"  Spread: model={sp['predicted_margin']:+.1f} pts")
                print(f"  Total:  model={tot['predicted_total']:.1f} pts")
                if pred["fatigue_alert"]:
                    for a in pred["fatigue_alert"]:
                        print(f"  *** B2B: {a}")
            except Exception as e:
                print(f"  Error {home} vs {away}: {e}")
                out["games"].append({"matchup":f"{home} vs {away}","error":str(e)})

    # Bet log summary
    bet_log = load_bet_log()
    out["bet_log_summary"] = bet_log.get("summary",{})
    out["pending_bets"]    = [b for b in bet_log.get("bets",[]) if b["result"]=="PENDING"]

    path = os.path.join(os.path.dirname(__file__),"predictions.json")
    with open(path,"w") as f:
        json.dump(out,f,indent=2)
    print(f"\nDone - {len(out['games'])} games written to predictions.json")
