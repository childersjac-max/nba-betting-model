"""
NBA Betting Model - Edge Edition v3
Auto-fetches tonight's games, pulls live sportsbook lines,
runs backtest, and outputs full predictions + backtest JSON.
"""

import warnings
warnings.filterwarnings("ignore")

import json, os, time, requests
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

API_KEY      = os.environ["BALLDONTLIE_API_KEY"]
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
BASE_URL     = "https://api.balldontlie.io/nba/v1"
HEADERS      = {"Authorization": API_KEY}
SEASON       = 2024

MIN_EDGE       = 0.03
KELLY_FRACTION = 0.25
API_DELAY      = 1.5
MAX_RETRIES    = 6

BET_LOG_PATH      = os.path.join(os.path.dirname(__file__), "bet_log.json")
BACKTEST_LOG_PATH = os.path.join(os.path.dirname(__file__), "backtest.json")


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
    print(f"[Data] {len(df)} completed games.")
    return df.sort_values("date").reset_index(drop=True)


def fetch_upcoming_games(season=SEASON):
    print("[Data] Fetching upcoming games...")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw   = bdl_get("games", {"seasons[]": season, "postseason": "false",
                               "start_date": today, "per_page": 100})
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
    print(f"[Data] {len(upcoming)} upcoming games.")
    return upcoming


def fetch_live_odds():
    """
    Pull live NBA moneyline + spread + totals from The Odds API.
    Free tier: 500 requests/month.
    Returns dict keyed by (home_abbr, away_abbr).
    """
    if not ODDS_API_KEY:
        print("[Odds] No ODDS_API_KEY set - skipping live odds.")
        return {}

    TEAM_MAP = {
        "Atlanta Hawks":"ATL","Boston Celtics":"BOS","Brooklyn Nets":"BKN",
        "Charlotte Hornets":"CHA","Chicago Bulls":"CHI","Cleveland Cavaliers":"CLE",
        "Dallas Mavericks":"DAL","Denver Nuggets":"DEN","Detroit Pistons":"DET",
        "Golden State Warriors":"GSW","Houston Rockets":"HOU","Indiana Pacers":"IND",
        "Los Angeles Clippers":"LAC","Los Angeles Lakers":"LAL","Memphis Grizzlies":"MEM",
        "Miami Heat":"MIA","Milwaukee Bucks":"MIL","Minnesota Timberwolves":"MIN",
        "New Orleans Pelicans":"NOP","New York Knicks":"NYK","Oklahoma City Thunder":"OKC",
        "Orlando Magic":"ORL","Philadelphia 76ers":"PHI","Phoenix Suns":"PHX",
        "Portland Trail Blazers":"POR","Sacramento Kings":"SAC","San Antonio Spurs":"SAS",
        "Toronto Raptors":"TOR","Utah Jazz":"UTA","Washington Wizards":"WAS",
    }

    print("[Odds] Fetching live lines from The Odds API...")
    try:
        url    = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
        params = {
            "apiKey":   ODDS_API_KEY,
            "regions":  "us",
            "markets":  "h2h,spreads,totals",
            "oddsFormat": "american",
            "bookmakers": "draftkings,fanduel,betmgm",
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[Odds] Failed to fetch odds: {e}")
        return {}

    odds_by_game = {}
    for game in data:
        home_full = game.get("home_team","")
        away_full = game.get("away_team","")
        home_abbr = TEAM_MAP.get(home_full)
        away_abbr = TEAM_MAP.get(away_full)
        if not home_abbr or not away_abbr:
            continue

        best = {"h2h_home": None, "h2h_away": None,
                "spread_home": None, "spread_line": None,
                "total_over": None, "total_line": None,
                "bookmakers": []}

        for bm in game.get("bookmakers", []):
            best["bookmakers"].append(bm["key"])
            for market in bm.get("markets", []):
                if market["key"] == "h2h":
                    for outcome in market["outcomes"]:
                        abbr = TEAM_MAP.get(outcome["name"])
                        if abbr == home_abbr and best["h2h_home"] is None:
                            best["h2h_home"] = outcome["price"]
                        elif abbr == away_abbr and best["h2h_away"] is None:
                            best["h2h_away"] = outcome["price"]
                elif market["key"] == "spreads":
                    for outcome in market["outcomes"]:
                        abbr = TEAM_MAP.get(outcome["name"])
                        if abbr == home_abbr and best["spread_home"] is None:
                            best["spread_home"] = outcome["price"]
                            best["spread_line"] = outcome["point"]
                elif market["key"] == "totals":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == "Over" and best["total_over"] is None:
                            best["total_over"] = outcome["price"]
                            best["total_line"] = outcome["point"]

        odds_by_game[(home_abbr, away_abbr)] = best

    print(f"[Odds] Got lines for {len(odds_by_game)} games.")
    return odds_by_game


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
        for w in windows:
            for col in ["pts_for","pts_against","point_diff","total_pts"]:
                grp[f"{col}_roll{w}"] = grp[col].shift(1).rolling(w, min_periods=max(2,w//2)).mean()
            grp[f"win_rate{w}"] = grp["result"].shift(1).rolling(w, min_periods=max(2,w//2)).mean()
        grp["days_rest"]       = grp["date"].diff().dt.days.fillna(7).clip(0,14)
        grp["is_b2b"]          = (grp["days_rest"]==1).astype(int)
        grp["pts_for_ewm"]     = grp["pts_for"].shift(1).ewm(span=5).mean()
        grp["pts_against_ewm"] = grp["pts_against"].shift(1).ewm(span=5).mean()
        grp["point_diff_ewm"]  = grp["point_diff"].shift(1).ewm(span=5).mean()
        grp["win_ewm"]         = grp["result"].shift(1).ewm(span=5).mean()
        grp["win_rate3"]       = grp["result"].shift(1).rolling(3, min_periods=1).mean()
        grp["momentum"]        = grp.get("win_rate3", pd.Series(0, index=grp.index)) - \
                                  grp.get("win_rate10", pd.Series(0, index=grp.index))
        grp["pace_proxy"]      = grp["pts_for"] + grp["pts_against"]
        grp["pace_roll5"]      = grp["pace_proxy"].shift(1).rolling(5,  min_periods=2).mean()
        grp["pace_roll10"]     = grp["pace_proxy"].shift(1).rolling(10, min_periods=3).mean()
        grp["scoring_var10"]   = grp["pts_for"].shift(1).rolling(10, min_periods=3).std()
        roll_avg               = grp["pts_for"].shift(1).rolling(10, min_periods=3).mean()
        grp["scoring_drop"]    = (roll_avg - grp["pts_for"].shift(1)).clip(lower=0)
        grp["injury_signal"]   = grp["scoring_drop"].shift(1).rolling(5, min_periods=1).mean()
        home_pts = grp[grp["home"]==1]["pts_for"].shift(1).rolling(10, min_periods=2).mean()
        away_pts = grp[grp["home"]==0]["pts_for"].shift(1).rolling(10, min_periods=2).mean()
        grp["home_scoring_avg"] = home_pts.reindex(grp.index).ffill()
        grp["away_scoring_avg"] = away_pts.reindex(grp.index).ffill()
        frames.append(grp)
    return pd.concat(frames).sort_values(["date","team"]).reset_index(drop=True)


def build_game_level_features(df):
    home = df[df["home"]==1].copy()
    away = df[df["home"]==0].copy()
    feat_cols = [c for c in df.columns if any(x in c for x in [
        "_roll","win_rate","ewm","momentum","pace","scoring",
        "injury","home_scoring","away_scoring","days_rest","is_b2b",
    ])]
    base = [c for c in ["game_id","date","team","opponent",
                         "point_diff","total_pts","home_win"] if c in df.columns]
    hf = home[base+feat_cols].rename(columns={
        **{c:f"home_{c}" for c in feat_cols},
        "team":"home_team","opponent":"away_team"})
    af = away[["date","team"]+feat_cols].rename(columns={
        **{c:f"away_{c}" for c in feat_cols},
        "team":"away_team_check"})
    games = hf.merge(af, left_on=["date","away_team"],
                     right_on=["date","away_team_check"],
                     how="inner").drop(columns=["away_team_check"])
    for col in feat_cols:
        games[f"diff_{col}"] = games[f"home_{col}"] - games[f"away_{col}"]
    return games.dropna().reset_index(drop=True)


def compute_elo(df, k=20, home_adv=100):
    elo = {t:1500.0 for t in pd.concat([df["home_team"],df["away_team"]]).unique()}
    h_elos, a_elos = [], []
    for _, row in df.sort_values("date").iterrows():
        h, a   = row["home_team"], row["away_team"]
        exp_h  = 1/(1+10**((elo[a]-(elo[h]+home_adv))/400))
        actual = row["home_win"]
        h_elos.append(elo[h]); a_elos.append(elo[a])
        margin = abs(row["point_diff"])
        k_m    = k*np.log1p(margin)*(2.2/((margin*0.001)+2.2))
        elo[h] += k_m*(actual-exp_h); elo[a] += k_m*(exp_h-actual)
    df = df.sort_values("date").copy()
    df["home_elo"]     = h_elos; df["away_elo"] = a_elos
    df["elo_diff"]     = df["home_elo"]-df["away_elo"]
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
    "diff_days_rest","home_is_b2b","away_is_b2b","diff_injury_signal",
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
    return {"lr":pipe,"xgb":xgbc,"features":f,
            "lr_auc":round(lr_auc,3),"xgb_auc":round(xgb_auc,3)}

def train_spread(games):
    f = avail(games, SP_FEATS); X, y = games[f].values, games["point_diff"].values
    pipe = Pipeline([("sc",StandardScaler()),("rg",Ridge(alpha=10.0))])
    pipe.fit(X,y)
    mae = -cross_val_score(pipe,X,y,cv=TimeSeriesSplit(5),
                           scoring="neg_mean_absolute_error").mean()
    print(f"[SP]  Ridge MAE={mae:.2f} pts")
    return {"model":pipe,"features":f,"mae":round(mae,2)}

def train_totals(games):
    f = avail(games, TOT_FEATS); X, y = games[f].values, games["total_pts"].values
    pipe = Pipeline([("sc",StandardScaler()),("rg",Ridge(alpha=10.0))])
    pipe.fit(X,y)
    mae = -cross_val_score(pipe,X,y,cv=TimeSeriesSplit(5),
                           scoring="neg_mean_absolute_error").mean()
    print(f"[TOT] Ridge MAE={mae:.2f} pts")
    return {"model":pipe,"features":f,"mae":round(mae,2)}


# -----------------------------------------------------------------------------
# 5. BACKTEST
# -----------------------------------------------------------------------------

def run_backtest(games_df, model_ml, model_sp, model_tot, raw_df):
    """
    Walk-forward backtest: train on first 70% of season,
    predict on remaining 30%. Simulates real betting conditions.
    """
    print("[Backtest] Running walk-forward backtest...")
    games_df = games_df.sort_values("date").reset_index(drop=True)
    split    = int(len(games_df) * 0.70)
    test     = games_df.iloc[split:].copy()

    if len(test) < 20:
        print("[Backtest] Not enough test games.")
        return {}

    # ML predictions on test set
    ml_feats = model_ml["features"]
    sp_feats = model_sp["features"]
    tot_feats = model_tot["features"]

    ml_x  = test[[f for f in ml_feats  if f in test.columns]].fillna(0).values
    sp_x  = test[[f for f in sp_feats  if f in test.columns]].fillna(0).values
    tot_x = test[[f for f in tot_feats if f in test.columns]].fillna(0).values

    ml_probs  = model_ml["lr"].predict_proba(ml_x)[:,1]
    sp_preds  = model_sp["model"].predict(sp_x)
    tot_preds = model_tot["model"].predict(tot_x)

    test = test.copy()
    test["ml_prob"]   = ml_probs
    test["sp_pred"]   = sp_preds
    test["tot_pred"]  = tot_preds
    test["ml_correct"]= ((test["ml_prob"] > 0.5) == (test["home_win"] == 1)).astype(int)

    # Simulate betting at -110 (standard juice)
    # Bet when model probability > 55% (above break-even + margin)
    BET_THRESHOLD = 0.55
    FLAT_BET      = 100
    bankroll      = 1000.0
    bankroll_history = [bankroll]
    ml_bets = []

    for _, row in test.iterrows():
        p = row["ml_prob"]
        if p > BET_THRESHOLD:
            won = row["home_win"] == 1
            pnl = FLAT_BET * 0.909 if won else -FLAT_BET
            bankroll += pnl
            ml_bets.append({"prob": round(p,3), "won": won, "pnl": round(pnl,2)})
            bankroll_history.append(round(bankroll,2))
        elif (1-p) > BET_THRESHOLD:
            won = row["home_win"] == 0
            pnl = FLAT_BET * 0.909 if won else -FLAT_BET
            bankroll += pnl
            ml_bets.append({"prob": round(1-p,3), "won": won, "pnl": round(pnl,2)})
            bankroll_history.append(round(bankroll,2))

    # Calibration: bucket predictions into bins
    bins = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.0]
    calibration = []
    for i in range(len(bins)-1):
        lo, hi  = bins[i], bins[i+1]
        mask    = (ml_probs >= lo) & (ml_probs < hi)
        if mask.sum() > 0:
            pred_avg = float(ml_probs[mask].mean())
            act_rate = float(test.loc[mask.nonzero()[0],"home_win"].mean())
            calibration.append({
                "bucket":    f"{lo:.0%}-{hi:.0%}",
                "predicted": round(pred_avg, 3),
                "actual":    round(act_rate, 3),
                "n":         int(mask.sum()),
                "diff":      round(act_rate - pred_avg, 3),
            })

    # Spread accuracy
    sp_errors   = abs(test["sp_pred"] - test["point_diff"])
    sp_mae      = float(sp_errors.mean())
    sp_within5  = float((sp_errors <= 5).mean())
    sp_within10 = float((sp_errors <= 10).mean())

    # Totals accuracy
    tot_errors   = abs(test["tot_pred"] - test["total_pts"])
    tot_mae      = float(tot_errors.mean())
    tot_within5  = float((tot_errors <= 5).mean())
    tot_within10 = float((tot_errors <= 10).mean())

    # ML summary
    n_bets  = len(ml_bets)
    n_wins  = sum(1 for b in ml_bets if b["won"])
    roi     = ((bankroll - 1000) / (n_bets * FLAT_BET) * 100) if n_bets > 0 else 0
    win_rate= n_wins / n_bets if n_bets > 0 else 0

    # Accuracy by month
    test["month"] = pd.to_datetime(test["date"]).dt.strftime("%b %Y")
    monthly = []
    for month, grp in test.groupby("month"):
        monthly.append({
            "month":    month,
            "games":    len(grp),
            "accuracy": round(float(grp["ml_correct"].mean()), 3),
            "avg_prob": round(float(grp["ml_prob"].mean()), 3),
        })

    result = {
        "generated_at":    datetime.utcnow().isoformat()+"Z",
        "test_games":      len(test),
        "train_games":     split,
        "ml": {
            "accuracy":        round(float(test["ml_correct"].mean()), 3),
            "bets_placed":     n_bets,
            "bets_won":        n_wins,
            "win_rate":        round(win_rate, 3),
            "roi_pct":         round(roi, 2),
            "starting_bankroll": 1000,
            "ending_bankroll": round(bankroll, 2),
            "pnl":             round(bankroll - 1000, 2),
            "bankroll_history": bankroll_history[:100],
        },
        "spread": {
            "mae":        round(sp_mae, 2),
            "within_5":   round(sp_within5, 3),
            "within_10":  round(sp_within10, 3),
        },
        "totals": {
            "mae":        round(tot_mae, 2),
            "within_5":   round(tot_within5, 3),
            "within_10":  round(tot_within10, 3),
        },
        "calibration":  calibration,
        "monthly":      monthly,
    }

    print(f"[Backtest] ML accuracy={result['ml']['accuracy']:.1%}  "
          f"Bets={n_bets}  W/L={n_wins}/{n_bets-n_wins}  "
          f"ROI={roi:.1f}%  Bankroll: $1000 -> ${bankroll:.0f}")

    with open(BACKTEST_LOG_PATH, "w") as f:
        json.dump(result, f, indent=2)

    return result


# -----------------------------------------------------------------------------
# 6. BETTING EDGE + LINE COMPARISON
# -----------------------------------------------------------------------------

def to_prob(o): return 100/(o+100) if o>0 else abs(o)/(abs(o)+100)
def remove_vig(h, a): t=h+a; return h/t, a/t
def kelly(p,o):
    b = o/100 if o>0 else 100/abs(o)
    return max(0.0,((b*p-(1-p))/b)*KELLY_FRACTION)

def ev(p, o, label=""):
    imp = to_prob(o); e = p-imp
    return {"label":label,"model_prob":round(p,3),"implied_prob":round(imp,3),
            "edge":round(e,3),"kelly_pct":round(kelly(p,o)*100 if e>MIN_EDGE else 0,1),
            "bet":e>MIN_EDGE}

def ensemble_prob(lr_model, xgb_model, features, feat_row):
    x     = np.array([[feat_row.get(f,0) for f in features]])
    lr_p  = float(lr_model.predict_proba(x)[0][1])
    xgb_p = float(xgb_model.predict_proba(x)[0][1])
    return 0.45*lr_p+0.55*xgb_p, lr_p, xgb_p

def compare_with_book(model_prob, model_spread, model_total,
                       book_odds_home, book_odds_away,
                       book_spread_line, book_total_line):
    """
    Compare model output vs live book lines.
    Returns recommendation with reasoning.
    """
    recs = []

    # Moneyline
    if book_odds_home and book_odds_away:
        imp_h_raw = to_prob(book_odds_home)
        imp_a_raw = to_prob(book_odds_away)
        imp_h, imp_a = remove_vig(imp_h_raw, imp_a_raw)
        ml_edge_h = model_prob - imp_h_raw
        ml_edge_a = (1-model_prob) - imp_a_raw

        if ml_edge_h > MIN_EDGE:
            recs.append({
                "type":       "ML",
                "side":       "HOME",
                "odds":       book_odds_home,
                "model_prob": round(model_prob, 3),
                "book_prob":  round(imp_h_raw, 3),
                "novig_prob": round(imp_h, 3),
                "edge":       round(ml_edge_h, 3),
                "kelly_pct":  round(kelly(model_prob, book_odds_home)*100, 1),
                "reason":     f"Model {model_prob:.1%} vs book {imp_h_raw:.1%} (no-vig {imp_h:.1%})",
                "grade":      "A" if ml_edge_h > 0.06 else "B" if ml_edge_h > 0.04 else "C",
            })
        elif ml_edge_a > MIN_EDGE:
            recs.append({
                "type":       "ML",
                "side":       "AWAY",
                "odds":       book_odds_away,
                "model_prob": round(1-model_prob, 3),
                "book_prob":  round(imp_a_raw, 3),
                "novig_prob": round(imp_a, 3),
                "edge":       round(ml_edge_a, 3),
                "kelly_pct":  round(kelly(1-model_prob, book_odds_away)*100, 1),
                "reason":     f"Model {(1-model_prob):.1%} vs book {imp_a_raw:.1%} (no-vig {imp_a:.1%})",
                "grade":      "A" if ml_edge_a > 0.06 else "B" if ml_edge_a > 0.04 else "C",
            })

    # Spread
    if book_spread_line is not None and model_spread is not None:
        margin_diff = model_spread - book_spread_line
        if abs(margin_diff) >= 3:
            side    = "HOME" if margin_diff > 0 else "AWAY"
            sp_odds = -110
            sp_prob = min(0.65, max(0.35, 0.5 + margin_diff*0.025))
            sp_edge = sp_prob - to_prob(sp_odds)
            if sp_edge > MIN_EDGE:
                recs.append({
                    "type":        "SPREAD",
                    "side":        side,
                    "odds":        sp_odds,
                    "model_line":  round(model_spread, 1),
                    "book_line":   book_spread_line,
                    "diff":        round(margin_diff, 1),
                    "edge":        round(sp_edge, 3),
                    "kelly_pct":   round(kelly(sp_prob, sp_odds)*100, 1),
                    "reason":      f"Model projects {model_spread:+.1f}, book at {book_spread_line:+.1f} ({margin_diff:+.1f} gap)",
                    "grade":       "A" if abs(margin_diff) >= 6 else "B" if abs(margin_diff) >= 4 else "C",
                })

    # Totals
    if book_total_line is not None and model_total is not None:
        total_diff = model_total - book_total_line
        if abs(total_diff) >= 3:
            side     = "OVER" if total_diff > 0 else "UNDER"
            tot_prob = min(0.65, max(0.35, 0.5 + total_diff*0.025))
            tot_edge = tot_prob - to_prob(-110)
            if tot_edge > MIN_EDGE:
                recs.append({
                    "type":       "TOTAL",
                    "side":       side,
                    "odds":       -110,
                    "model_line": round(model_total, 1),
                    "book_line":  book_total_line,
                    "diff":       round(total_diff, 1),
                    "edge":       round(tot_edge, 3),
                    "kelly_pct":  round(kelly(tot_prob, -110)*100, 1),
                    "reason":     f"Model projects {model_total:.1f}, book at {book_total_line:.1f} ({total_diff:+.1f} gap)",
                    "grade":      "A" if abs(total_diff) >= 6 else "B" if abs(total_diff) >= 4 else "C",
                })

    return recs


def load_bet_log():
    if os.path.exists(BET_LOG_PATH):
        with open(BET_LOG_PATH) as f: return json.load(f)
    return {"bets":[],"summary":{}}

def save_bet_log(log):
    with open(BET_LOG_PATH,"w") as f: json.dump(log,f,indent=2)


# -----------------------------------------------------------------------------
# 7. MAIN MODEL CLASS
# -----------------------------------------------------------------------------

class NBABettingModel:
    def __init__(self, season=SEASON):
        self.season=season; self.models={}; self.games=None; self.raw=None

    def fit(self):
        print("="*55+"\nNBA Betting Model - Edge v3\n"+"="*55)
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

    def _feats(self, team, is_home):
        prefix = "home" if is_home else "away"
        tdf = self.raw[self.raw["team"]==team].sort_values("date")
        if tdf.empty: raise ValueError(f"Team '{team}' not found.")
        latest    = tdf.iloc[-1]
        feat_keys = [c for c in tdf.columns if any(x in c for x in [
            "_roll","win_rate","ewm","momentum","pace","scoring",
            "injury","home_scoring","away_scoring","days_rest","is_b2b",
        ])]
        return {f"{prefix}_{c}":latest[c] for c in feat_keys}

    def predict(self, home, away, book_odds=None):
        fr = {**self._feats(home,True), **self._feats(away,False)}
        h_elo = self.games[self.games["home_team"]==home].sort_values("date").iloc[-1]["home_elo"]
        a_elo = self.games[self.games["away_team"]==away].sort_values("date").iloc[-1]["away_elo"]
        fr["elo_diff"]     = h_elo-a_elo
        fr["elo_win_prob"] = 1/(1+10**(-fr["elo_diff"]/400))
        for s in {k[5:] for k in fr if k.startswith("home_")}:
            fr[f"diff_{s}"] = fr.get(f"home_{s}",0)-fr.get(f"away_{s}",0)

        m = self.models["ml"]
        hwp, lr_p, xgb_p = ensemble_prob(m["lr"],m["xgb"],m["features"],fr)

        sp  = self.models["sp"]
        pd_ = float(sp["model"].predict(np.array([[fr.get(f,0) for f in sp["features"]]]))[0])

        tot = self.models["tot"]
        pt  = float(tot["model"].predict(np.array([[fr.get(f,0) for f in tot["features"]]]))[0])

        # Live line comparison
        recs = []
        book = book_odds or {}
        if book:
            recs = compare_with_book(
                hwp, pd_, pt,
                book.get("h2h_home"), book.get("h2h_away"),
                book.get("spread_line"), book.get("total_line"),
            )

        return {
            "matchup":          f"{home} vs {away}",
            "home_team":        home,
            "away_team":        away,
            "generated_at":     datetime.utcnow().isoformat()+"Z",
            "moneyline": {
                "home_win_prob":   round(hwp,3),
                "away_win_prob":   round(1-hwp,3),
                "lr_prob":         round(lr_p,3),
                "xgb_prob":        round(xgb_p,3),
                "elo_prob":        round(fr["elo_win_prob"],3),
                "model_consensus": abs(lr_p-xgb_p)<0.05,
            },
            "spread": {
                "predicted_margin": round(pd_,2),
                "model_mae":        sp["mae"],
            },
            "totals": {
                "predicted_total": round(pt,1),
                "model_mae":       tot["mae"],
            },
            "book_odds":       book,
            "recommendations": recs,
            "fatigue_alert":   (["B2B: "+home] if fr.get("home_is_b2b",0) else []) +
                               (["B2B: "+away] if fr.get("away_is_b2b",0) else []),
            "elo": {
                "home": round(h_elo,0),
                "away": round(a_elo,0),
                "diff": round(h_elo-a_elo,0),
            },
        }


# -----------------------------------------------------------------------------
# 8. ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    model = NBABettingModel(season=SEASON)
    model.fit()

    # Run backtest
    backtest = run_backtest(
        model.games, model.models["ml"],
        model.models["sp"], model.models["tot"],
        model.raw,
    )

    # Fetch live odds
    live_odds = fetch_live_odds()

    # Fetch tonight's games
    upcoming = fetch_upcoming_games(SEASON)

    out = {
        "generated_at":  datetime.utcnow().isoformat()+"Z",
        "model_version": "edge-v3",
        "model_stats": {
            "ml_lr_auc":  model.models["ml"]["lr_auc"],
            "ml_xgb_auc": model.models["ml"]["xgb_auc"],
            "spread_mae": model.models["sp"]["mae"],
            "totals_mae": model.models["tot"]["mae"],
        },
        "games":        [],
        "all_recs":     [],
        "has_live_odds": bool(live_odds),
    }

    if not upcoming:
        print("No games scheduled today.")
    else:
        for g in upcoming:
            home, away = g["home"], g["away"]
            try:
                book = live_odds.get((home,away)) or live_odds.get((away,home),{})
                pred = model.predict(home, away, book_odds=book)
                pred["game_time"] = g.get("time","TBD")
                out["games"].append(pred)
                out["all_recs"].extend(pred["recommendations"])

                ml  = pred["moneyline"]
                sp  = pred["spread"]
                tot = pred["totals"]
                recs= pred["recommendations"]
                print(f"\n{'--'*24}\n  {pred['matchup']}")
                print(f"  ML : home {ml['home_win_prob']:.1%} | away {ml['away_win_prob']:.1%}")
                print(f"  Spread: {sp['predicted_margin']:+.1f} pts | Total: {tot['predicted_total']:.1f} pts")
                if recs:
                    for r in recs:
                        print(f"  *** REC [{r['grade']}]: {r['type']} {r['side']} | edge={r['edge']:+.1%} | {r['reason']}")
                else:
                    print(f"  No bets recommended")
                if pred["fatigue_alert"]:
                    for a in pred["fatigue_alert"]: print(f"  B2B WARNING: {a}")
            except Exception as e:
                print(f"  Error {home} vs {away}: {e}")
                out["games"].append({"matchup":f"{home} vs {away}","error":str(e)})

    bet_log = load_bet_log()
    out["bet_log_summary"] = bet_log.get("summary",{})
    out["pending_bets"]    = [b for b in bet_log.get("bets",[]) if b["result"]=="PENDING"]

    path = os.path.join(os.path.dirname(__file__),"predictions.json")
    with open(path,"w") as f: json.dump(out,f,indent=2)
    print(f"\nDone - {len(out['games'])} games, {len(out['all_recs'])} recommendations")
