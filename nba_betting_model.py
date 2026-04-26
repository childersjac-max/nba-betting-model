"""
NBA Betting Model - Edge Edition v5
=====================================
Data sources (all work in GitHub Actions):
  - balldontlie API      : game scores, upcoming games
  - api.server.nbaapi.com: player stats (pts/reb/ast/FG%/3P%/+- etc) - FREE, no auth
  - nbainjuries package  : official NBA injury reports (Out/Questionable/Doubtful)
  - The Odds API         : live sportsbook lines (optional, free tier)

Enhanced with:
  - Player-level stats: pts/reb/ast/tov/FG%/3P%/FT%/+- /min trends
  - Injury reports: Out/Questionable players with impact scoring
  - Coach rotation trends: minutes concentration, starter load
  - Factor attribution: top reasons why a bet is flagged
"""

import warnings
warnings.filterwarnings("ignore")

import json, os, time, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

try:
    from nbainjuries import injury as nba_injury
    HAS_INJURY_PKG = True
except ImportError:
    HAS_INJURY_PKG = False
    print("[Injuries] nbainjuries package not installed - skipping injury data")


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

API_KEY       = os.environ["BALLDONTLIE_API_KEY"]
ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
BASE_URL      = "https://api.balldontlie.io/nba/v1"
NBAAPI_URL    = "https://api.server.nbaapi.com/api"
HEADERS       = {"Authorization": API_KEY}
SEASON        = 2024       # balldontlie season (2024 = 2024-25)
NBAAPI_SEASON = 2025       # nbaapi.com season (year season ends)

MIN_EDGE       = 0.03
KELLY_FRACTION = 0.25
API_DELAY      = 1.5
MAX_RETRIES    = 6

BET_LOG_PATH      = os.path.join(os.path.dirname(__file__), "bet_log.json")
BACKTEST_LOG_PATH = os.path.join(os.path.dirname(__file__), "backtest.json")

# Player impact weights for lineup quality score
STAT_WEIGHTS = {
    "points":    0.30,
    "plus_minus":0.25,
    "assists":   0.12,
    "rebounds":  0.10,
    "fg_pct":    0.10,
    "fg3_pct":   0.08,
    "steals":    0.03,
    "blocks":    0.02,
}

# NBA team name -> abbreviation map for injury matching
TEAM_NAME_MAP = {
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


# -----------------------------------------------------------------------------
# 1. GAME DATA - balldontlie
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
                resp = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS,
                                    params=params, timeout=30)
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
                               "start_date": today})
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


# -----------------------------------------------------------------------------
# 2. PLAYER STATS - api.server.nbaapi.com (free, no auth, works in CI)
# -----------------------------------------------------------------------------

def fetch_player_totals(season=NBAAPI_SEASON, page_size=100):
    """
    Fetch per-game player stats from community NBA API.
    Returns DataFrame with all players' season averages.
    Includes: pts, reb, ast, tov, stl, blk, fg%, 3p%, ft%, +/-, min, games
    """
    print(f"[Players] Fetching player totals from nbaapi.com (season {season})...")
    all_players = []
    page = 1

    while True:
        try:
            time.sleep(0.3)
            resp = requests.get(
                f"{NBAAPI_URL}/playertotals",
                params={"season": season, "page": page,
                        "pageSize": page_size, "isPlayoff": "false"},
                timeout=20,
            )
            if resp.status_code != 200:
                print(f"  [Players] HTTP {resp.status_code} on page {page}, stopping.")
                break
            data = resp.json()
            players = data.get("data", [])
            if not players:
                break
            all_players.extend(players)
            total_pages = data.get("pagination", {}).get("pages", 1)
            print(f"  Players: {len(all_players)} / page {page} of {total_pages}")
            if page >= total_pages:
                break
            page += 1
        except Exception as e:
            print(f"  [Players] Error page {page}: {e}")
            break

    if not all_players:
        print("[Players] No player data retrieved.")
        return pd.DataFrame()

    df = pd.DataFrame(all_players)
    print(f"[Players] {len(df)} players loaded.")
    return df


def fetch_player_advanced(season=NBAAPI_SEASON):
    """
    Fetch advanced stats: PER, TS%, usage%, win shares, VORP, BPM.
    """
    print(f"[Players] Fetching advanced stats (season {season})...")
    all_players = []
    page = 1

    while True:
        try:
            time.sleep(0.3)
            resp = requests.get(
                f"{NBAAPI_URL}/playeradvancedstats",
                params={"season": season, "page": page,
                        "pageSize": 100, "isPlayoff": "false"},
                timeout=20,
            )
            if resp.status_code != 200:
                break
            data    = resp.json()
            players = data.get("data", [])
            if not players:
                break
            all_players.extend(players)
            total_pages = data.get("pagination", {}).get("pages", 1)
            if page >= total_pages:
                break
            page += 1
        except Exception as e:
            print(f"  [Advanced] Error: {e}")
            break

    if not all_players:
        return pd.DataFrame()

    df = pd.DataFrame(all_players)
    print(f"[Players] {len(df)} advanced player rows loaded.")
    return df


def build_team_player_profiles(totals_df, advanced_df):
    """
    Merge totals + advanced stats and compute team-level features.
    Returns dict keyed by team abbreviation.
    """
    if totals_df.empty:
        return {}

    # Standardise column names from nbaapi.com
    rename_tot = {
        "playerName":    "name",
        "team":          "team",
        "games":         "games",
        "minutesPg":     "min",
        "points":        "pts",
        "totalRb":       "reb",
        "assists":       "ast",
        "steals":        "stl",
        "blocks":        "blk",
        "turnovers":     "tov",
        "fieldPercent":  "fg_pct",
        "threePercent":  "fg3_pct",
        "ftPercent":     "ft_pct",
        "personalFouls": "fouls",
    }
    df = totals_df.rename(columns={k:v for k,v in rename_tot.items() if k in totals_df.columns})

    # Merge advanced if available
    if not advanced_df.empty:
        adv_rename = {
            "playerName": "name", "team": "team",
            "per": "per", "tsPercent": "ts_pct",
            "usagePercent": "usage_pct", "winShares": "win_shares",
            "vorp": "vorp", "box": "bpm",
        }
        adv = advanced_df.rename(columns={k:v for k,v in adv_rename.items()
                                           if k in advanced_df.columns})
        adv_cols = ["name","team"] + [c for c in ["per","ts_pct","usage_pct",
                                                    "win_shares","vorp","bpm"]
                                       if c in adv.columns]
        df = df.merge(adv[adv_cols], on=["name","team"], how="left")

    # Filter to players with meaningful minutes (>= 8 min/game)
    if "min" in df.columns:
        df = df[df["min"] >= 8].copy()

    # Numeric coercion
    num_cols = ["min","pts","reb","ast","stl","blk","tov","fg_pct","fg3_pct",
                "ft_pct","fouls","games","per","ts_pct","usage_pct","win_shares","vorp","bpm"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Build team profiles
    profiles = {}
    for team, grp in df.groupby("team"):
        grp = grp.sort_values("min", ascending=False)
        top8 = grp.head(8)
        if len(top8) == 0:
            continue

        total_min = top8["min"].sum() or 1

        # Weighted plus/minus (we don't have +/- in totals, use win_shares as proxy)
        ws_col  = "win_shares" if "win_shares" in top8.columns else None
        pm_score = float(top8[ws_col].sum()) if ws_col else 0.0

        # Lineup quality score (composite)
        quality = 0.0
        for stat, weight in STAT_WEIGHTS.items():
            col = {"points":"pts","plus_minus":ws_col or "pts",
                   "assists":"ast","rebounds":"reb",
                   "fg_pct":"fg_pct","fg3_pct":"fg3_pct",
                   "steals":"stl","blocks":"blk"}.get(stat)
            if col and col in top8.columns:
                quality += float(top8[col].mean()) * weight

        # Minutes concentration (Herfindahl)
        shares = (top8["min"] / total_min).values
        hhi    = float(np.sum(shares**2))

        # Rotation stress: starters averaging >34 min
        starters      = top8.head(5)
        heavy_load    = int((starters["min"] > 34).sum())
        rotation_stress = heavy_load / 5

        # Usage skew (high = team relies on 1-2 stars)
        usage_skew = float(top8["usage_pct"].std()) if "usage_pct" in top8.columns else 0

        # Advanced metrics
        team_per    = float(top8["per"].mean())        if "per"        in top8.columns else 15.0
        team_ts     = float(top8["ts_pct"].mean())     if "ts_pct"     in top8.columns else 0.55
        team_usage  = float(top8["usage_pct"].mean())  if "usage_pct"  in top8.columns else 20.0
        team_vorp   = float(top8["vorp"].sum())        if "vorp"       in top8.columns else 0.0
        team_bpm    = float(top8["bpm"].mean())        if "bpm"        in top8.columns else 0.0

        # Shooting
        team_fg  = float(top8["fg_pct"].mean())  if "fg_pct"  in top8.columns else 0.45
        team_3p  = float(top8["fg3_pct"].mean()) if "fg3_pct" in top8.columns else 0.35
        team_ft  = float(top8["ft_pct"].mean())  if "ft_pct"  in top8.columns else 0.75
        team_tov = float(top8["tov"].mean())     if "tov"     in top8.columns else 2.5
        team_ast = float(top8["ast"].mean())     if "ast"     in top8.columns else 3.0
        team_pts = float(top8["pts"].mean())     if "pts"     in top8.columns else 10.0

        profiles[team] = {
            "quality_score":    round(quality, 3),
            "win_shares_total": round(pm_score, 2),
            "lineup_hhi":       round(hhi, 3),
            "lineup_depth":     round(1 - hhi, 3),
            "rotation_stress":  round(rotation_stress, 3),
            "usage_skew":       round(usage_skew, 3),
            "team_per":         round(team_per, 2),
            "team_ts_pct":      round(team_ts, 3),
            "team_usage":       round(team_usage, 2),
            "team_vorp":        round(team_vorp, 2),
            "team_bpm":         round(team_bpm, 2),
            "fg_pct":           round(team_fg, 3),
            "fg3_pct":          round(team_3p, 3),
            "ft_pct":           round(team_ft, 3),
            "avg_tov":          round(team_tov, 2),
            "avg_ast":          round(team_ast, 2),
            "avg_pts_per_player": round(team_pts, 2),
            # Top players for display
            "top_players": [
                {
                    "name":  row.get("name",""),
                    "min":   round(float(row.get("min",0)), 1),
                    "pts":   round(float(row.get("pts",0)), 1),
                    "reb":   round(float(row.get("reb",0)), 1),
                    "ast":   round(float(row.get("ast",0)), 1),
                    "fg_pct":round(float(row.get("fg_pct",0)), 3),
                    "fg3_pct":round(float(row.get("fg3_pct",0)), 3),
                    "tov":   round(float(row.get("tov",0)), 1),
                    "per":   round(float(row.get("per",0)), 1),
                    "vorp":  round(float(row.get("vorp",0)), 2),
                }
                for _, row in top8.iterrows()
            ],
        }

    print(f"[Players] Built profiles for {len(profiles)} teams.")
    return profiles


# -----------------------------------------------------------------------------
# 3. INJURY REPORTS - nbainjuries package (official NBA data, works in CI)
# -----------------------------------------------------------------------------

def fetch_injury_report():
    """
    Fetch today's official NBA injury report.
    Returns dict: {team_abbr: [{"player": name, "status": Out/Questionable, "reason": ...}]}
    """
    if not HAS_INJURY_PKG:
        return {}

    print("[Injuries] Fetching official NBA injury report...")
    try:
        now    = datetime.now()
        report = nba_injury.get_reportdata(now, return_df=False)
        if not report:
            print("[Injuries] No injury data for current time.")
            return {}

        injuries = {}
        for entry in report:
            team_full = entry.get("Team", "")
            abbr      = TEAM_NAME_MAP.get(team_full)
            if not abbr:
                continue
            player = entry.get("Player Name", "")
            status = entry.get("Current Status", "")
            reason = entry.get("Reason", "")
            if abbr not in injuries:
                injuries[abbr] = []
            injuries[abbr].append({
                "player": player,
                "status": status,
                "reason": reason,
                "is_out": status.upper() in ("OUT", "INACTIVE", "SUSPENSION"),
                "is_questionable": "QUESTIONABLE" in status.upper(),
            })

        total = sum(len(v) for v in injuries.values())
        print(f"[Injuries] {total} player statuses loaded across {len(injuries)} teams.")
        return injuries

    except Exception as e:
        print(f"[Injuries] Error: {e}")
        return {}


def compute_injury_impact(team_abbr, injury_report, player_profiles):
    """
    Score the impact of injuries on a team using player quality data.
    Returns:
      - injury_impact_score: 0-1 (higher = worse for the team)
      - out_players: list of confirmed out players
      - questionable_players: list of questionable players
      - star_out: bool (top 2 player is out)
    """
    out_players  = []
    q_players    = []
    star_out     = False

    team_injuries = injury_report.get(team_abbr, [])
    for inj in team_injuries:
        if inj["is_out"]:
            out_players.append(inj["player"])
        elif inj["is_questionable"]:
            q_players.append(inj["player"])

    # Cross-reference with player profiles to estimate impact
    impact_score = 0.0
    if player_profiles and team_abbr in player_profiles:
        top_players = player_profiles[team_abbr].get("top_players", [])
        if top_players:
            total_pts = sum(p["pts"] for p in top_players) or 1
            # Check if any top players are out/questionable
            for i, player in enumerate(top_players[:8]):
                pname = player["name"].lower()
                for out_p in out_players:
                    last = out_p.split(",")[0].strip().lower() if "," in out_p else out_p.split()[-1].lower()
                    if last in pname or pname in out_p.lower():
                        # Weight by scoring share
                        share = player["pts"] / total_pts
                        impact_score += share * 1.0  # full impact if out
                        if i < 2:
                            star_out = True
                for q_p in q_players:
                    last = q_p.split(",")[0].strip().lower() if "," in q_p else q_p.split()[-1].lower()
                    if last in pname or pname in q_p.lower():
                        share = player["pts"] / total_pts
                        impact_score += share * 0.4  # partial impact if questionable

    return {
        "impact_score":        round(min(impact_score, 1.0), 3),
        "out_players":         out_players,
        "questionable_players": q_players,
        "star_out":            star_out,
        "total_missing":       len(out_players) + len(q_players),
    }


# -----------------------------------------------------------------------------
# 4. GAME LOG FEATURES
# -----------------------------------------------------------------------------

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
    away["result"]     = (1 - away["home_win"]).astype(int)
    logs = pd.concat([home, away], ignore_index=True)
    return logs.sort_values(["team","date"]).reset_index(drop=True)


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
        grp["win_rate10_c"]    = grp["result"].shift(1).rolling(10, min_periods=3).mean()
        grp["momentum"]        = grp["win_rate3"] - grp["win_rate10_c"]
        grp["pace_proxy"]      = grp["pts_for"] + grp["pts_against"]
        grp["pace_roll5"]      = grp["pace_proxy"].shift(1).rolling(5,  min_periods=2).mean()
        grp["pace_roll10"]     = grp["pace_proxy"].shift(1).rolling(10, min_periods=3).mean()
        grp["scoring_var10"]   = grp["pts_for"].shift(1).rolling(10, min_periods=3).std()
        roll_avg               = grp["pts_for"].shift(1).rolling(10, min_periods=3).mean()
        grp["scoring_drop"]    = (roll_avg - grp["pts_for"].shift(1)).clip(lower=0)
        grp["inj_signal"]      = grp["scoring_drop"].shift(1).rolling(5, min_periods=1).mean()
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
        "inj_signal","home_scoring","away_scoring","days_rest","is_b2b",
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
    df["home_elo"]     = h_elos;  df["away_elo"]     = a_elos
    df["elo_diff"]     = df["home_elo"]-df["away_elo"]
    df["elo_win_prob"] = 1/(1+10**(-df["elo_diff"]/400))
    return df


# -----------------------------------------------------------------------------
# 5. FEATURE SETS
# -----------------------------------------------------------------------------

BASE_ML = [
    "elo_diff","elo_win_prob",
    "diff_pts_for_roll10","diff_pts_against_roll10","diff_point_diff_roll10",
    "diff_pts_for_roll5","diff_pts_against_roll5",
    "diff_win_rate10","diff_win_rate5","home_win_rate10","away_win_rate10",
    "diff_momentum","diff_win_ewm","diff_point_diff_ewm",
    "diff_days_rest","home_is_b2b","away_is_b2b",
    "diff_inj_signal","home_inj_signal","away_inj_signal",
]
PLAYER_ML = [
    "diff_quality_score","diff_team_bpm","diff_team_vorp",
    "diff_team_per","diff_team_ts_pct",
    "diff_fg_pct","diff_fg3_pct","diff_avg_tov",
    "home_injury_impact","away_injury_impact","diff_injury_impact",
    "home_star_out","away_star_out",
    "diff_lineup_depth","diff_rotation_stress",
]
BASE_SP = [
    "elo_diff",
    "diff_pts_for_roll10","diff_pts_against_roll10","diff_point_diff_roll10",
    "diff_pts_for_roll5","diff_pts_against_roll5",
    "diff_momentum","diff_point_diff_ewm",
    "home_pts_for_roll10","home_pts_against_roll10",
    "away_pts_for_roll10","away_pts_against_roll10",
    "diff_days_rest","home_is_b2b","away_is_b2b","diff_inj_signal",
    "home_home_scoring_avg","away_away_scoring_avg",
]
PLAYER_SP = [
    "diff_quality_score","diff_team_bpm","diff_team_per",
    "diff_fg_pct","diff_avg_tov",
    "home_injury_impact","away_injury_impact","diff_injury_impact",
    "home_star_out","away_star_out",
]
BASE_TOT = [
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
PLAYER_TOT = [
    "diff_fg_pct","diff_fg3_pct","diff_ft_pct",
    "home_team_ts_pct","away_team_ts_pct",
    "home_injury_impact","away_injury_impact",
    "diff_quality_score",
]

def avail(df, cols): return [c for c in cols if c in df.columns]


# -----------------------------------------------------------------------------
# 6. MODELS
# -----------------------------------------------------------------------------

def train_moneyline(games, has_player=False):
    base  = avail(games, BASE_ML)
    extra = avail(games, PLAYER_ML) if has_player else []
    f     = base + [x for x in extra if x not in base]
    X, y  = games[f].values, games["home_win"].values
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
    tag = "(+player stats)" if has_player else "(game scores only)"
    print(f"[ML]  LR AUC={lr_auc:.3f}  XGB AUC={xgb_auc:.3f} {tag}")
    importance = sorted(zip(f, xgbc.feature_importances_),
                        key=lambda x: x[1], reverse=True)[:12]
    return {"lr":pipe,"xgb":xgbc,"features":f,
            "lr_auc":round(lr_auc,3),"xgb_auc":round(xgb_auc,3),
            "feature_importance":importance,"has_player":has_player}


def train_spread(games, has_player=False):
    base  = avail(games, BASE_SP)
    extra = avail(games, PLAYER_SP) if has_player else []
    f     = base + [x for x in extra if x not in base]
    X, y  = games[f].values, games["point_diff"].values
    pipe  = Pipeline([("sc",StandardScaler()),("rg",Ridge(alpha=10.0))])
    pipe.fit(X,y)
    mae = -cross_val_score(pipe,X,y,cv=TimeSeriesSplit(5),
                           scoring="neg_mean_absolute_error").mean()
    print(f"[SP]  Ridge MAE={mae:.2f} pts")
    return {"model":pipe,"features":f,"mae":round(mae,2)}


def train_totals(games, has_player=False):
    base  = avail(games, BASE_TOT)
    extra = avail(games, PLAYER_TOT) if has_player else []
    f     = base + [x for x in extra if x not in base]
    X, y  = games[f].values, games["total_pts"].values
    pipe  = Pipeline([("sc",StandardScaler()),("rg",Ridge(alpha=10.0))])
    pipe.fit(X,y)
    mae = -cross_val_score(pipe,X,y,cv=TimeSeriesSplit(5),
                           scoring="neg_mean_absolute_error").mean()
    print(f"[TOT] Ridge MAE={mae:.2f} pts")
    return {"model":pipe,"features":f,"mae":round(mae,2)}


# -----------------------------------------------------------------------------
# 7. FACTOR ATTRIBUTION
# -----------------------------------------------------------------------------

FACTOR_LABELS = {
    "elo_diff":                  ("ELO rating advantage", "Team strength gap based on season results"),
    "elo_win_prob":              ("ELO win probability",  "Historical win rate implied by ELO"),
    "diff_point_diff_roll10":    ("10-game margin edge",  "Average winning margin differential, last 10"),
    "diff_point_diff_roll5":     ("5-game margin edge",   "Recent scoring margin differential"),
    "diff_pts_for_roll10":       ("Scoring advantage",    "Offensive output gap over 10 games"),
    "diff_pts_against_roll10":   ("Defensive advantage",  "Points allowed differential"),
    "diff_win_rate10":           ("Win rate edge",        "Win percentage gap last 10 games"),
    "diff_momentum":             ("Momentum edge",        "Recent form vs season average"),
    "diff_win_ewm":              ("Exponential momentum", "Recency-weighted form advantage"),
    "diff_point_diff_ewm":       ("Scoring trend",        "Weighted recent scoring margin"),
    "home_is_b2b":               ("Home B2B fatigue",     "Home team on second night of B2B"),
    "away_is_b2b":               ("Away B2B fatigue",     "Away team on second night of B2B"),
    "diff_days_rest":            ("Rest advantage",       "Days of rest differential"),
    "diff_inj_signal":           ("Injury signal",        "Scoring drop proxy for injuries"),
    "diff_quality_score":        ("Roster quality edge",  "Composite player quality differential"),
    "diff_team_bpm":             ("Box Plus/Minus edge",  "Team BPM differential (advanced)"),
    "diff_team_vorp":            ("VORP advantage",       "Value over replacement player gap"),
    "diff_team_per":             ("PER advantage",        "Player efficiency rating differential"),
    "diff_team_ts_pct":          ("True shooting edge",   "Team TS% differential (shooting efficiency)"),
    "diff_fg_pct":               ("FG% advantage",        "Field goal percentage gap"),
    "diff_fg3_pct":              ("3PT% advantage",       "Three-point shooting gap"),
    "diff_avg_tov":              ("Turnover edge",        "Average turnovers differential"),
    "home_injury_impact":        ("Home injury impact",   "Estimated pts lost to home injuries"),
    "away_injury_impact":        ("Away injury impact",   "Estimated pts lost to away injuries"),
    "diff_injury_impact":        ("Injury advantage",     "Relative injury burden differential"),
    "home_star_out":             ("Home star missing",    "Top-2 home player confirmed out"),
    "away_star_out":             ("Away star missing",    "Top-2 away player confirmed out"),
    "diff_lineup_depth":         ("Depth advantage",      "Roster depth differential"),
    "diff_rotation_stress":      ("Rotation load",        "Starter fatigue from heavy minutes"),
    "diff_pace_roll10":          ("Pace mismatch",        "Team pace differential (fast vs slow)"),
}


def get_top_factors(feat_row, features, lr_model, top_n=5):
    """
    Attribute prediction to top contributing features using LR coefficients.
    Returns list of factor dicts with label, direction, strength, detail.
    """
    try:
        lr_clf = lr_model["lr"].named_steps["cl"].estimator
        scaler = lr_model["lr"].named_steps["sc"]
        x_raw  = np.array([[feat_row.get(f,0) for f in features]])
        x_sc   = scaler.transform(x_raw)[0]
        coefs  = lr_clf.coef_[0]
        contribs = [(features[i], x_sc[i]*coefs[i]) for i in range(len(features))]
        contribs.sort(key=lambda x: abs(x[1]), reverse=True)
        factors = []
        for feat, contrib in contribs[:top_n]:
            if abs(contrib) < 0.005:
                continue
            label, detail = FACTOR_LABELS.get(feat, (feat.replace("_"," ").title(), ""))
            direction = "+" if contrib > 0 else "-"
            strength  = "Strong" if abs(contrib) > 0.25 else "Moderate" if abs(contrib) > 0.12 else "Mild"
            factors.append({
                "feature":   feat,
                "label":     label,
                "detail":    detail,
                "direction": direction,
                "strength":  strength,
                "contrib":   round(contrib, 4),
                "raw_val":   round(feat_row.get(feat, 0), 3),
            })
        return factors
    except Exception:
        return []


# -----------------------------------------------------------------------------
# 8. BACKTEST
# -----------------------------------------------------------------------------

def run_backtest(games_df, model_ml, model_sp, model_tot):
    print("[Backtest] Running walk-forward backtest...")
    games_df = games_df.sort_values("date").reset_index(drop=True)
    split    = int(len(games_df)*0.70)
    test     = games_df.iloc[split:].copy().reset_index(drop=True)
    if len(test) < 20:
        return {}

    ml_f  = [f for f in model_ml["features"]  if f in test.columns]
    sp_f  = [f for f in model_sp["features"]  if f in test.columns]
    tot_f = [f for f in model_tot["features"] if f in test.columns]

    ml_probs  = model_ml["lr"].predict_proba(test[ml_f].fillna(0).values)[:,1]
    sp_preds  = model_sp["model"].predict(test[sp_f].fillna(0).values)
    tot_preds = model_tot["model"].predict(test[tot_f].fillna(0).values)

    test["ml_prob"]    = ml_probs
    test["sp_pred"]    = sp_preds
    test["tot_pred"]   = tot_preds
    test["ml_correct"] = ((test["ml_prob"]>0.5)==(test["home_win"]==1)).astype(int)

    BET_THRESHOLD = 0.55; FLAT_BET = 100; bankroll = 1000.0
    bankroll_hist = [bankroll]; ml_bets = []
    for _, row in test.iterrows():
        p = row["ml_prob"]
        if p>BET_THRESHOLD or (1-p)>BET_THRESHOLD:
            side_p = p if p>BET_THRESHOLD else 1-p
            won    = (p>BET_THRESHOLD and row["home_win"]==1) or \
                     ((1-p)>BET_THRESHOLD and row["home_win"]==0)
            pnl    = FLAT_BET*0.909 if won else -FLAT_BET
            bankroll += pnl
            ml_bets.append({"prob":round(side_p,3),"won":won,"pnl":round(pnl,2)})
            bankroll_hist.append(round(bankroll,2))

    bins=[0.45,0.50,0.55,0.60,0.65,0.70,0.75,1.0]; calibration=[]
    for i in range(len(bins)-1):
        lo,hi=bins[i],bins[i+1]
        mask=(ml_probs>=lo)&(ml_probs<hi)
        if mask.sum()>0:
            calibration.append({
                "bucket":f"{lo:.0%}-{hi:.0%}",
                "predicted":round(float(ml_probs[mask].mean()),3),
                "actual":round(float(test.loc[mask,"home_win"].mean()),3),
                "n":int(mask.sum()),
                "diff":round(float(test.loc[mask,"home_win"].mean())-float(ml_probs[mask].mean()),3)
            })

    sp_e=abs(test["sp_pred"]-test["point_diff"])
    tot_e=abs(test["tot_pred"]-test["total_pts"])
    n_bets=len(ml_bets); n_wins=sum(1 for b in ml_bets if b["won"])
    roi=((bankroll-1000)/(n_bets*FLAT_BET)*100) if n_bets>0 else 0

    test["month"]=pd.to_datetime(test["date"]).dt.strftime("%b %Y")
    monthly=[{"month":m,"games":len(g),"accuracy":round(float(g["ml_correct"].mean()),3),
              "avg_prob":round(float(g["ml_prob"].mean()),3)}
             for m,g in test.groupby("month")]

    feat_imp = [{"feature":f,"label":FACTOR_LABELS.get(f,(f,""))[0],
                 "importance":round(float(v),4)}
                for f,v in model_ml.get("feature_importance",[])]

    result = {
        "generated_at":datetime.utcnow().isoformat()+"Z",
        "test_games":len(test),"train_games":split,
        "has_player_data":model_ml.get("has_player",False),
        "ml":{"accuracy":round(float(test["ml_correct"].mean()),3),
              "bets_placed":n_bets,"bets_won":n_wins,
              "win_rate":round(n_wins/n_bets if n_bets>0 else 0,3),
              "roi_pct":round(roi,2),"starting_bankroll":1000,
              "ending_bankroll":round(bankroll,2),
              "pnl":round(bankroll-1000,2),
              "bankroll_history":bankroll_hist[:100]},
        "spread":{"mae":round(float(sp_e.mean()),2),
                  "within_5":round(float((sp_e<=5).mean()),3),
                  "within_10":round(float((sp_e<=10).mean()),3)},
        "totals":{"mae":round(float(tot_e.mean()),2),
                  "within_5":round(float((tot_e<=5).mean()),3),
                  "within_10":round(float((tot_e<=10).mean()),3)},
        "calibration":calibration,"monthly":monthly,
        "feature_importance":feat_imp,
    }
    print(f"[Backtest] Acc={result['ml']['accuracy']:.1%} Bets={n_bets} "
          f"W/L={n_wins}/{n_bets-n_wins} ROI={roi:.1f}% $1k->${ bankroll:.0f}")
    with open(BACKTEST_LOG_PATH,"w") as f: json.dump(result,f,indent=2)
    return result


# -----------------------------------------------------------------------------
# 9. LIVE ODDS
# -----------------------------------------------------------------------------

def fetch_live_odds():
    if not ODDS_API_KEY: return {}
    print("[Odds] Fetching live lines...")
    try:
        resp=requests.get("https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
                          params={"apiKey":ODDS_API_KEY,"regions":"us",
                                  "markets":"h2h,spreads,totals","oddsFormat":"american",
                                  "bookmakers":"draftkings,fanduel,betmgm"},timeout=15)
        resp.raise_for_status(); data=resp.json()
    except Exception as e:
        print(f"[Odds] Failed: {e}"); return {}
    odds={}
    for game in data:
        ha=TEAM_NAME_MAP.get(game.get("home_team","")); aa=TEAM_NAME_MAP.get(game.get("away_team",""))
        if not ha or not aa: continue
        best={"h2h_home":None,"h2h_away":None,"spread_home":None,
              "spread_line":None,"total_over":None,"total_line":None}
        for bm in game.get("bookmakers",[]):
            for market in bm.get("markets",[]):
                if market["key"]=="h2h":
                    for o in market["outcomes"]:
                        a=TEAM_NAME_MAP.get(o["name"])
                        if a==ha and best["h2h_home"] is None: best["h2h_home"]=o["price"]
                        elif a==aa and best["h2h_away"] is None: best["h2h_away"]=o["price"]
                elif market["key"]=="spreads":
                    for o in market["outcomes"]:
                        if TEAM_NAME_MAP.get(o["name"])==ha and best["spread_home"] is None:
                            best["spread_home"]=o["price"]; best["spread_line"]=o["point"]
                elif market["key"]=="totals":
                    for o in market["outcomes"]:
                        if o["name"]=="Over" and best["total_over"] is None:
                            best["total_over"]=o["price"]; best["total_line"]=o["point"]
        odds[(ha,aa)]=best
    print(f"[Odds] {len(odds)} games with lines."); return odds


# -----------------------------------------------------------------------------
# 10. BETTING EDGE
# -----------------------------------------------------------------------------

def to_prob(o): return 100/(o+100) if o>0 else abs(o)/(abs(o)+100)
def remove_vig(h,a): t=h+a; return h/t,a/t
def kelly(p,o):
    b=o/100 if o>0 else 100/abs(o)
    return max(0.0,((b*p-(1-p))/b)*KELLY_FRACTION)

def ev(p,o,label=""):
    imp=to_prob(o); e=p-imp
    return {"label":label,"model_prob":round(p,3),"implied_prob":round(imp,3),
            "edge":round(e,3),"kelly_pct":round(kelly(p,o)*100 if e>MIN_EDGE else 0,1),
            "bet":e>MIN_EDGE}

def ensemble_prob(lr_model,xgb_model,features,feat_row):
    x=np.array([[feat_row.get(f,0) for f in features]])
    lr_p=float(lr_model.predict_proba(x)[0][1])
    xgb_p=float(xgb_model.predict_proba(x)[0][1])
    return 0.45*lr_p+0.55*xgb_p,lr_p,xgb_p


def build_recommendations(hwp, pd_, pt, book_odds, factors, home, away):
    recs=[]; book=book_odds or {}
    boh=book.get("h2h_home"); boa=book.get("h2h_away")
    bsl=book.get("spread_line"); btl=book.get("total_line")
    top_factor_labels=[f["label"] for f in factors[:3]] if factors else []

    if boh and boa:
        ih=to_prob(boh); ia=to_prob(boa)
        for prob,edge_val,side,odds,label in [
            (hwp,hwp-ih,"HOME",boh,f"{home} ML"),
            (1-hwp,(1-hwp)-ia,"AWAY",boa,f"{away} ML"),
        ]:
            if edge_val>MIN_EDGE:
                grade="A" if edge_val>0.06 else "B" if edge_val>0.04 else "C"
                recs.append({"type":"ML","side":side,"odds":odds,
                             "model_prob":round(prob,3),
                             "book_prob":round(ih if side=="HOME" else ia,3),
                             "edge":round(edge_val,3),
                             "kelly_pct":round(kelly(prob,odds)*100,1),
                             "grade":grade,"label":label,
                             "reason":f"Model {prob:.1%} vs book {(ih if side=='HOME' else ia):.1%}",
                             "top_factors":top_factor_labels})

    if bsl is not None and pd_ is not None:
        diff=pd_-bsl
        if abs(diff)>=3:
            side="HOME" if diff>0 else "AWAY"
            sp_p=min(0.65,max(0.35,0.5+diff*0.025))
            se=sp_p-to_prob(-110)
            if se>MIN_EDGE:
                grade="A" if abs(diff)>=6 else "B" if abs(diff)>=4 else "C"
                recs.append({"type":"SPREAD","side":side,"odds":-110,
                             "model_line":round(pd_,1),"book_line":bsl,
                             "diff":round(diff,1),"edge":round(se,3),
                             "kelly_pct":round(kelly(sp_p,-110)*100,1),
                             "grade":grade,"label":f"{home if side=='HOME' else away} {bsl:+.1f}",
                             "reason":f"Model {pd_:+.1f} vs book {bsl:+.1f} ({diff:+.1f} gap)",
                             "top_factors":top_factor_labels})

    if btl is not None and pt is not None:
        diff=pt-btl
        if abs(diff)>=3:
            side="OVER" if diff>0 else "UNDER"
            tp=min(0.65,max(0.35,0.5+diff*0.025))
            te=tp-to_prob(-110)
            if te>MIN_EDGE:
                grade="A" if abs(diff)>=6 else "B" if abs(diff)>=4 else "C"
                recs.append({"type":"TOTAL","side":side,"odds":-110,
                             "model_line":round(pt,1),"book_line":btl,
                             "diff":round(diff,1),"edge":round(te,3),
                             "kelly_pct":round(kelly(tp,-110)*100,1),
                             "grade":grade,"label":f"{side} {btl}",
                             "reason":f"Model {pt:.1f} vs book {btl:.1f} ({diff:+.1f})",
                             "top_factors":top_factor_labels})
    return recs


def load_bet_log():
    if os.path.exists(BET_LOG_PATH):
        with open(BET_LOG_PATH) as f: return json.load(f)
    return {"bets":[],"summary":{}}


# -----------------------------------------------------------------------------
# 11. MAIN MODEL CLASS
# -----------------------------------------------------------------------------

class NBABettingModel:
    def __init__(self,season=SEASON):
        self.season=season; self.models={}; self.games=None; self.raw=None
        self.player_profiles={}; self.has_player=False

    def fit(self):
        print("="*55+"\nNBA Betting Model - Edge v5\n"+"="*55)

        # Game data
        games  = fetch_games(self.season)
        logs   = build_team_logs(games)
        rolled = build_all_features(logs)
        gf     = build_game_level_features(rolled)
        gf     = compute_elo(gf)
        self.games=gf; self.raw=rolled

        # Player data
        totals_df  = fetch_player_totals(NBAAPI_SEASON)
        advanced_df= fetch_player_advanced(NBAAPI_SEASON)
        self.player_profiles = build_team_player_profiles(totals_df, advanced_df)
        self.has_player = bool(self.player_profiles)

        # Train models
        self.models["ml"]  = train_moneyline(gf, has_player=False)
        self.models["sp"]  = train_spread(gf,    has_player=False)
        self.models["tot"] = train_totals(gf,     has_player=False)

        if self.has_player:
            print(f"[Players] Player profiles loaded for {len(self.player_profiles)} teams.")
        print("Done.\n")

    def _game_feats(self, team, is_home):
        prefix="home" if is_home else "away"
        tdf=self.raw[self.raw["team"]==team].sort_values("date")
        if tdf.empty: raise ValueError(f"Team '{team}' not found.")
        latest=tdf.iloc[-1]
        feat_keys=[c for c in tdf.columns if any(x in c for x in [
            "_roll","win_rate","ewm","momentum","pace","scoring",
            "inj_signal","home_scoring","away_scoring","days_rest","is_b2b",
        ])]
        return {f"{prefix}_{c}":latest[c] for c in feat_keys}

    def predict(self, home, away, book_odds=None, injury_report=None):
        fr={**self._game_feats(home,True), **self._game_feats(away,False)}
        h_elo=self.games[self.games["home_team"]==home].sort_values("date").iloc[-1]["home_elo"]
        a_elo=self.games[self.games["away_team"]==away].sort_values("date").iloc[-1]["away_elo"]
        fr["elo_diff"]=h_elo-a_elo
        fr["elo_win_prob"]=1/(1+10**(-fr["elo_diff"]/400))
        for s in {k[5:] for k in fr if k.startswith("home_")}:
            fr[f"diff_{s}"]=fr.get(f"home_{s}",0)-fr.get(f"away_{s}",0)

        # Player features
        hp=self.player_profiles.get(home,{}); ap=self.player_profiles.get(away,{})
        player_fields=["quality_score","team_bpm","team_vorp","team_per","team_ts_pct",
                       "fg_pct","fg3_pct","ft_pct","avg_tov","lineup_depth","rotation_stress"]
        for key in player_fields:
            fr[f"home_{key}"]=hp.get(key,0)
            fr[f"away_{key}"]=ap.get(key,0)
            fr[f"diff_{key}"]=hp.get(key,0)-ap.get(key,0)

        # Injury features
        inj_h=compute_injury_impact(home,injury_report or {},self.player_profiles)
        inj_a=compute_injury_impact(away,injury_report or {},self.player_profiles)
        fr["home_injury_impact"]=inj_h["impact_score"]
        fr["away_injury_impact"]=inj_a["impact_score"]
        fr["diff_injury_impact"]=inj_a["impact_score"]-inj_h["impact_score"]  # positive = away worse
        fr["home_star_out"]=1.0 if inj_h["star_out"] else 0.0
        fr["away_star_out"]=1.0 if inj_a["star_out"] else 0.0

        m=self.models["ml"]
        hwp,lr_p,xgb_p=ensemble_prob(m["lr"],m["xgb"],m["features"],fr)

        sp=self.models["sp"]
        pd_=float(sp["model"].predict(np.array([[fr.get(f,0) for f in sp["features"]]]))[0])

        tot=self.models["tot"]
        pt=float(tot["model"].predict(np.array([[fr.get(f,0) for f in tot["features"]]]))[0])

        factors=get_top_factors(fr,m["features"],m,top_n=6)
        recs=build_recommendations(hwp,pd_,pt,book_odds or {},factors,home,away)

        return {
            "matchup":f"{home} vs {away}","home_team":home,"away_team":away,
            "generated_at":datetime.utcnow().isoformat()+"Z",
            "moneyline":{"home_win_prob":round(hwp,3),"away_win_prob":round(1-hwp,3),
                         "lr_prob":round(lr_p,3),"xgb_prob":round(xgb_p,3),
                         "elo_prob":round(fr["elo_win_prob"],3),
                         "model_consensus":abs(lr_p-xgb_p)<0.05},
            "spread":{"predicted_margin":round(pd_,2),"model_mae":sp["mae"]},
            "totals":{"predicted_total":round(pt,1),"model_mae":tot["mae"]},
            "book_odds":book_odds or {},
            "recommendations":recs,
            "top_factors":factors,
            "injuries":{
                "home":{"out":inj_h["out_players"],"questionable":inj_h["questionable_players"],
                        "impact":inj_h["impact_score"],"star_out":inj_h["star_out"]},
                "away":{"out":inj_a["out_players"],"questionable":inj_a["questionable_players"],
                        "impact":inj_a["impact_score"],"star_out":inj_a["star_out"]},
            },
            "player_profiles":{
                "home":{"top_players":hp.get("top_players",[])[:5],
                        "team_bpm":hp.get("team_bpm",0),"team_per":hp.get("team_per",0),
                        "team_vorp":hp.get("team_vorp",0),"fg_pct":hp.get("fg_pct",0),
                        "fg3_pct":hp.get("fg3_pct",0),"quality_score":hp.get("quality_score",0)},
                "away":{"top_players":ap.get("top_players",[])[:5],
                        "team_bpm":ap.get("team_bpm",0),"team_per":ap.get("team_per",0),
                        "team_vorp":ap.get("team_vorp",0),"fg_pct":ap.get("fg_pct",0),
                        "fg3_pct":ap.get("fg3_pct",0),"quality_score":ap.get("quality_score",0)},
                "has_data":self.has_player,
            },
            "fatigue_alert":(["B2B: "+home] if fr.get("home_is_b2b",0) else [])+
                            (["B2B: "+away] if fr.get("away_is_b2b",0) else []),
            "elo":{"home":round(h_elo,0),"away":round(a_elo,0),"diff":round(h_elo-a_elo,0)},
        }


# -----------------------------------------------------------------------------
# 12. ENTRY POINT
# -----------------------------------------------------------------------------

if __name__=="__main__":
    model=NBABettingModel(season=SEASON)
    model.fit()

    backtest=run_backtest(model.games,model.models["ml"],
                          model.models["sp"],model.models["tot"])

    injury_report=fetch_injury_report()
    live_odds=fetch_live_odds()
    upcoming=fetch_upcoming_games(SEASON)

    out={"generated_at":datetime.utcnow().isoformat()+"Z","model_version":"edge-v5",
         "has_player_data":model.has_player,"has_injury_data":bool(injury_report),
         "model_stats":{"ml_lr_auc":model.models["ml"]["lr_auc"],
                        "ml_xgb_auc":model.models["ml"]["xgb_auc"],
                        "spread_mae":model.models["sp"]["mae"],
                        "totals_mae":model.models["tot"]["mae"]},
         "games":[],"all_recs":[],"has_live_odds":bool(live_odds)}

    if not upcoming:
        print("No games scheduled today.")
    else:
        for g in upcoming:
            home,away=g["home"],g["away"]
            try:
                book=live_odds.get((home,away)) or live_odds.get((away,home),{})
                pred=model.predict(home,away,book_odds=book,injury_report=injury_report)
                pred["game_time"]=g.get("time","TBD")
                out["games"].append(pred); out["all_recs"].extend(pred["recommendations"])
                ml=pred["moneyline"]; recs=pred["recommendations"]
                print(f"\n  {pred['matchup']} | home {ml['home_win_prob']:.1%} away {ml['away_win_prob']:.1%}")
                if recs:
                    for r in recs:
                        print(f"  [{r['grade']}] {r['type']} {r['side']} edge={r['edge']:+.1%}")
                        print(f"      Factors: {', '.join(r.get('top_factors',[]))}")
                inj=pred.get("injuries",{})
                if inj.get("home",{}).get("out"): print(f"  OUT ({home}): {inj['home']['out']}")
                if inj.get("away",{}).get("out"): print(f"  OUT ({away}): {inj['away']['out']}")
            except Exception as e:
                print(f"  Error {home} vs {away}: {e}")
                out["games"].append({"matchup":f"{home} vs {away}","error":str(e)})

    bet_log=load_bet_log()
    out["bet_log_summary"]=bet_log.get("summary",{})
    out["pending_bets"]=[b for b in bet_log.get("bets",[]) if b["result"]=="PENDING"]

    path=os.path.join(os.path.dirname(__file__),"predictions.json")
    with open(path,"w") as f: json.dump(out,f,indent=2)
    print(f"\nDone - {len(out['games'])} games, {len(out['all_recs'])} recs")
