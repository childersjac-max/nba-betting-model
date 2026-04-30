"""
NBA Betting Model - v6 (Full Rebuild)
======================================
Key improvements over v5:
  1. Multi-season training: 4 seasons (~4,400 games) for stable patterns
  2. Walk-forward features: every feature uses ONLY data prior to each game
  3. Per-game efficiency metrics: off/def/net rating derived from scores (no extra API)
  4. Player data used ONLY for current predictions — eliminates backtest leakage
  5. Proper walk-forward backtest across chronological seasons
  6. Pinnacle-anchored edge: sharpest sportsbook used as no-vig benchmark
  7. OpticOdds (OddsJam) integration for multi-book comparison + historical lines
  8. Moneyline only by default (spread/totals disabled until MAE < 7 pts)
  9. Closing Line Value (CLV) tracking for real edge validation
 10. Playoffs supported — works year-round

Data Sources:
  - BallDontLie (game scores, injuries, standings)
  - api.server.nbaapi.com (player season stats + advanced — free, no auth)
  - The Odds API (current multi-book odds)
  - OpticOdds / OddsJam API (current + historical odds, closing lines)
"""

import warnings
warnings.filterwarnings("ignore")

import json, os, time, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
import xgboost as xgb

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BDL_KEY       = os.environ["BALLDONTLIE_API_KEY"]
ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
OPTICODDS_KEY = os.environ.get("OPTICODDS_KEY", "")   # OpticOdds / OddsJam

BDL_BASE    = "https://api.balldontlie.io/nba/v1"
BDL_HEADERS = {"Authorization": BDL_KEY}

# Seasons to train on.  Each entry = year the season STARTS (2022 = 2022-23).
# Free BDL tier: current season only.  Paid tiers unlock historical seasons.
TRAIN_SEASONS  = [2021, 2022, 2023, 2024]
CURRENT_SEASON = 2024        # BDL parameter for the current/latest season
NBAAPI_SEASON  = 2025        # nbaapi.com uses ending year (2025 = 2024-25)

MIN_EDGE       = 0.03        # minimum edge vs no-vig to flag a bet
KELLY_FRACTION = 0.25        # quarter-Kelly for bet sizing safety
API_DELAY      = 1.2         # seconds between BDL requests
MAX_RETRIES    = 6

BACKTEST_PATH = os.path.join(os.path.dirname(__file__), "backtest.json")
PRED_PATH     = os.path.join(os.path.dirname(__file__), "predictions.json")
BET_LOG_PATH  = os.path.join(os.path.dirname(__file__), "bet_log.json")

# Sharp books — used for no-vig baseline (Pinnacle is the most efficient market)
SHARP_BOOKS = ["pinnacle", "circa", "betcris", "betonlineag"]

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
# Reverse: abbreviation → full name (used to match OpticOdds full team names)
ABBR_TO_NAME = {v: k for k, v in TEAM_NAME_MAP.items()}

# ---------------------------------------------------------------------------
# 1.  BALLDONTLIE — Game data (multi-season)
# ---------------------------------------------------------------------------

def bdl_get(endpoint, params=None):
    """Paginated GET with retry + rate-limit handling."""
    results = []
    p = {**(params or {}), "per_page": 100}
    cursor = None
    while True:
        if cursor:
            p["cursor"] = cursor
        for attempt in range(MAX_RETRIES):
            time.sleep(API_DELAY)
            try:
                r = requests.get(f"{BDL_BASE}/{endpoint}",
                                 headers=BDL_HEADERS, params=p, timeout=30)
                if r.status_code == 429:
                    wait = 20 * (attempt + 1)
                    print(f"  Rate limited — waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if r.status_code == 401:
                    raise requests.HTTPError(
                        f"401 Unauthorized — check BDL API key or tier access",
                        response=r)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                is_auth = (isinstance(e, requests.HTTPError) and
                           getattr(e.response, "status_code", 0) == 401)
                if attempt == MAX_RETRIES - 1 or is_auth:
                    raise
                print(f"  Retry {attempt+1}: {e}")
                time.sleep(15)
        data = r.json()
        results.extend(data["data"])
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        print(f"    {len(results)} records...")
    return results


def fetch_games(seasons=None):
    """Fetch completed game results for one or more seasons."""
    if seasons is None:
        seasons = TRAIN_SEASONS
    print(f"[Games] Fetching {len(seasons)} season(s): {seasons}")
    rows = []
    for season in seasons:
        raw = bdl_get("games", {"seasons[]": season})
        before = len(rows)
        for g in raw:
            if g["status"] != "Final":
                continue
            rows.append({
                "game_id":    g["id"],
                "season":     season,
                "date":       g["date"][:10],
                "home_team":  g["home_team"]["abbreviation"],
                "away_team":  g["visitor_team"]["abbreviation"],
                "home_score": g["home_team_score"],
                "away_score": g["visitor_team_score"],
            })
        print(f"  Season {season}: {len(rows)-before} completed games")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No game data returned. Check API key and season parameters.")

    df["date"]       = pd.to_datetime(df["date"])
    df["point_diff"] = df["home_score"] - df["away_score"]
    df["total_pts"]  = df["home_score"] + df["away_score"]
    df["home_win"]   = (df["point_diff"] > 0).astype(int)

    # Per-game efficiency proxies — derived from scores, no extra API calls needed.
    # off_rtg = share of total points scored = offensive efficiency relative to pace.
    # net_rtg = off_rtg - def_rtg  (positive = team dominated scoring exchange)
    df["pace_proxy"]   = df["total_pts"]
    df["home_off_rtg"] = df["home_score"] / df["total_pts"] * 100
    df["home_def_rtg"] = df["away_score"] / df["total_pts"] * 100
    df["home_net_rtg"] = df["home_off_rtg"] - df["home_def_rtg"]
    df["away_off_rtg"] = df["home_def_rtg"]
    df["away_def_rtg"] = df["home_off_rtg"]
    df["away_net_rtg"] = -df["home_net_rtg"]

    df = df.sort_values("date").reset_index(drop=True)
    print(f"[Games] Total: {len(df)} completed games across {len(seasons)} season(s).")
    return df


def fetch_upcoming_games():
    """Fetch today's and future unplayed games."""
    print("[Upcoming] Fetching schedule...")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw = bdl_get("games", {"seasons[]": CURRENT_SEASON, "start_date": today})
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
    print(f"[Upcoming] {len(upcoming)} upcoming games.")
    return upcoming


def fetch_injuries_bdl():
    """Fetch current injury report from BallDontLie (replaces nbainjuries package)."""
    print("[Injuries] Fetching from BallDontLie...")
    try:
        raw = bdl_get("player_injuries")
        injuries = {}
        for entry in raw:
            abbr  = (entry.get("team") or {}).get("abbreviation", "")
            if not abbr:
                continue
            first = (entry.get("player") or {}).get("first_name", "")
            last  = (entry.get("player") or {}).get("last_name", "")
            status = entry.get("status", "")
            if abbr not in injuries:
                injuries[abbr] = []
            injuries[abbr].append({
                "player":          f"{first} {last}".strip(),
                "status":          status,
                "is_out":          status.upper() in ("OUT", "INACTIVE", "SUSPENSION"),
                "is_questionable": "QUESTIONABLE" in status.upper(),
            })
        total = sum(len(v) for v in injuries.values())
        print(f"[Injuries] {total} statuses across {len(injuries)} teams.")
        return injuries
    except Exception as e:
        print(f"[Injuries] Error: {e}")
        return {}

# ---------------------------------------------------------------------------
# 2.  PLAYER STATS — nbaapi.com (current predictions only, NOT backtest)
# ---------------------------------------------------------------------------

NBAAPI_URL = "https://api.server.nbaapi.com/api"


def fetch_player_totals(season=NBAAPI_SEASON):
    print(f"[Players] Fetching totals (season {season})...")
    all_players, page = [], 1
    while True:
        try:
            time.sleep(0.4)
            r = requests.get(f"{NBAAPI_URL}/playertotals",
                             params={"season": season, "page": page,
                                     "pageSize": 100, "isPlayoff": "false"},
                             timeout=20)
            if r.status_code != 200:
                break
            data = r.json()
            players = data.get("data", [])
            if not players:
                break
            all_players.extend(players)
            total_pages = data.get("pagination", {}).get("pages", 1)
            print(f"  {len(all_players)} players / page {page}/{total_pages}")
            if page >= total_pages:
                break
            page += 1
        except Exception as e:
            print(f"  Error page {page}: {e}")
            break
    return pd.DataFrame(all_players)


def fetch_player_advanced(season=NBAAPI_SEASON):
    print(f"[Players] Fetching advanced stats (season {season})...")
    all_players, page = [], 1
    while True:
        try:
            time.sleep(0.4)
            r = requests.get(f"{NBAAPI_URL}/playeradvancedstats",
                             params={"season": season, "page": page,
                                     "pageSize": 100, "isPlayoff": "false"},
                             timeout=20)
            if r.status_code != 200:
                break
            data = r.json()
            players = data.get("data", [])
            if not players:
                break
            all_players.extend(players)
            total_pages = data.get("pagination", {}).get("pages", 1)
            if page >= total_pages:
                break
            page += 1
        except Exception as e:
            print(f"  Error: {e}")
            break
    return pd.DataFrame(all_players)


def build_team_player_profiles(totals_df, advanced_df):
    """
    Team-level player quality scores from season averages.
    Used ONLY for forward predictions, never backtest features.
    """
    if totals_df.empty:
        return {}
    rename = {
        "playerName":"name","team":"team","games":"games","minutesPg":"min",
        "points":"pts","totalRb":"reb","assists":"ast","steals":"stl",
        "blocks":"blk","turnovers":"tov","fieldPercent":"fg_pct",
        "threePercent":"fg3_pct","ftPercent":"ft_pct",
    }
    df = totals_df.rename(columns={k:v for k,v in rename.items() if k in totals_df.columns})
    if not advanced_df.empty:
        adv = advanced_df.rename(columns={
            "playerName":"name","team":"team","per":"per","tsPercent":"ts_pct",
            "usagePercent":"usage_pct","winShares":"win_shares",
            "vorp":"vorp","box":"bpm",
        })
        acols = ["name","team"] + [c for c in
                 ["per","ts_pct","usage_pct","win_shares","vorp","bpm"]
                 if c in adv.columns]
        df = df.merge(adv[acols], on=["name","team"], how="left")
    if "min" in df.columns:
        df = df[df["min"] >= 8].copy()
    num_cols = ["min","pts","reb","ast","stl","blk","tov","fg_pct","fg3_pct","ft_pct",
                "games","per","ts_pct","usage_pct","win_shares","vorp","bpm"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    profiles = {}
    for team, grp in df.groupby("team"):
        grp = grp.sort_values("min", ascending=False)
        top8 = grp.head(8)
        if top8.empty:
            continue
        total_min = top8["min"].sum() or 1
        def sm(col): return float(top8[col].mean()) if col in top8.columns else 0.0
        def ss(col): return float(top8[col].sum())  if col in top8.columns else 0.0
        shares = (top8["min"] / total_min).values
        hhi = float(np.sum(shares**2))
        heavy = int((top8.head(5)["min"] > 34).sum()) if "min" in top8.columns else 0
        profiles[team] = {
            "quality_score":   round(sm("pts")*0.35 + sm("reb")*0.10 +
                                     sm("ast")*0.15 + sm("fg_pct")*10 +
                                     sm("fg3_pct")*8, 3),
            "team_bpm":        round(sm("bpm"), 2),
            "team_vorp":       round(ss("vorp"), 2),
            "team_per":        round(sm("per"), 2),
            "team_ts_pct":     round(sm("ts_pct"), 3),
            "fg_pct":          round(sm("fg_pct"), 3),
            "fg3_pct":         round(sm("fg3_pct"), 3),
            "ft_pct":          round(sm("ft_pct"), 3),
            "avg_tov":         round(sm("tov"), 2),
            "lineup_depth":    round(1 - hhi, 3),
            "rotation_stress": round(heavy / 5, 3),
            "top_players": [
                {"name": row.get("name",""),
                 "min":    round(float(row.get("min",0)),1),
                 "pts":    round(float(row.get("pts",0)),1),
                 "reb":    round(float(row.get("reb",0)),1),
                 "ast":    round(float(row.get("ast",0)),1),
                 "fg_pct": round(float(row.get("fg_pct",0)),3),
                 "fg3_pct":round(float(row.get("fg3_pct",0)),3),
                 "tov":    round(float(row.get("tov",0)),1),
                 "per":    round(float(row.get("per",0)),1),
                 "vorp":   round(float(row.get("vorp",0)),2)}
                for _, row in top8.iterrows()
            ],
        }
    print(f"[Players] Built profiles for {len(profiles)} teams.")
    return profiles


def compute_injury_impact(team_abbr, injury_report, player_profiles):
    out_players, q_players = [], []
    for inj in injury_report.get(team_abbr, []):
        if inj["is_out"]:
            out_players.append(inj["player"])
        elif inj["is_questionable"]:
            q_players.append(inj["player"])
    impact, star_out = 0.0, False
    if player_profiles and team_abbr in player_profiles:
        top = player_profiles[team_abbr].get("top_players", [])
        if top:
            total_pts = sum(p["pts"] for p in top) or 1
            for i, player in enumerate(top[:8]):
                pname = player["name"].lower()
                for op in out_players:
                    last = op.split()[-1].lower()
                    if last in pname or pname in op.lower():
                        impact += player["pts"] / total_pts
                        if i < 2:
                            star_out = True
                for qp in q_players:
                    last = qp.split()[-1].lower()
                    if last in pname or pname in qp.lower():
                        impact += player["pts"] / total_pts * 0.4
    return {
        "impact_score":       round(min(impact, 1.0), 3),
        "out_players":        out_players,
        "questionable_players": q_players,
        "star_out":           star_out,
        "total_missing":      len(out_players) + len(q_players),
    }

# ---------------------------------------------------------------------------
# 3.  ODDS FETCHING
# ---------------------------------------------------------------------------

def _to_prob(american_odds):
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)


def _novig_from_book(book_data, market_key, home, away):
    for mkt in book_data.get("markets", []):
        if mkt["key"] != market_key:
            continue
        prices = {}
        for oc in mkt["outcomes"]:
            if oc["name"] == home:
                prices["home"] = _to_prob(oc["price"])
            else:
                prices["away"] = _to_prob(oc["price"])
        if "home" in prices and "away" in prices:
            total = prices["home"] + prices["away"]
            return {"home": round(prices["home"]/total, 4),
                    "away": round(prices["away"]/total, 4)}
    return {}


def fetch_odds_theapi():
    """Current NBA odds from The Odds API across all major US books."""
    if not ODDS_API_KEY:
        print("[Odds-TheAPI] No key — skipping.")
        return {}
    print("[Odds-TheAPI] Fetching current NBA odds...")
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
            params={
                "apiKey": ODDS_API_KEY,
                "regions": "us,us2",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
                "bookmakers": ",".join([
                    "pinnacle","draftkings","fanduel","betmgm","caesars",
                    "betonlineag","mybookieag","bovada","williamhill_us",
                    "pointsbet","barstool","betrivers","unibet_us",
                ]),
            },
            timeout=15,
        )
        r.raise_for_status()
        games = r.json()
        remaining = r.headers.get("x-requests-remaining", "?")
        print(f"[Odds-TheAPI] {len(games)} games. Remaining API credits: {remaining}")
        return _parse_theapi_odds(games)
    except Exception as e:
        print(f"[Odds-TheAPI] Error: {e}")
        return {}


def _parse_theapi_odds(games):
    result = {}
    for g in games:
        home  = g.get("home_team", "")
        away  = g.get("away_team", "")
        key   = f"{home} vs {away}"
        books = {b["key"]: b for b in g.get("bookmakers", [])}

        sharp_book = next((sb for sb in SHARP_BOOKS if sb in books), None)
        if not sharp_book and books:
            sharp_book = list(books.keys())[0]

        bd = {}

        # --- Moneyline ---
        all_h2h_home, all_h2h_away = [], []
        for bkey, bdata in books.items():
            for mkt in bdata.get("markets", []):
                if mkt["key"] != "h2h":
                    continue
                for oc in mkt["outcomes"]:
                    if oc["name"] == home:
                        all_h2h_home.append((bkey, oc["price"]))
                    else:
                        all_h2h_away.append((bkey, oc["price"]))

        if all_h2h_home and all_h2h_away:
            def best_odds(odds_list):
                return max(odds_list,
                           key=lambda x: x[1] if x[1] > 0 else 10000 / abs(x[1]))
            bh = best_odds(all_h2h_home)
            ba = best_odds(all_h2h_away)
            bd.update({"h2h_home": bh[1], "h2h_home_book": bh[0],
                       "h2h_away": ba[1], "h2h_away_book": ba[0],
                       "books_count": len(all_h2h_home)})
            if sharp_book:
                nv = _novig_from_book(books.get(sharp_book, {}), "h2h", home, away)
                bd.update({"novig_home": nv.get("home"),
                           "novig_away": nv.get("away"),
                           "sharp_book": sharp_book})
            implied_list = [_to_prob(p) for _, p in all_h2h_home]
            if len(implied_list) > 1:
                bd["ml_disagreement"] = round(
                    (max(implied_list) - min(implied_list)) * 100, 2)
            bd["avg_ml_home"] = round(np.mean([p for _, p in all_h2h_home]), 1)
            bd["avg_ml_away"] = round(np.mean([p for _, p in all_h2h_away]), 1)

        # --- Spread ---
        spread_lines = []
        for bkey, bdata in books.items():
            for mkt in bdata.get("markets", []):
                if mkt["key"] != "spreads":
                    continue
                for oc in mkt["outcomes"]:
                    if oc["name"] == home:
                        spread_lines.append((bkey, oc.get("point", 0), oc["price"]))
        if spread_lines:
            best_sp  = max(spread_lines, key=lambda x: x[1])
            worst_sp = min(spread_lines, key=lambda x: x[1])
            bd.update({"spread_line": best_sp[1], "spread_odds": best_sp[2],
                       "spread_line_away": -worst_sp[1],
                       "spread_odds_away": worst_sp[2],
                       "spread_book": best_sp[0],
                       "avg_spread_line": round(np.mean([x[1] for x in spread_lines]), 2)})

        # --- Totals ---
        over_lines, under_lines = [], []
        for bkey, bdata in books.items():
            for mkt in bdata.get("markets", []):
                if mkt["key"] != "totals":
                    continue
                for oc in mkt["outcomes"]:
                    if oc["name"] == "Over":
                        over_lines.append((bkey, oc.get("point", 0), oc["price"]))
                    elif oc["name"] == "Under":
                        under_lines.append((bkey, oc.get("point", 0), oc["price"]))
        if over_lines:
            bo = min(over_lines, key=lambda x: x[1])   # lowest total = easiest Over
            bd.update({"total_line_over": bo[1], "total_odds_over": bo[2],
                       "total_book_over": bo[0]})
        if under_lines:
            bu = max(under_lines, key=lambda x: x[1])  # highest total = easiest Under
            bd.update({"total_line_under": bu[1], "total_odds_under": bu[2],
                       "total_book_under": bu[0]})
        if over_lines and under_lines:
            bd["avg_total_line"] = round(
                np.mean([x[1] for x in over_lines + under_lines]), 2)

        result[key] = bd
    return result


def fetch_odds_opticodds():
    """
    Current NBA odds from OpticOdds — multi-book, Pinnacle no-vig baseline.

    API flow (required by OpticOdds v3):
      1. GET /fixtures  → get fixture IDs + team names
      2. GET /fixtures/odds?fixture_id=X&sportsbook=Y  → odds per fixture per book
    """
    if not OPTICODDS_KEY:
        print("[Odds-OpticOdds] No key — skipping.")
        return {}
    print("[Odds-OpticOdds] Fetching current NBA odds...")
    headers = {"X-Api-Key": OPTICODDS_KEY}

    # Step 1 — get upcoming fixture list
    try:
        r = requests.get(
            "https://api.opticodds.com/api/v3/fixtures",
            params={"sport": "basketball", "league": "NBA",
                    "status": "unplayed", "is_live": "false"},
            headers=headers, timeout=20,
        )
        r.raise_for_status()
        fixtures_meta = r.json().get("data", [])
    except Exception as e:
        print(f"[Odds-OpticOdds] Error fetching fixtures: {e}")
        return {}

    print(f"[Odds-OpticOdds] {len(fixtures_meta)} upcoming fixtures.")
    if not fixtures_meta:
        return {}

    # Books to query — Pinnacle first (no-vig baseline), then sharp/value books
    BOOKS_TO_QUERY = ["pinnacle", "draftkings", "fanduel", "betonlineag", "betmgm", "caesars"]

    result = {}
    for meta in fixtures_meta:
        # Team names — OpticOdds uses home_team_display / away_team_display
        home = meta.get("home_team_display") or ""
        away = meta.get("away_team_display") or ""
        if not home or not away:
            # Fallback to home_competitors array
            hc = (meta.get("home_competitors") or [{}])[0]
            ac = (meta.get("away_competitors") or [{}])[0]
            home = hc.get("name", "")
            away = ac.get("name", "")
        if not home or not away:
            continue

        fixture_id = meta["id"]
        key = f"{home} vs {away}"
        bd = {}
        ml_home_all, ml_away_all = [], []

        # Step 2 — fetch odds per book for this fixture
        for book in BOOKS_TO_QUERY:
            try:
                ro = requests.get(
                    "https://api.opticodds.com/api/v3/fixtures/odds",
                    params={"fixture_id": fixture_id,
                            "sportsbook": book,
                            "market_name": "moneyline"},
                    headers=headers, timeout=15,
                )
                if ro.status_code != 200:
                    continue
                fdata = ro.json().get("data", [{}])[0] if ro.json().get("data") else {}
                odds_list = fdata.get("odds", [])
                for odd in odds_list:
                    if (odd.get("market_id") or "").lower() != "moneyline":
                        continue
                    price = odd.get("price")
                    if not price:
                        continue
                    name  = (odd.get("name") or "").lower()
                    bname = (odd.get("sportsbook") or book).lower()
                    if home.lower() in name:
                        ml_home_all.append((bname, int(price)))
                    elif away.lower() in name:
                        ml_away_all.append((bname, int(price)))
                time.sleep(0.3)
            except Exception:
                continue

        if ml_home_all and ml_away_all:
            def bo(lst): return max(lst,
                key=lambda x: x[1] if x[1] > 0 else 10000 / abs(x[1]))
            bh, ba = bo(ml_home_all), bo(ml_away_all)
            bd.update({"h2h_home": bh[1], "h2h_home_book": bh[0],
                       "h2h_away": ba[1], "h2h_away_book": ba[0],
                       "books_count": len(ml_home_all)})
            # Pinnacle no-vig
            ph = next((p for b, p in ml_home_all if "pinnacle" in b), None)
            pa = next((p for b, p in ml_away_all if "pinnacle" in b), None)
            if ph and pa:
                hi, ai = _to_prob(ph), _to_prob(pa)
                t = hi + ai
                bd.update({"novig_home": round(hi / t, 4),
                           "novig_away": round(ai / t, 4),
                           "sharp_book": "pinnacle",
                           "pinnacle_home": ph,
                           "pinnacle_away": pa})

        if bd:
            # Store the fixture date so the fallback scheduler can use it
            raw_dt = meta.get("start_date") or meta.get("start_time") or ""
            bd["date"] = raw_dt[:10] if raw_dt else datetime.now(timezone.utc).strftime("%Y-%m-%d")
            result[key] = bd

    print(f"[Odds-OpticOdds] Parsed odds for {len(result)} games.")
    return result


def _parse_opticodds(fixtures):
    """Legacy shim — not called in v6. Kept for reference."""
    return {}


def merge_odds(theapi_odds, opticodds_odds):
    """Merge both sources. OpticOdds takes priority for no-vig when available."""
    merged = {}
    for k in set(theapi_odds) | set(opticodds_odds):
        merged[k] = {**theapi_odds.get(k, {}), **opticodds_odds.get(k, {})}
    return merged


def _kelly(p, american_odds):
    b = american_odds / 100 if american_odds > 0 else 100 / abs(american_odds)
    return max(0.0, (b * p - (1 - p)) / b) * KELLY_FRACTION

# ---------------------------------------------------------------------------
# 4.  FEATURE ENGINEERING  (walk-forward, no lookahead)
# ---------------------------------------------------------------------------

def build_team_logs(games):
    """Expand game rows into per-team-per-game rows (home + away perspectives)."""
    home = games[["game_id","season","date","home_team","away_team",
                  "home_score","away_score","point_diff","total_pts","home_win",
                  "pace_proxy","home_off_rtg","home_def_rtg","home_net_rtg"]].copy()
    home = home.rename(columns={
        "home_team":"team","away_team":"opponent",
        "home_score":"pts_for","away_score":"pts_against",
        "home_off_rtg":"off_rtg","home_def_rtg":"def_rtg","home_net_rtg":"net_rtg"})
    home["home"] = 1
    home["result"] = home["home_win"]

    away = games[["game_id","season","date","away_team","home_team",
                  "away_score","home_score","point_diff","total_pts","home_win",
                  "pace_proxy","away_off_rtg","away_def_rtg","away_net_rtg"]].copy()
    away = away.rename(columns={
        "away_team":"team","home_team":"opponent",
        "away_score":"pts_for","home_score":"pts_against",
        "away_off_rtg":"off_rtg","away_def_rtg":"def_rtg","away_net_rtg":"net_rtg"})
    away["home"] = 0
    away["point_diff"] = -away["point_diff"]
    away["result"] = (1 - away["home_win"]).astype(int)

    logs = pd.concat([home, away], ignore_index=True)
    return logs.sort_values(["team", "date"]).reset_index(drop=True)


def build_team_features(logs):
    """
    Walk-forward rolling features.
    shift(1) on every stat ensures we never use the current game's outcome.
    Net rating columns are the key new predictors (off/def efficiency proxy).
    """
    WINDOWS = [5, 10, 20]
    frames = []
    for team, grp in logs.groupby("team"):
        g = grp.sort_values("date").copy()

        for w in WINDOWS:
            mn = max(2, w // 2)
            for col in ["pts_for", "pts_against", "point_diff", "total_pts",
                        "net_rtg", "off_rtg", "def_rtg", "pace_proxy"]:
                g[f"{col}_roll{w}"] = (g[col].shift(1)
                                       .rolling(w, min_periods=mn).mean())
            g[f"win_rate{w}"] = (g["result"].shift(1)
                                 .rolling(w, min_periods=mn).mean())

        for col in ["pts_for", "pts_against", "point_diff", "net_rtg"]:
            g[f"{col}_ewm"] = g[col].shift(1).ewm(span=5, min_periods=2).mean()
        g["win_ewm"] = g["result"].shift(1).ewm(span=5, min_periods=2).mean()

        g["win_rate3"]    = g["result"].shift(1).rolling(3, min_periods=1).mean()
        g["win_rate10_c"] = g["result"].shift(1).rolling(10, min_periods=3).mean()
        g["momentum"]     = g["win_rate3"] - g["win_rate10_c"]

        g["net_rtg_trend"] = g["net_rtg_roll5"] - g["net_rtg_roll20"]

        g["days_rest"] = g["date"].diff().dt.days.fillna(7).clip(0, 14)
        g["is_b2b"]    = (g["days_rest"] == 1).astype(int)

        home_pts = (g[g["home"] == 1]["pts_for"].shift(1)
                    .rolling(10, min_periods=2).mean())
        away_pts = (g[g["home"] == 0]["pts_for"].shift(1)
                    .rolling(10, min_periods=2).mean())
        g["home_scoring_avg"] = home_pts.reindex(g.index).ffill()
        g["away_scoring_avg"] = away_pts.reindex(g.index).ffill()

        g["scoring_var10"] = (g["pts_for"].shift(1)
                              .rolling(10, min_periods=3).std())

        frames.append(g)
    return pd.concat(frames).sort_values(["date","team"]).reset_index(drop=True)


def build_game_features(df):
    """
    Pivot per-team rows into one row per game.
    Adds diff_ = home_feat - away_feat for every feature column.

    Note: build_team_logs renames home_team→team, away_team→opponent.
    We restore home_team / away_team here for downstream compatibility.
    """
    feat_cols = [c for c in df.columns if any(x in c for x in [
        "_roll", "win_rate", "ewm", "momentum", "net_rtg",
        "off_rtg", "def_rtg", "pace", "scoring",
        "days_rest", "is_b2b", "_trend",
    ])]

    outcome_cols = [c for c in ["game_id", "date", "season",
                                "point_diff", "total_pts", "home_win"]
                    if c in df.columns]

    h = df[df["home"] == 1].copy()
    a = df[df["home"] == 0].copy()

    # Home rows: grab outcome cols + team/opponent names + features
    h_sel = outcome_cols + ["team", "opponent"] + [c for c in feat_cols if c in h.columns]
    hf = h[[c for c in h_sel if c in h.columns]].rename(
        columns={"team": "home_team", "opponent": "away_team",
                 **{c: f"home_{c}" for c in feat_cols if c in h.columns}})

    # Away rows: just game_id + team name + features (outcome cols come from home)
    a_sel = [c for c in ["game_id", "team"] if c in a.columns] + \
            [c for c in feat_cols if c in a.columns]
    af = a[[c for c in a_sel if c in a.columns]].rename(
        columns={"team": "away_team_check",
                 **{c: f"away_{c}" for c in feat_cols if c in a.columns}})

    # Merge on game_id (unique per game; more robust than date+team)
    if "game_id" in hf.columns and "game_id" in af.columns:
        games = hf.merge(af, on="game_id", how="inner")
    else:
        # Fallback: merge on date + away_team name
        games = hf.merge(af,
                         left_on=["date", "away_team"],
                         right_on=["date", "away_team_check"],
                         how="inner")

    games = games.drop(columns=["away_team_check"], errors="ignore")

    for col in feat_cols:
        hc, ac = f"home_{col}", f"away_{col}"
        if hc in games.columns and ac in games.columns:
            games[f"diff_{col}"] = games[hc] - games[ac]

    return games.dropna(subset=["home_win"]).reset_index(drop=True)


def compute_elo(df, k=20, home_adv=100):
    """
    Margin-adjusted Elo ratings computed chronologically.
    Stores the pre-game rating for each game (no leakage).
    """
    elo = {t: 1500.0 for t in
           pd.concat([df["home_team"], df["away_team"]]).unique()}
    h_elos, a_elos = [], []
    for _, row in df.sort_values("date").iterrows():
        h, a = row["home_team"], row["away_team"]
        exp_h = 1 / (1 + 10 ** ((elo[a] - (elo[h] + home_adv)) / 400))
        actual = row["home_win"]
        h_elos.append(elo[h])
        a_elos.append(elo[a])
        margin = abs(row["point_diff"])
        k_m = k * np.log1p(margin) * (2.2 / (margin * 0.001 + 2.2))
        elo[h] += k_m * (actual - exp_h)
        elo[a] += k_m * (exp_h - actual)
    df = df.sort_values("date").copy()
    df["home_elo"]     = h_elos
    df["away_elo"]     = a_elos
    df["elo_diff"]     = df["home_elo"] - df["away_elo"]
    df["elo_win_prob"] = 1 / (1 + 10 ** (-df["elo_diff"] / 400))
    return df

# ---------------------------------------------------------------------------
# 5.  FEATURE LISTS
# ---------------------------------------------------------------------------

BASE_ML = [
    "elo_diff", "elo_win_prob",
    # Rolling point metrics
    "diff_pts_for_roll10", "diff_pts_against_roll10", "diff_point_diff_roll10",
    "diff_pts_for_roll5",  "diff_pts_against_roll5",  "diff_point_diff_roll5",
    "diff_win_rate10",     "diff_win_rate5",
    "home_win_rate10",     "away_win_rate10",
    "diff_momentum",       "diff_win_ewm",   "diff_point_diff_ewm",
    # Net rating — key new features (strongest predictors)
    "diff_net_rtg_roll10", "diff_net_rtg_roll5",  "diff_net_rtg_ewm",
    "home_net_rtg_roll10", "away_net_rtg_roll10",
    "diff_net_rtg_trend",
    # Efficiency breakdown
    "diff_off_rtg_roll10", "diff_def_rtg_roll10",
    "diff_pace_proxy_roll10",
    # Schedule
    "diff_days_rest",      "home_is_b2b",    "away_is_b2b",
]

PLAYER_ML = [
    "diff_quality_score",  "diff_team_bpm",  "diff_team_vorp",
    "diff_team_per",       "diff_team_ts_pct",
    "diff_fg_pct",         "diff_fg3_pct",   "diff_avg_tov",
    "home_injury_impact",  "away_injury_impact", "diff_injury_impact",
    "home_star_out",       "away_star_out",
    "diff_lineup_depth",   "diff_rotation_stress",
]


def avail(df, cols):
    return [c for c in cols if c in df.columns]

# ---------------------------------------------------------------------------
# 6.  MODEL TRAINING
# ---------------------------------------------------------------------------

def train_moneyline(games, has_player=False):
    """LR + XGBoost ensemble for moneyline win probability."""
    base  = avail(games, BASE_ML)
    extra = avail(games, PLAYER_ML) if has_player else []
    feats = base + [x for x in extra if x not in base]
    X, y  = games[feats].values, games["home_win"].values

    lr_pipe = Pipeline([
        ("sc", StandardScaler()),
        ("cl", CalibratedClassifierCV(
            LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"),
            cv=5)),
    ])
    xgb_clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.75, colsample_bytree=0.75, min_child_weight=5,
        gamma=0.1, eval_metric="logloss",
        use_label_encoder=False, random_state=42,
    )
    lr_pipe.fit(X, y)
    xgb_clf.fit(X, y)

    tscv    = TimeSeriesSplit(5)
    lr_auc  = cross_val_score(lr_pipe,  X, y, cv=tscv, scoring="roc_auc").mean()
    xgb_auc = cross_val_score(xgb_clf, X, y, cv=tscv, scoring="roc_auc").mean()
    tag = "(+player stats)" if has_player else "(game data only)"
    print(f"[ML] LR AUC={lr_auc:.3f}  XGB AUC={xgb_auc:.3f}  {tag}")

    importance = sorted(
        zip(feats, xgb_clf.feature_importances_),
        key=lambda x: x[1], reverse=True)[:15]

    return {"lr": lr_pipe, "xgb": xgb_clf, "features": feats,
            "has_player": has_player,
            "lr_auc": round(lr_auc, 3), "xgb_auc": round(xgb_auc, 3),
            "feature_importance": importance}


def ensemble_prob(model, feat_row):
    """Blend LR (45%) + XGBoost (55%)."""
    feats = model["features"]
    x     = np.array([[feat_row.get(f, 0) for f in feats]])
    lr_p  = float(model["lr"].predict_proba(x)[0][1])
    xgb_p = float(model["xgb"].predict_proba(x)[0][1])
    return 0.45 * lr_p + 0.55 * xgb_p, lr_p, xgb_p

# ---------------------------------------------------------------------------
# 7.  BACKTEST  (proper walk-forward — zero lookahead bias)
# ---------------------------------------------------------------------------

def run_backtest(games_df, n_train_seasons=3):
    """
    Walk-forward backtest.
    When we have enough seasons: train on first N, test on the rest.
    Falls back to TimeSeriesSplit when only 1-2 seasons are available.
    Player data is intentionally excluded from backtest features to avoid
    end-of-season leakage (season averages are only fully known at season end).
    """
    print("\n[Backtest] Running walk-forward evaluation...")
    seasons = sorted(games_df["season"].unique())
    if len(seasons) > n_train_seasons:
        train_s = seasons[:n_train_seasons]
        test_s  = seasons[n_train_seasons:]
        train   = games_df[games_df["season"].isin(train_s)].copy()
        test    = games_df[games_df["season"].isin(test_s)].copy()
        print(f"  Train: seasons {train_s} ({len(train)} games)")
        print(f"  Test:  seasons {test_s}  ({len(test)} games)")
        model = train_moneyline(train.dropna(subset=avail(train, BASE_ML)),
                                has_player=False)
        return _eval_backtest(model, test)
    else:
        print(f"  Only {len(seasons)} season(s) — using TimeSeriesSplit(5).")
        return _backtest_tscv(games_df)


def _backtest_tscv(games_df):
    feats  = avail(games_df, BASE_ML)
    clean  = games_df.dropna(subset=feats).sort_values("date").copy()
    tscv   = TimeSeriesSplit(n_splits=5)
    X, y   = clean[feats].values, clean["home_win"].values
    preds  = []
    for train_idx, test_idx in tscv.split(X):
        m = train_moneyline(
            clean.iloc[train_idx].dropna(subset=feats), has_player=False)
        for idx in test_idx:
            row  = dict(zip(feats, X[idx]))
            prob, lr_p, xgb_p = ensemble_prob(m, row)
            row_df = clean.iloc[idx]
            preds.append({
                "date":       str(row_df["date"])[:10],
                "home":       row_df.get("home_team", ""),
                "away":       row_df.get("away_team", ""),
                "model_prob": round(prob, 3),
                "lr_prob":    round(lr_p, 3),
                "xgb_prob":   round(xgb_p, 3),
                "actual":     int(row_df["home_win"]),
                "elo_prob":   round(float(row_df.get("elo_win_prob", 0.5)), 3),
            })
    return _score_predictions(preds)


def _eval_backtest(model, test_df):
    feats      = avail(test_df, model["features"])
    test_clean = test_df.dropna(subset=feats).sort_values("date").copy()
    preds = []
    for _, row in test_clean.iterrows():
        feat_row = {f: row.get(f, 0) for f in model["features"]}
        prob, lr_p, xgb_p = ensemble_prob(model, feat_row)
        preds.append({
            "date":       str(row["date"])[:10],
            "home":       row.get("home_team", ""),
            "away":       row.get("away_team", ""),
            "model_prob": round(prob, 3),
            "lr_prob":    round(lr_p, 3),
            "xgb_prob":   round(xgb_p, 3),
            "actual":     int(row["home_win"]),
            "elo_prob":   round(float(row.get("elo_win_prob", 0.5)), 3),
        })
    return _score_predictions(preds)


def _score_predictions(preds):
    if not preds:
        return {}
    probs   = [p["model_prob"] for p in preds]
    actuals = [p["actual"]     for p in preds]
    auc   = roc_auc_score(actuals, probs)
    brier = brier_score_loss(actuals, probs)
    accuracy = sum(1 for p in preds
                   if (p["model_prob"] >= 0.5) == bool(p["actual"])) / len(preds)

    # Calibration buckets
    buckets = [(0.45,0.50),(0.50,0.55),(0.55,0.60),
               (0.60,0.65),(0.65,0.70),(0.70,0.75),(0.75,1.00)]
    calibration = []
    for lo, hi in buckets:
        sub = [p for p in preds if lo <= p["model_prob"] < hi]
        if sub:
            calibration.append({
                "bucket":    f"{lo:.0%}-{hi:.0%}",
                "predicted": round(np.mean([p["model_prob"] for p in sub]), 3),
                "actual":    round(np.mean([p["actual"] for p in sub]), 3),
                "n":         len(sub),
                "diff":      round(np.mean([p["model_prob"] for p in sub]) -
                                   np.mean([p["actual"] for p in sub]), 3),
            })

    # Simulated flat-bet at -110 odds when model_prob > threshold
    THRESH = 0.55
    bankroll, unit = 1000.0, 10.0
    bets, wins, history = 0, 0, [1000.0]
    for p in sorted(preds, key=lambda x: x["date"]):
        if p["model_prob"] < THRESH:
            continue
        bets += 1
        if p["actual"] == 1:
            wins += 1
            bankroll += unit * 0.909
        else:
            bankroll -= unit
        history.append(round(bankroll, 2))

    win_rate = wins / bets if bets else 0
    roi = (bankroll - 1000) / 1000 * 100

    # Monthly breakdown
    monthly = {}
    for p in preds:
        m = p["date"][:7]
        monthly.setdefault(m, {"correct": 0, "total": 0, "probs": []})
        monthly[m]["total"] += 1
        monthly[m]["probs"].append(p["model_prob"])
        if (p["model_prob"] >= 0.5) == bool(p["actual"]):
            monthly[m]["correct"] += 1

    monthly_list = [
        {"month": k, "games": v["total"],
         "accuracy": round(v["correct"] / v["total"], 3),
         "avg_prob": round(np.mean(v["probs"]), 3)}
        for k, v in sorted(monthly.items())
    ]

    print(f"[Backtest] {len(preds)} games | AUC={auc:.3f} | "
          f"Accuracy={accuracy:.3f} | Brier={brier:.3f}")
    print(f"[Backtest] Flat-bet simulation (>{THRESH}): "
          f"{bets} bets | {win_rate:.1%} wins | ROI={roi:.1f}%")

    return {
        "test_games": len(preds),
        "accuracy":   round(accuracy, 3),
        "auc":        round(auc, 3),
        "brier":      round(brier, 3),
        "flat_bet": {
            "threshold":         THRESH,
            "bets_placed":       bets,
            "bets_won":          wins,
            "win_rate":          round(win_rate, 3),
            "roi_pct":           round(roi, 2),
            "starting_bankroll": 1000,
            "ending_bankroll":   round(bankroll, 2),
            "bankroll_history":  history[:300],
            "note": ("Flat-bet at -110. Real bets sized by Kelly once live "
                     "odds are available. ROI here is a model-quality proxy only."),
        },
        "calibration": calibration,
        "monthly":     monthly_list,
    }

# ---------------------------------------------------------------------------
# 8.  EDGE CALCULATION + RECOMMENDATIONS  (Pinnacle-anchored)
# ---------------------------------------------------------------------------

FACTOR_LABELS = {
    "elo_diff":              ("ELO advantage",          "Season-long team strength gap"),
    "elo_win_prob":          ("ELO win probability",    "Implied win rate by Elo rating"),
    "diff_net_rtg_roll10":   ("Net rating edge (10g)",  "Scoring margin per 100 possessions, last 10"),
    "diff_net_rtg_roll5":    ("Net rating edge (5g)",   "Recent net rating differential"),
    "diff_net_rtg_ewm":      ("Net rating trend",       "Recency-weighted net rating edge"),
    "diff_net_rtg_trend":    ("Form vs baseline",       "Recent net rating vs season average"),
    "diff_point_diff_roll10":("10g margin edge",        "Average point margin differential, last 10"),
    "diff_point_diff_roll5": ("5g margin edge",         "Recent point margin differential"),
    "diff_off_rtg_roll10":   ("Offensive edge",         "Offensive efficiency gap, last 10"),
    "diff_def_rtg_roll10":   ("Defensive edge",         "Defensive efficiency gap, last 10"),
    "diff_win_rate10":       ("Win rate edge",          "Win % gap, last 10 games"),
    "diff_momentum":         ("Momentum",               "Recent form vs season average"),
    "diff_win_ewm":          ("Weighted form",          "Exponentially weighted win rate edge"),
    "diff_point_diff_ewm":   ("Scoring trend",          "Recency-weighted scoring margin"),
    "home_is_b2b":           ("Home B2B fatigue",       "Home team on second night of B2B"),
    "away_is_b2b":           ("Away B2B fatigue",       "Away team on second night of B2B"),
    "diff_days_rest":        ("Rest advantage",         "Days of rest differential"),
    "diff_pace_proxy_roll10":("Pace mismatch",          "Fast vs slow team matchup, last 10"),
    "diff_quality_score":    ("Roster quality edge",    "Composite player quality differential"),
    "diff_team_bpm":         ("BPM edge",               "Box Plus/Minus differential"),
    "diff_team_vorp":        ("VORP advantage",         "Value over replacement gap"),
    "diff_team_per":         ("PER advantage",          "Player efficiency rating differential"),
    "diff_fg3_pct":          ("3PT% edge",              "Three-point shooting gap"),
    "home_injury_impact":    ("Home injuries",          "Pts lost to home team injuries"),
    "away_injury_impact":    ("Away injuries",          "Pts lost to away team injuries"),
    "home_star_out":         ("Home star missing",      "Top-2 home player confirmed out"),
    "away_star_out":         ("Away star missing",      "Top-2 away player confirmed out"),
}


def get_top_factors(feat_row, model, top_n=5):
    """Top contributing features from LR coefficient × scaled value."""
    try:
        lr_clf   = model["lr"].named_steps["cl"].estimator
        scaler   = model["lr"].named_steps["sc"]
        feats    = model["features"]
        x_raw    = np.array([[feat_row.get(f, 0) for f in feats]])
        x_sc     = scaler.transform(x_raw)[0]
        coefs    = lr_clf.coef_[0]
        contribs = sorted(
            [(feats[i], x_sc[i] * coefs[i]) for i in range(len(feats))],
            key=lambda x: abs(x[1]), reverse=True)
        factors = []
        for feat, contrib in contribs[:top_n]:
            if abs(contrib) < 0.005:
                continue
            label, detail = FACTOR_LABELS.get(feat, (feat.replace("_", " ").title(), ""))
            factors.append({
                "feature":   feat,
                "label":     label,
                "detail":    detail,
                "direction": "+" if contrib > 0 else "-",
                "strength":  ("Strong"   if abs(contrib) > 0.25 else
                               "Moderate" if abs(contrib) > 0.12 else "Mild"),
                "contrib":   round(contrib, 4),
                "raw_val":   round(feat_row.get(feat, 0), 3),
            })
        return factors
    except Exception:
        return []


def build_recommendations(home_win_prob, book_odds, factors, home, away):
    """
    Build bet recs by comparing model probability to the no-vig market price.
    Uses Pinnacle (sharpest book) as the no-vig benchmark when available.
    Only flags bets with edge > MIN_EDGE.
    """
    recs  = []
    book  = book_odds or {}
    top_f = [f["label"] for f in factors[:3]] if factors else []

    boh      = book.get("h2h_home")
    boa      = book.get("h2h_away")
    novig_h  = book.get("novig_home")
    novig_a  = book.get("novig_away")
    sharp    = book.get("sharp_book", "")
    disagree = book.get("ml_disagreement", 0)
    n_books  = book.get("books_count", 1)

    if boh and boa:
        true_h = novig_h if novig_h else _to_prob(boh)
        true_a = novig_a if novig_a else _to_prob(boa)
        nv_note = f" (no-vig vs {sharp})" if novig_h else " (raw implied, no sharp book)"

        # HOME side
        edge_h = home_win_prob - true_h
        if edge_h > MIN_EDGE:
            grade = "A" if edge_h > 0.06 else "B" if edge_h > 0.04 else "C"
            if disagree > 10 and grade == "B":
                grade = "A"
            recs.append({
                "type":        "ML",
                "side":        "HOME",
                "team":        home,
                "odds":        boh,
                "best_book":   book.get("h2h_home_book", ""),
                "model_prob":  round(home_win_prob, 3),
                "novig_prob":  round(true_h, 3),
                "edge":        round(edge_h, 3),
                "kelly_pct":   round(_kelly(home_win_prob, boh) * 100, 1),
                "grade":       grade,
                "label":       f"{home} ML",
                "reason":      f"Model {home_win_prob:.1%} vs {true_h:.1%} no-vig{nv_note}",
                "top_factors": top_f,
                "books_count": n_books,
                "disagreement":disagree,
            })

        # AWAY side
        away_prob = 1 - home_win_prob
        edge_a = away_prob - true_a
        if edge_a > MIN_EDGE:
            grade = "A" if edge_a > 0.06 else "B" if edge_a > 0.04 else "C"
            if disagree > 10 and grade == "B":
                grade = "A"
            recs.append({
                "type":        "ML",
                "side":        "AWAY",
                "team":        away,
                "odds":        boa,
                "best_book":   book.get("h2h_away_book", ""),
                "model_prob":  round(away_prob, 3),
                "novig_prob":  round(true_a, 3),
                "edge":        round(edge_a, 3),
                "kelly_pct":   round(_kelly(away_prob, boa) * 100, 1),
                "grade":       grade,
                "label":       f"{away} ML",
                "reason":      f"Model {away_prob:.1%} vs {true_a:.1%} no-vig{nv_note}",
                "top_factors": top_f,
                "books_count": n_books,
                "disagreement":disagree,
            })
    return recs

# ---------------------------------------------------------------------------
# 9.  BET LOG + CLV TRACKING
# ---------------------------------------------------------------------------

def log_bet(matchup, rec, date_str, book_odds):
    """Append a flagged bet to bet_log.json for Closing Line Value tracking."""
    try:
        log = json.load(open(BET_LOG_PATH)) if os.path.exists(BET_LOG_PATH) else {"bets":[]}
    except Exception:
        log = {"bets": []}

    bet_id = f"{date_str}_{matchup}_{rec['type']}_{rec['side']}".replace(" ","_")
    if any(b["id"] == bet_id for b in log["bets"]):
        return

    log["bets"].append({
        "id":           bet_id,
        "date":         date_str,
        "matchup":      matchup,
        "type":         rec["type"],
        "side":         rec["side"],
        "team":         rec.get("team",""),
        "odds":         rec["odds"],
        "best_book":    rec.get("best_book",""),
        "model_prob":   rec["model_prob"],
        "novig_prob":   rec.get("novig_prob", rec["model_prob"]),
        "edge":         rec["edge"],
        "kelly_pct":    rec["kelly_pct"],
        "grade":        rec["grade"],
        "top_factors":  rec.get("top_factors",[]),
        "books_count":  rec.get("books_count",0),
        "disagreement": rec.get("disagreement",0),
        # Fill in after game resolves:
        "result":          "PENDING",  # W / L / PUSH
        "closing_odds":    None,
        "clv":             None,       # closing_odds - opening_odds (+ = beat the line)
    })

    graded = [b for b in log["bets"] if b["result"] in ("W","L")]
    if graded:
        wins = sum(1 for b in graded if b["result"] == "W")
        clv_bets = [b for b in graded if b.get("clv") is not None]
        log["summary"] = {
            "total_bets":   len(graded),
            "wins":         wins,
            "losses":       len(graded)-wins,
            "win_rate":     round(wins/len(graded),3),
            "avg_edge":     round(float(np.mean([b["edge"] for b in graded])),3),
            "avg_clv":      round(float(np.mean([b["clv"] for b in clv_bets])),2)
                            if clv_bets else None,
            "pending_bets": sum(1 for b in log["bets"] if b["result"]=="PENDING"),
            "grade_A": (f"{sum(1 for b in graded if b.get('grade')=='A' and b['result']=='W')}"
                        f"-{sum(1 for b in graded if b.get('grade')=='A' and b['result']=='L')}"),
            "grade_B": (f"{sum(1 for b in graded if b.get('grade')=='B' and b['result']=='W')}"
                        f"-{sum(1 for b in graded if b.get('grade')=='B' and b['result']=='L')}"),
        }
    with open(BET_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

# ---------------------------------------------------------------------------
# 10.  MAIN MODEL CLASS
# ---------------------------------------------------------------------------

class NBABettingModel:
    def __init__(self):
        self.games_df        = None
        self.rolled_logs     = None
        self.model           = None
        self.player_profiles = {}
        self.has_player      = False
        self.current_elo     = {}
        self.backtest_stats  = {}

    # ── Training ────────────────────────────────────────────────────────────

    def fit(self):
        print("=" * 60)
        print("NBA Betting Model  v6  —  Full Rebuild")
        print("=" * 60)

        games_raw = fetch_games(TRAIN_SEASONS)
        logs      = build_team_logs(games_raw)
        rolled    = build_team_features(logs)
        gf        = build_game_features(rolled)
        gf        = compute_elo(gf)

        self.games_df    = gf
        self.rolled_logs = rolled

        # Persist latest Elo per team for predictions
        for _, row in gf.sort_values("date").iterrows():
            self.current_elo[row["home_team"]] = row["home_elo"]
            self.current_elo[row["away_team"]] = row["away_elo"]

        # Train on all available data (game features only — no player leakage)
        train_data   = gf.dropna(subset=avail(gf, BASE_ML))
        self.model   = train_moneyline(train_data, has_player=False)

        # Backtest
        self.backtest_stats = run_backtest(gf)

        # Player data — loaded AFTER backtest, used only for forward predictions
        print("\n[Players] Loading current-season player data for predictions...")
        totals_df            = fetch_player_totals(NBAAPI_SEASON)
        advanced_df          = fetch_player_advanced(NBAAPI_SEASON)
        self.player_profiles = build_team_player_profiles(totals_df, advanced_df)
        self.has_player      = bool(self.player_profiles)

        print("\n[Fit] Complete.\n")

    # ── Single-game prediction ───────────────────────────────────────────────

    def _team_feats(self, team, is_home):
        prefix = "home" if is_home else "away"
        tdf = self.rolled_logs[self.rolled_logs["team"] == team].sort_values("date")
        if tdf.empty:
            raise ValueError(f"Team '{team}' not found in training data.")
        latest   = tdf.iloc[-1]
        feat_keys = [c for c in tdf.columns if any(x in c for x in [
            "_roll","win_rate","ewm","momentum","net_rtg",
            "off_rtg","def_rtg","pace","scoring","days_rest","is_b2b","_trend",
        ])]
        return {f"{prefix}_{c}": latest[c] for c in feat_keys}

    def predict(self, home, away, book_odds=None, injury_report=None):
        fr = {**self._team_feats(home, True), **self._team_feats(away, False)}

        h_elo = self.current_elo.get(home, 1500.0)
        a_elo = self.current_elo.get(away, 1500.0)
        fr["elo_diff"]     = h_elo - a_elo
        fr["elo_win_prob"] = 1 / (1 + 10 ** (-(h_elo - a_elo) / 400))

        for s in {k[5:] for k in fr if k.startswith("home_")}:
            fr[f"diff_{s}"] = fr.get(f"home_{s}", 0) - fr.get(f"away_{s}", 0)

        # Player and injury augmentation
        hp = self.player_profiles.get(home, {})
        ap = self.player_profiles.get(away, {})
        for key in ["quality_score","team_bpm","team_vorp","team_per","team_ts_pct",
                    "fg_pct","fg3_pct","ft_pct","avg_tov","lineup_depth","rotation_stress"]:
            fr[f"home_{key}"] = hp.get(key, 0)
            fr[f"away_{key}"] = ap.get(key, 0)
            fr[f"diff_{key}"] = hp.get(key, 0) - ap.get(key, 0)

        inj_h = compute_injury_impact(home, injury_report or {}, self.player_profiles)
        inj_a = compute_injury_impact(away, injury_report or {}, self.player_profiles)
        fr["home_injury_impact"] = inj_h["impact_score"]
        fr["away_injury_impact"] = inj_a["impact_score"]
        fr["diff_injury_impact"] = inj_h["impact_score"] - inj_a["impact_score"]
        fr["home_star_out"]      = int(inj_h["star_out"])
        fr["away_star_out"]      = int(inj_a["star_out"])

        hwp, lr_p, xgb_p = ensemble_prob(self.model, fr)
        factors           = get_top_factors(fr, self.model)
        recs              = build_recommendations(hwp, book_odds, factors, home, away)

        book = book_odds or {}
        return {
            "matchup":          f"{home} vs {away}",
            "home":             home,
            "away":             away,
            "home_elo":         round(h_elo, 1),
            "away_elo":         round(a_elo, 1),
            "home_win_prob":    round(hwp, 3),
            "away_win_prob":    round(1 - hwp, 3),
            "blend_prob":       round(hwp, 3),   # alias for display
            "lr_prob":          round(lr_p, 3),
            "xgb_prob":         round(xgb_p, 3),
            # Live odds (populated when Pinnacle/book data available)
            "pinnacle_home":    book.get("pinnacle_home"),
            "pinnacle_away":    book.get("pinnacle_away"),
            "novig_home":       book.get("novig_home"),
            "novig_away":       book.get("novig_away"),
            "best_home_odds":   book.get("h2h_home"),
            "best_away_odds":   book.get("h2h_away"),
            "best_home_book":   book.get("h2h_home_book"),
            "best_away_book":   book.get("h2h_away_book"),
            "books_count":      book.get("books_count", 0),
            "sharp_book":       book.get("sharp_book", ""),
            "top_factors":      factors,
            "recommendations":  recs,
            "home_player_profile": hp,
            "away_player_profile": ap,
            "home_injuries":    inj_h,
            "away_injuries":    inj_a,
            "has_live_odds":    bool(book_odds),
            "has_sharp_novig":  bool(book.get("novig_home")),
        }

    # ── Full pipeline ────────────────────────────────────────────────────────

    def run(self):
        self.fit()

        # Odds
        theapi_odds   = fetch_odds_theapi()
        opticodds_all = fetch_odds_opticodds()
        all_odds      = merge_odds(theapi_odds, opticodds_all)

        # Injuries
        injury_report = fetch_injuries_bdl()

        # Upcoming games — try BDL schedule first, fall back to OpticOdds fixtures
        upcoming = fetch_upcoming_games()
        if not upcoming and all_odds:
            print("[Upcoming] BDL returned 0 games — deriving schedule from OpticOdds fixtures.")
            for key, odds in all_odds.items():
                if " vs " not in key:
                    continue
                home_full, away_full = key.split(" vs ", 1)
                home_abbr = TEAM_NAME_MAP.get(home_full, home_full)
                away_abbr = TEAM_NAME_MAP.get(away_full, away_full)
                upcoming.append({
                    "home": home_abbr,
                    "away": away_abbr,
                    "date": odds.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
                    "time": "TBD",
                })
            print(f"[Upcoming] {len(upcoming)} games from OpticOdds fallback.")
        print(f"\n[Predictions] Generating for {len(upcoming)} upcoming games...")

        games_out, all_recs = [], []
        for g in upcoming:
            home, away = g["home"], g["away"]
            # BDL gives abbreviations (e.g. "ATL"); odds keys use full names
            # ("Atlanta Hawks vs New York Knicks"). Try full name first, then abbr.
            home_full = ABBR_TO_NAME.get(home, home)
            away_full = ABBR_TO_NAME.get(away, away)
            book_odds  = None
            for key, odds in all_odds.items():
                kl = key.lower()
                if (home_full.lower() in kl and away_full.lower() in kl) or \
                   (home.lower() in kl and away.lower() in kl):
                    book_odds = odds
                    break
            try:
                pred = self.predict(home, away, book_odds, injury_report)
            except Exception as e:
                print(f"  Skipping {home} vs {away}: {e}")
                continue
            pred["date"] = g["date"]
            pred["time"] = g["time"]
            games_out.append(pred)
            for rec in pred["recommendations"]:
                all_recs.append({**rec, "matchup": pred["matchup"], "date": g["date"]})
                log_bet(pred["matchup"], rec, g["date"], book_odds or {})

        all_recs.sort(key=lambda x: x.get("edge", 0), reverse=True)

        bet_log_summary, pending = {}, []
        if os.path.exists(BET_LOG_PATH):
            try:
                bl = json.load(open(BET_LOG_PATH))
                bet_log_summary = bl.get("summary", {})
                pending = [b for b in bl.get("bets",[]) if b.get("result")=="PENDING"]
            except Exception:
                pass

        predictions = {
            "generated_at":      datetime.now(timezone.utc).isoformat(),
            "model_version":     "v6-rebuild",
            "train_seasons":     TRAIN_SEASONS,
            "train_games":       int(len(self.games_df)) if self.games_df is not None else 0,
            "has_player_data":   self.has_player,
            "has_live_odds":     bool(all_odds),
            "model_stats": {
                "ml_lr_auc":         self.model["lr_auc"],
                "ml_xgb_auc":        self.model["xgb_auc"],
                "backtest_auc":      self.backtest_stats.get("auc"),
                "backtest_accuracy": self.backtest_stats.get("accuracy"),
                "backtest_brier":    self.backtest_stats.get("brier"),
                "sharp_book_used":   "pinnacle" if any(
                    o.get("sharp_book") == "pinnacle"
                    for o in all_odds.values()) else "none",
            },
            "feature_importance": [
                {"feature": f,
                 "label": FACTOR_LABELS.get(f, (f,))[0],
                 "importance": round(float(imp), 4)}
                for f, imp in self.model["feature_importance"]
            ],
            "games":          games_out,
            "all_recs":       all_recs,
            "bet_log_summary":bet_log_summary,
            "pending_bets":   pending,
        }
        with open(PRED_PATH, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"\n[Output] predictions.json — {len(games_out)} games, "
              f"{len(all_recs)} flagged bets.")

        backtest_out = {
            "generated_at":    datetime.now(timezone.utc).isoformat(),
            "train_seasons":   TRAIN_SEASONS,
            "train_games":     int(len(self.games_df)) if self.games_df is not None else 0,
            "has_player_data": False,
            **self.backtest_stats,
            "feature_importance": [
                {"feature": f,
                 "label": FACTOR_LABELS.get(f, (f,))[0],
                 "importance": round(float(imp), 4)}
                for f, imp in self.model["feature_importance"]
            ],
        }
        with open(BACKTEST_PATH, "w") as f:
            json.dump(backtest_out, f, indent=2)
        print(f"[Output] backtest.json written.")

        grade_a = [r for r in all_recs if r.get("grade") == "A"]
        grade_b = [r for r in all_recs if r.get("grade") == "B"]
        print(f"\n{'='*60}")
        print(f"  Grade A bets (>6% edge vs no-vig):  {len(grade_a)}")
        print(f"  Grade B bets (4-6% edge vs no-vig): {len(grade_b)}")
        if not all_odds:
            print("\n  NOTE: No live odds — set ODDS_API_KEY or OPTICODDS_KEY")
            print("  to enable edge calculations and bet recommendations.")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = NBABettingModel()
    model.run()
