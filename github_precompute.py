"""
Standalone precompute script for GitHub Actions.

Runs the SAME precompute logic as app.py's run_precompute(), but with
no Streamlit dependency, so it can run headless on a schedule. Writes
the same sahi_zones_cache.json file app.py reads on startup -- when
this script's output is committed back to the repo, Streamlit Cloud's
auto-redeploy-on-push picks up the fresh cache immediately, so opening
the app later shows current data without ever clicking "Run Precompute".

Reads the Upstox token from the UPSTOX_ANALYTICAL_TOKEN environment
variable (set as a GitHub Actions secret) -- this is the long-lived
(~1 year) analytical token, separate from the daily-refreshed token the
live app's OAuth login flow uses.

NOTE: this deliberately DUPLICATES several functions/constants from
app.py (equity/futures resolvers, fetch_candles, zone computation)
rather than importing them, since app.py has Streamlit-specific code
that can't run headless. If those functions change in app.py, update
this file to match.

Usage:
    UPSTOX_ANALYTICAL_TOKEN=xxx python github_precompute.py
"""
import gzip
import io
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from sahi_style_key_levels import sahi_style_key_levels

IST = timezone(timedelta(hours=5, minutes=30))


def now_ist():
    return datetime.now(IST)


# ---------------------------------------------------------------------------
# Constants (copied from app.py)
# ---------------------------------------------------------------------------
CACHE_PATH = "sahi_zones_cache.json"
INSTRUMENT_SEARCH_URL = "https://api.upstox.com/v2/instruments/search"
INSTRUMENT_MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"

DAILY_LOOKBACK_DAYS = 60
COMPOSITE_LOOKBACK_DAYS = 18
RVOL_BASELINE_DAYS = 20

COMPOSITE_N_BINS = 50
INTRADAY_N_BINS = 45
MIN_PROMINENCE_PCT = 0.08
MIN_BIN_DISTANCE = 2
MAX_ZONES = 6
MIN_DISPLAY_PCT = 2.0

EQUITY_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "AXISBANK",
    "KOTAKBANK", "BAJFINANCE", "BHARTIARTL", "ITC", "LT", "HINDUNILVR",
    "MARUTI", "TMPV", "TATASTEEL", "SUNPHARMA", "TITAN", "ULTRACEMCO",
    "ASIANPAINT", "WIPRO", "NTPC", "POWERGRID", "M&M", "ADANIENT",
    "ADANIPORTS", "BAJAJFINSV", "HCLTECH", "JSWSTEEL", "ONGC", "COALINDIA",
    "TECHM", "GRASIM", "DIVISLAB", "DRREDDY", "CIPLA", "EICHERMOT",
    "HEROMOTOCO", "HINDALCO", "BPCL", "BRITANNIA", "APOLLOHOSP", "SBILIFE",
    "HDFCLIFE", "INDUSINDBK", "BAJAJ-AUTO", "TATACONSUM", "UPL", "SHREECEM",
    "NESTLEIND", "GAIL", "PIDILITIND", "DLF", "GODREJCP",
    "SIEMENS", "AMBUJACEM", "BANDHANBNK", "BANKBARODA", "PNB", "CANBK",
    "IDFCFIRSTB", "FEDERALBNK", "AUROPHARMA", "BEL", "BIOCON", "CHOLAFIN",
    "COLPAL", "CONCOR", "CUMMINSIND", "DABUR", "DEEPAKNTR", "ESCORTS",
    "EXIDEIND", "GODREJPROP", "HAVELLS", "HDFCAMC", "ICICIGI", "ICICIPRULI",
    "IEX", "INDIGO", "INDUSTOWER", "IOC", "IRCTC", "JINDALSTEL", "JUBLFOOD",
    "LICHSGFIN", "LTM", "LUPIN", "MANAPPURAM", "MARICO", "UNITDSPR",
    "MFSL", "MOTHERSON", "MPHASIS", "MRF", "MUTHOOTFIN", "NAUKRI",
    "NMDC", "OBEROIRLTY", "OFSS", "PAGEIND", "PIRAMALFIN", "PERSISTENT",
    "PETRONET", "PFC", "PIIND", "POLYCAB", "RECLTD", "SAIL", "SBICARD",
    "SRF", "SYNGENE", "TATACOMM", "TATAPOWER", "TORNTPHARM", "TRENT",
    "TVSMOTOR", "UBL", "VOLTAS", "ZEEL", "ZYDUSLIFE", "CDSL", "IRFC",
    "IDEA", "YESBANK", "SUZLON", "ETERNAL", "DMART", "JIOFIN", "PAYTM",
    "NYKAA", "POLICYBZR", "DELHIVERY", "LODHA", "PATANJALI", "ABCAPITAL",
    "ALKEM", "APLAPOLLO", "ASHOKLEY", "ASTRAL", "ATUL", "BALKRISIND",
    "BATAINDIA", "BHARATFORG", "BHEL", "BSOFT", "CANFINHOME", "CROMPTON",
    "CUB", "DALBHARAT", "GLENMARK", "GMRAIRPORT", "GNFC", "GRANULES",
    "HAL", "HINDCOPPER", "HINDPETRO", "SAMMAANCAP", "IGL",
    "INDHOTEL", "INDIAMART", "IPCALAB", "JKCEMENT", "LALPATHLAB",
    "LAURUSLABS", "M&MFIN", "METROPOLIS", "NATIONALUM", "NAVINFLUOR",
    "OIL", "PVRINOX", "RAIN", "RBLBANK", "SUNTV", "TATACHEM",
    "TATAELXSI", "TORNTPOWER", "UNIONBANK", "VBL", "WHIRLPOOL",
    "AARTIIND", "ABFRL", "ANGELONE", "APOLLOTYRE", "AUBANK", "BANKINDIA",
    "BSE", "CGPOWER", "CHAMBLFERT", "COFORGE", "COROMANDEL", "DIXON",
    "FORTIS", "GICRE", "GODFRYPHLP", "GRAPHITE", "HFCL",
    "HUDCO", "IIFL", "INDIACEM", "IRB", "ITI", "KALYANKJIL",
    "KEI", "LTF", "MANKIND", "MAXHEALTH", "MGL", "MOTILALOFS",
    "NBCC", "NCC", "NHPC", "PFIZER", "PGEL", "POWERINDIA",
    "PRESTIGE", "RVNL", "SJVN", "SOLARINDS", "SONACOMS", "STARHEALTH",
    "SUPREMEIND", "TIINDIA", "TITAGARH", "VEDL", "ZFCVINDIA",
    "SHRIRAMFIN",
]
FUTURES_SYMBOLS = ["NIFTY", "BANKNIFTY"]


def get_token():
    token = os.environ.get("UPSTOX_ANALYTICAL_TOKEN")
    if not token:
        raise RuntimeError("UPSTOX_ANALYTICAL_TOKEN environment variable not set.")
    return token.strip()


# ---------------------------------------------------------------------------
# Retry-with-backoff (copied from app.py)
# ---------------------------------------------------------------------------
def _get_with_backoff(url, headers=None, params=None, timeout=20, max_retries=5, base_delay=1.5):
    last_exc = None
    resp = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exc = e
            time.sleep(base_delay * (2 ** attempt))
            continue
        if resp.status_code != 429:
            return resp
        retry_after = resp.headers.get("Retry-After")
        delay = float(retry_after) if retry_after else base_delay * (2 ** attempt)
        time.sleep(delay)
    if resp is None and last_exc is not None:
        raise last_exc
    return resp


# ---------------------------------------------------------------------------
# Bulk instrument master lookups (copied from app.py)
# ---------------------------------------------------------------------------
def _equity_master_cache_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"instrument_master_{now_ist().date().isoformat()}.json")


def _load_master_raw():
    path = _equity_master_cache_path()
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    resp = requests.get(INSTRUMENT_MASTER_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    resp.raise_for_status()
    with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as gz:
        master = json.load(gz)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(master, f)
    os.replace(tmp, path)
    return master


_EQUITY_MASTER_MAP = None
_FUTURES_MASTER_MAP = None


def _load_equity_master_map():
    master = _load_master_raw()
    return {
        inst["trading_symbol"].upper(): inst["instrument_key"]
        for inst in master
        if inst.get("segment") == "NSE_EQ" and inst.get("instrument_type") in ("EQ", "BE")
        and inst.get("trading_symbol")
    }


def _load_futures_master_map():
    master = _load_master_raw()
    nearest_by_underlying = {}
    for inst in master:
        if inst.get("segment") != "NSE_FO" or inst.get("instrument_type") != "FUT":
            continue
        underlying = (inst.get("underlying_symbol") or "").upper()
        if not underlying:
            continue
        expiry = inst.get("expiry", 0)
        current = nearest_by_underlying.get(underlying)
        if current is None or expiry < current[2]:
            nearest_by_underlying[underlying] = (inst["instrument_key"], inst.get("lot_size"), expiry)
    return {k: (v[0], v[1]) for k, v in nearest_by_underlying.items()}


def resolve_equity_instrument_key(symbol, token):
    global _EQUITY_MASTER_MAP
    if _EQUITY_MASTER_MAP is None:
        try:
            _EQUITY_MASTER_MAP = _load_equity_master_map()
        except Exception:
            _EQUITY_MASTER_MAP = {}
    key = _EQUITY_MASTER_MAP.get(symbol.upper())
    if key:
        return key, 1

    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    params = {"query": symbol, "exchanges": "NSE", "segments": "EQ",
              "instrument_types": "EQ", "page_number": 1, "records": 10}
    resp = _get_with_backoff(INSTRUMENT_SEARCH_URL, headers=headers, params=params)
    resp.raise_for_status()
    candidates = [inst for inst in resp.json().get("data", [])
                  if inst.get("trading_symbol", "").upper() == symbol.upper()]
    return (candidates[0]["instrument_key"], 1) if candidates else (None, None)


def resolve_futures_instrument_key(name, token):
    global _FUTURES_MASTER_MAP
    if _FUTURES_MASTER_MAP is None:
        try:
            _FUTURES_MASTER_MAP = _load_futures_master_map()
        except Exception:
            _FUTURES_MASTER_MAP = {}
    bulk_hit = _FUTURES_MASTER_MAP.get(name.upper())
    if bulk_hit:
        return bulk_hit[0], (int(bulk_hit[1]) if bulk_hit[1] else None)

    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    params = {"query": name, "exchanges": "NSE", "segments": "FO",
              "instrument_types": "FUT", "page_number": 1, "records": 30}
    resp = _get_with_backoff(INSTRUMENT_SEARCH_URL, headers=headers, params=params)
    resp.raise_for_status()
    candidates = [inst for inst in resp.json().get("data", [])
                  if inst.get("instrument_type") == "FUT"
                  and inst.get("underlying_symbol", "").upper() == name.upper()]
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x["expiry"])
    nearest = candidates[0]
    lot_size = nearest.get("lot_size")
    return nearest["instrument_key"], (int(lot_size) if lot_size else None)


# ---------------------------------------------------------------------------
# Candle fetch + zone computation (copied from app.py)
# ---------------------------------------------------------------------------
def fetch_candles(instrument_key, token, unit, interval, lookback_days):
    to_date = now_ist().strftime("%Y-%m-%d")
    from_date = (now_ist() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    resp = _get_with_backoff(url, headers=headers)
    resp.raise_for_status()
    candles = resp.json().get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    return df


def compute_composite_zones(intraday_df):
    if intraday_df.empty:
        return []
    try:
        _, shown = sahi_style_key_levels(
            intraday_df, n_bins=COMPOSITE_N_BINS, max_zones=MAX_ZONES,
            min_display_pct=MIN_DISPLAY_PCT, min_prominence_pct=MIN_PROMINENCE_PCT,
            min_bin_distance=MIN_BIN_DISTANCE,
        )
        return [asdict(z) for z in shown]
    except Exception:
        return []


def compute_ema_200(closes, period=200):
    if len(closes) < period:
        return None
    return float(pd.Series(closes).ewm(span=period, adjust=False).mean().iloc[-1])


def compute_intraday_zones(today_only_df):
    if today_only_df.empty:
        return []
    today = today_only_df["date"].max()
    today_df = today_only_df[today_only_df["date"] == today]
    try:
        _, shown = sahi_style_key_levels(
            today_df, n_bins=INTRADAY_N_BINS, max_zones=MAX_ZONES,
            min_display_pct=MIN_DISPLAY_PCT, min_prominence_pct=MIN_PROMINENCE_PCT,
            min_bin_distance=MIN_BIN_DISTANCE,
        )
        return [asdict(z) for z in shown]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main precompute loop (mirrors app.py's run_precompute, no Streamlit)
# ---------------------------------------------------------------------------
def run_precompute(token):
    cache = {}
    all_symbols = [(s, "futures") for s in FUTURES_SYMBOLS] + [(s, "equity") for s in EQUITY_SYMBOLS]
    total = len(all_symbols)
    for i, (symbol, kind) in enumerate(all_symbols):
        try:
            key, lot_size = (resolve_equity_instrument_key(symbol, token) if kind == "equity"
                              else resolve_futures_instrument_key(symbol, token))
            if key is None:
                print(f"  [{i + 1}/{total}] {symbol}: could not resolve instrument key, skipping.")
                continue
            daily_df = fetch_candles(key, token, "days", "1", DAILY_LOOKBACK_DAYS)
            intraday_df = fetch_candles(key, token, "minutes", "5", COMPOSITE_LOOKBACK_DAYS)

            prev_close = float(daily_df["close"].iloc[-1]) if not daily_df.empty else None
            if not intraday_df.empty:
                intraday_last_date = intraday_df["timestamp"].iloc[-1].date()
                daily_last_date = daily_df["timestamp"].iloc[-1].date() if not daily_df.empty else None
                if daily_last_date is None or intraday_last_date > daily_last_date:
                    prev_close = float(intraday_df["close"].iloc[-1])

            avg_daily_volume = (float(daily_df["volume"].tail(RVOL_BASELINE_DAYS).mean())
                                 if len(daily_df) >= RVOL_BASELINE_DAYS else None)
            composite_zones = compute_composite_zones(intraday_df)
            intraday_zones = compute_intraday_zones(intraday_df)
            ema_200 = compute_ema_200(intraday_df["close"].tolist()) if not intraday_df.empty else None

            cache[symbol] = {
                "instrument_key": key,
                "lot_size": lot_size,
                "prev_close": prev_close,
                "avg_daily_volume": avg_daily_volume,
                "composite_zones": composite_zones,
                "intraday_zones": intraday_zones,
                "ema_200": ema_200,
                "last_signal": "-",
                "zones_updated_at": now_ist().strftime("%Y-%m-%d %H:%M:%S"),
            }
            print(f"  [{i + 1}/{total}] {symbol}: ok (prev_close={prev_close})")
        except Exception as e:
            print(f"  [{i + 1}/{total}] {symbol}: precompute failed ({e}), skipping.")
        time.sleep(0.15)

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    print(f"\nPrecompute done. {len(cache)} symbols cached -> {CACHE_PATH}")
    return cache


if __name__ == "__main__":
    tok = get_token()
    run_precompute(tok)
