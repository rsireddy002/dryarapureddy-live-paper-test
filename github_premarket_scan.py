"""
Standalone premarket/intraday scan script for GitHub Actions.

Companion to github_precompute.py: that script runs once/day after close
and writes sahi_zones_cache.json (instrument keys, 18-day COMPOSITE zones,
avg_daily_volume). THIS script runs shortly after market open, reads that
committed cache, fetches ONLY today's still-forming intraday candles per
symbol (the fast "Refresh Zones" tier -- no re-fetch of 18 days of
history), and sends a buy/sell screen straight to your phone via
Telegram -- sector, trend, CVD direction, RVOL, and room to the nearest
validated zone -- with zero manual steps once this is set up.

Reads:
    UPSTOX_ANALYTICAL_TOKEN  -- same long-lived token github_precompute.py uses
    TELEGRAM_BOT_TOKEN       -- see setup notes at the bottom of this file
    TELEGRAM_CHAT_ID         -- see setup notes at the bottom of this file

Reuses (imports directly, does NOT duplicate) the two vendored modules
already in this repo:
    sahi_style_key_levels.py  -- zone segmentation (same as the live app)
    zone_validation.py        -- cross_validated_zones (same as the live app)
These two are pure-Python, no Streamlit dependency, so -- like
github_precompute.py already does with sahi_style_key_levels -- they run
fine headless in CI.

NOTE: EQUITY_SYMBOLS/SECTOR_MAP/nearest_zones below are deliberately
DUPLICATED from app.py (same reasoning github_precompute.py documents:
app.py itself can't be imported headless because it's full of
Streamlit-specific code). If you change these in app.py, mirror the
change here too.

Usage:
    UPSTOX_ANALYTICAL_TOKEN=xxx TELEGRAM_BOT_TOKEN=xxx TELEGRAM_CHAT_ID=xxx \
        python github_premarket_scan.py
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import timedelta, timezone, datetime

import numpy as np
import pandas as pd
import requests

from sahi_style_key_levels import sahi_style_key_levels
from zone_validation import cross_validated_zones

IST = timezone(timedelta(hours=5, minutes=30))
CACHE_PATH = "sahi_zones_cache.json"

INTRADAY_N_BINS = 45
MIN_PROMINENCE_PCT = 0.08
MIN_BIN_DISTANCE = 2
MAX_ZONES = 6
MIN_DISPLAY_PCT = 2.0

MAX_WORKERS = 16          # bounded concurrency -- see run_zone_refresh's
                          # docstring in app.py for why this isn't unbounded
TOP_N_BY_RVOL = 25        # how many of the most-active names to report on
MIN_CANDLES_FOR_SIGNAL = 3  # too early in the session (1-2 candles) is noise

SESSION_OPEN = "09:15"
SESSION_CLOSE = "15:30"
SESSION_MINUTES = 375  # 09:15 -> 15:30


# ---------------------------------------------------------------------------
# Sector grouping (copied from app.py's SECTOR_MAP)
# ---------------------------------------------------------------------------
SECTOR_MAP = {
    # Banks
    "HDFCBANK": "Banks", "ICICIBANK": "Banks", "SBIN": "Banks",
    "KOTAKBANK": "Banks", "AXISBANK": "Banks", "INDUSINDBK": "Banks",
    "BANDHANBNK": "Banks", "BANKBARODA": "Banks", "PNB": "Banks",
    "CANBK": "Banks", "IDFCFIRSTB": "Banks", "FEDERALBNK": "Banks",
    "RBLBANK": "Banks", "AUBANK": "Banks", "BANKINDIA": "Banks",
    "UNIONBANK": "Banks", "CUB": "Banks", "YESBANK": "Banks",

    # NBFC / Financial Services
    "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC", "CHOLAFIN": "NBFC",
    "MANAPPURAM": "NBFC", "MUTHOOTFIN": "NBFC", "LICHSGFIN": "NBFC",
    "MFSL": "NBFC", "PFC": "NBFC", "RECLTD": "NBFC", "SBICARD": "NBFC",
    "ABCAPITAL": "NBFC", "CANFINHOME": "NBFC", "SAMMAANCAP": "NBFC",
    "L&TFH": "NBFC", "M&MFIN": "NBFC", "LTF": "NBFC", "HUDCO": "NBFC",
    "IIFL": "NBFC", "MOTILALOFS": "NBFC", "ANGELONE": "NBFC",
    "JIOFIN": "NBFC", "SHRIRAMFIN": "NBFC", "HDFCAMC": "NBFC",
    "PAYTM": "NBFC", "POLICYBZR": "NBFC", "PIRAMALFIN": "NBFC",

    # Insurance
    "SBILIFE": "Insurance", "HDFCLIFE": "Insurance", "ICICIGI": "Insurance",
    "ICICIPRULI": "Insurance", "STARHEALTH": "Insurance", "GICRE": "Insurance",

    # Financial Infra / Exchanges
    "BSE": "Financial Infra", "CDSL": "Financial Infra", "IEX": "Financial Infra",

    # IT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "LTM": "IT", "MPHASIS": "IT", "PERSISTENT": "IT", "COFORGE": "IT",
    "OFSS": "IT", "NAUKRI": "IT", "BSOFT": "IT", "TATAELXSI": "IT",
    "INDIAMART": "IT",

    # Auto & Ancillaries
    "MARUTI": "Auto", "M&M": "Auto", "TMPV": "Auto", "EICHERMOT": "Auto",
    "HEROMOTOCO": "Auto", "BAJAJ-AUTO": "Auto", "TVSMOTOR": "Auto",
    "ASHOKLEY": "Auto", "MOTHERSON": "Auto", "BHARATFORG": "Auto",
    "BALKRISIND": "Auto", "MRF": "Auto", "APOLLOTYRE": "Auto",
    "EXIDEIND": "Auto", "ESCORTS": "Auto", "SONACOMS": "Auto",
    "TIINDIA": "Auto", "ZFCVINDIA": "Auto",

    # Pharma & Healthcare
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "AUROPHARMA": "Pharma", "BIOCON": "Pharma",
    "LUPIN": "Pharma", "TORNTPHARM": "Pharma", "ALKEM": "Pharma",
    "GLENMARK": "Pharma", "GRANULES": "Pharma", "LAURUSLABS": "Pharma",
    "IPCALAB": "Pharma", "ZYDUSLIFE": "Pharma", "MANKIND": "Pharma",
    "SYNGENE": "Pharma", "PFIZER": "Pharma",
    "APOLLOHOSP": "Healthcare", "FORTIS": "Healthcare", "MAXHEALTH": "Healthcare",
    "LALPATHLAB": "Healthcare", "METROPOLIS": "Healthcare",

    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "TATACONSUM": "FMCG",
    "BRITANNIA": "FMCG", "NESTLEIND": "FMCG", "DABUR": "FMCG",
    "MARICO": "FMCG", "COLPAL": "FMCG", "GODREJCP": "FMCG",
    "MCDOWELL-N": "FMCG", "UBL": "FMCG", "VBL": "FMCG", "PATANJALI": "FMCG",
    "JUBLFOOD": "FMCG", "GODFRYPHLP": "FMCG",

    # Metals & Mining
    "TATASTEEL": "Metals & Mining", "JSWSTEEL": "Metals & Mining",
    "HINDALCO": "Metals & Mining", "ADANIENT": "Metals & Mining",
    "VEDANTA": "Metals & Mining", "VEDL": "Metals & Mining",
    "NMDC": "Metals & Mining", "NATIONALUM": "Metals & Mining",
    "HINDCOPPER": "Metals & Mining", "SAIL": "Metals & Mining",
    "JINDALSTEL": "Metals & Mining", "APLAPOLLO": "Metals & Mining",

    # Oil & Gas / Energy
    "RELIANCE": "Oil & Gas", "ONGC": "Oil & Gas", "COALINDIA": "Oil & Gas",
    "GAIL": "Oil & Gas", "BPCL": "Oil & Gas", "IOC": "Oil & Gas",
    "HINDPETRO": "Oil & Gas", "PETRONET": "Oil & Gas", "OIL": "Oil & Gas",
    "IGL": "Oil & Gas", "MGL": "Oil & Gas",

    # Power
    "NTPC": "Power", "POWERGRID": "Power", "TATAPOWER": "Power",
    "TORNTPOWER": "Power", "NHPC": "Power", "SJVN": "Power",
    "CGPOWER": "Power",

    # Capital Goods & Defence
    "LT": "Capital Goods", "SIEMENS": "Capital Goods", "CUMMINSIND": "Capital Goods",
    "BHEL": "Capital Goods", "BEL": "Capital Goods", "HAL": "Capital Goods",
    "POLYCAB": "Capital Goods", "KEI": "Capital Goods", "SOLARINDS": "Capital Goods",
    "POWERINDIA": "Capital Goods", "GRAPHITE": "Capital Goods",
    "SUZLON": "Capital Goods", "ITI": "Capital Goods",

    # Cement & Construction Materials
    "ULTRACEMCO": "Cement", "GRASIM": "Cement", "SHREECEM": "Cement",
    "AMBUJACEM": "Cement", "DALBHARAT": "Cement", "JKCEMENT": "Cement",
    "INDIACEM": "Cement",

    # Chemicals
    "PIDILITIND": "Chemicals", "UPL": "Chemicals", "SRF": "Chemicals",
    "DEEPAKNTR": "Chemicals", "ATUL": "Chemicals", "GNFC": "Chemicals",
    "NAVINFLUOR": "Chemicals", "AARTIIND": "Chemicals", "TATACHEM": "Chemicals",
    "PIIND": "Chemicals", "ASTRAL": "Chemicals", "RAIN": "Chemicals",
    "SUPREMEIND": "Chemicals",

    # Consumer Durables
    "TITAN": "Consumer Durables", "ASIANPAINT": "Consumer Durables",
    "HAVELLS": "Consumer Durables", "VOLTAS": "Consumer Durables",
    "CROMPTON": "Consumer Durables", "DIXON": "Consumer Durables",
    "WHIRLPOOL": "Consumer Durables", "BATAINDIA": "Consumer Durables",
    "PGEL": "Consumer Durables",

    # Telecom
    "BHARTIARTL": "Telecom", "INDUSTOWER": "Telecom", "IDEA": "Telecom",
    "HFCL": "Telecom", "TATACOMM": "Telecom",

    # Realty
    "DLF": "Realty", "GODREJPROP": "Realty", "OBEROIRLTY": "Realty",
    "PRESTIGE": "Realty", "LODHA": "Realty",

    # Media & Entertainment
    "ZEEL": "Media", "SUNTV": "Media", "PVRINOX": "Media",

    # Retail / Consumer Services
    "TRENT": "Retail", "ETERNAL": "Retail", "DMART": "Retail",
    "NYKAA": "Retail", "PAGEIND": "Retail", "ABFRL": "Retail",
    "KALYANKJIL": "Retail",

    # Aviation / Logistics
    "INDIGO": "Aviation & Logistics", "CONCOR": "Aviation & Logistics",
    "GMRAIRPORT": "Aviation & Logistics", "DELHIVERY": "Aviation & Logistics",

    # Hotels & Travel
    "IRCTC": "Hotels & Travel", "INDHOTEL": "Hotels & Travel",

    # Construction & Infra
    "IRB": "Construction & Infra", "NBCC": "Construction & Infra",
    "NCC": "Construction & Infra", "RVNL": "Construction & Infra",
    "TITAGARH": "Construction & Infra",

    # Agri & Fertilizers
    "CHAMBLFERT": "Agri & Fertilizers", "COROMANDEL": "Agri & Fertilizers",

    # PSU Financial (rail/infra financing, distinct enough from private NBFC)
    "IRFC": "PSU Financial",

    # Diversified / Services
    "ADANIPORTS": "Diversified / Services",
}


def now_ist():
    return datetime.now(IST)


def get_env(name, required=True):
    val = os.environ.get(name)
    if required and not val:
        raise RuntimeError(f"{name} environment variable not set.")
    return val.strip() if val else val


# ---------------------------------------------------------------------------
# Retry-with-backoff (copied from app.py / github_precompute.py)
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
# Today's intraday candles (same endpoint as app.py's fetch_intraday_candles)
# ---------------------------------------------------------------------------
def fetch_intraday_candles(instrument_key, token, unit="minutes", interval="5"):
    url = f"https://api.upstox.com/v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    try:
        resp = _get_with_backoff(url, headers=headers, timeout=20)
        resp.raise_for_status()
        candles = resp.json().get("data", {}).get("candles", [])
    except requests.exceptions.RequestException:
        return pd.DataFrame()
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_intraday_zones(today_df):
    if today_df.empty:
        return []
    try:
        _, shown = sahi_style_key_levels(
            today_df, n_bins=INTRADAY_N_BINS, max_zones=MAX_ZONES,
            min_display_pct=MIN_DISPLAY_PCT, min_prominence_pct=MIN_PROMINENCE_PCT,
            min_bin_distance=MIN_BIN_DISTANCE,
        )
        return [asdict(z) for z in shown]
    except Exception:
        return []


def compute_cumulative_volume_delta(df):
    """Same candle-color approximation as candles_with_levels.py -- an
    up-close candle counts its volume as buying pressure, a down-close
    candle as selling. This is a proxy, not tick-accurate order flow."""
    delta = np.where(df["close"] > df["open"], df["volume"],
                      np.where(df["close"] < df["open"], -df["volume"], 0))
    return pd.Series(delta, index=df.index).cumsum()


def nearest_zones(ltp, validated_zones):
    """Same logic as app.py's nearest_zones -- closest validated support
    (below ltp) and resistance (above ltp), with % distance to each."""
    support, support_dist = None, None
    resistance, resistance_dist = None, None
    for z in validated_zones:
        if ltp is None:
            break
        if z["price_mode"] <= ltp:
            dist = abs(ltp - z["price_high"]) / ltp * 100
            if support_dist is None or dist < support_dist:
                support, support_dist = z, dist
        else:
            dist = abs(z["price_low"] - ltp) / ltp * 100
            if resistance_dist is None or dist < resistance_dist:
                resistance, resistance_dist = z, dist
    return support, support_dist, resistance, resistance_dist


def classify_symbol(symbol, today_df, composite_zones, avg_daily_volume):
    """Combines trend (price vs. today's open) + CVD direction (candle-
    color proxy) + position relative to the nearest cross-validated zone
    -- same three ingredients used for the manual By-RVOL read, just
    computed directly from candles instead of read off a chart."""
    if today_df is None or today_df.empty or len(today_df) < MIN_CANDLES_FOR_SIGNAL:
        return None

    ltp = float(today_df["close"].iloc[-1])
    day_open = float(today_df["open"].iloc[0])
    cvd = compute_cumulative_volume_delta(today_df)
    cvd_now = float(cvd.iloc[-1])
    lookback = min(6, len(cvd) - 1)
    cvd_slope = float(cvd.iloc[-1] - cvd.iloc[-1 - lookback]) if lookback > 0 else 0.0

    intraday_zones = compute_intraday_zones(today_df)
    try:
        val_comp, _, _ = cross_validated_zones(composite_zones or [], intraday_zones)
    except Exception:
        val_comp = []
    support, support_dist, resistance, resistance_dist = nearest_zones(ltp, val_comp)

    trend = "up" if ltp > day_open else ("down" if ltp < day_open else "flat")
    cvd_dir = "buying" if cvd_slope > 0 else ("selling" if cvd_slope < 0 else "flat")

    if trend == "up" and cvd_dir != "selling":
        bias = "BUY"
    elif trend == "down" and cvd_dir != "buying":
        bias = "SELL"
    elif trend == "up" and cvd_dir == "selling":
        bias = "CAUTION"       # price up, volume selling -- bearish divergence
    elif trend == "down" and cvd_dir == "buying":
        bias = "WATCH"         # price down, volume buying -- possible reversal
    else:
        bias = "NEUTRAL"

    today_vol = float(today_df["volume"].sum())
    elapsed_minutes = max(
        5.0,
        (today_df["timestamp"].iloc[-1] - today_df["timestamp"].iloc[0]).total_seconds() / 60.0 + 5.0,
    )
    rvol_pct = None
    if avg_daily_volume:
        expected_by_now = float(avg_daily_volume) * (elapsed_minutes / SESSION_MINUTES)
        if expected_by_now > 0:
            rvol_pct = today_vol / expected_by_now * 100.0

    return {
        "symbol": symbol,
        "sector": SECTOR_MAP.get(symbol, "Other"),
        "ltp": ltp,
        "trend": trend,
        "cvd_dir": cvd_dir,
        "bias": bias,
        "rvol_pct": rvol_pct,
        "support": support,
        "support_dist": support_dist,
        "resistance": resistance,
        "resistance_dist": resistance_dist,
    }


def send_telegram(text, bot_token, chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    # Telegram's hard cap is 4096 chars per message -- split on blank
    # lines so a long screen arrives as a few readable messages instead
    # of one truncated one.
    chunks, current = [], ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > 3500:
            chunks.append(current)
            current = ""
        current += line + "\n"
    if current:
        chunks.append(current)
    for chunk in chunks:
        resp = requests.post(url, data={"chat_id": chat_id, "text": chunk}, timeout=20)
        if resp.status_code != 200:
            print(f"Telegram send failed ({resp.status_code}): {resp.text}", file=sys.stderr)


def format_message(ranked):
    ts = now_ist().strftime("%d %b %Y, %H:%M IST")
    lines = [f"Premarket scan -- {ts}", f"Top {len(ranked)} by RVOL", ""]

    buys = [r for r in ranked if r["bias"] == "BUY"]
    sells = [r for r in ranked if r["bias"] == "SELL"]
    watch = [r for r in ranked if r["bias"] in ("CAUTION", "WATCH")]

    def _room(r):
        if r["bias"] == "SELL" and r["support_dist"] is not None:
            return f", room to support {r['support_dist']:.1f}%"
        if r["bias"] == "BUY" and r["resistance_dist"] is not None:
            return f", room to resistance {r['resistance_dist']:.1f}%"
        return ""

    if buys:
        lines.append("BUY bias:")
        for r in buys:
            rvol = f"{r['rvol_pct']:.0f}%" if r["rvol_pct"] else "n/a"
            lines.append(f"  {r['symbol']} ({r['sector']}) - RVOL {rvol}, CVD {r['cvd_dir']}{_room(r)}")
        lines.append("")

    if sells:
        lines.append("SELL bias:")
        for r in sells:
            rvol = f"{r['rvol_pct']:.0f}%" if r["rvol_pct"] else "n/a"
            lines.append(f"  {r['symbol']} ({r['sector']}) - RVOL {rvol}, CVD {r['cvd_dir']}{_room(r)}")
        lines.append("")

    if watch:
        lines.append("Divergence / watch:")
        for r in watch:
            note = "price up, selling volume" if r["bias"] == "CAUTION" else "price down, buying volume"
            lines.append(f"  {r['symbol']} ({r['sector']}) - {note}")
        lines.append("")

    # Simple sector-strength rollup: how many BUY vs SELL names per sector
    # among everything scanned (not just the printed top names) -- this is
    # the "sector strength" ingredient from the manual read, automated.
    sector_tally = {}
    for r in ranked:
        if r["bias"] not in ("BUY", "SELL"):
            continue
        tally = sector_tally.setdefault(r["sector"], {"BUY": 0, "SELL": 0})
        tally[r["bias"]] += 1
    notable_sectors = [
        (sector, t) for sector, t in sector_tally.items()
        if t["BUY"] + t["SELL"] >= 2 and abs(t["BUY"] - t["SELL"]) >= 2
    ]
    if notable_sectors:
        lines.append("Sector theme:")
        for sector, t in sorted(notable_sectors, key=lambda kv: -(kv[1]["BUY"] + kv[1]["SELL"])):
            lean = "buying" if t["BUY"] > t["SELL"] else "selling"
            lines.append(f"  {sector}: broadly {lean} ({t['BUY']} buy / {t['SELL']} sell)")

    lines.append("")
    lines.append("CVD is the candle-color volume approximation, not tick-level order flow. Not investment advice.")
    return "\n".join(lines)


def main():
    token = get_env("UPSTOX_ANALYTICAL_TOKEN")
    bot_token = get_env("TELEGRAM_BOT_TOKEN")
    chat_id = get_env("TELEGRAM_CHAT_ID")

    if not os.path.exists(CACHE_PATH):
        raise RuntimeError(
            f"{CACHE_PATH} not found -- run the scheduled-precompute workflow "
            f"at least once (or 'workflow_dispatch' it manually) before this scan."
        )
    with open(CACHE_PATH) as f:
        cache = json.load(f)

    symbols = list(cache.keys())
    print(f"Scanning {len(symbols)} symbols from cached instrument keys...")

    def _fetch(symbol):
        key = cache[symbol]["instrument_key"]
        df = fetch_intraday_candles(key, token)
        return symbol, df

    fetched = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch, s): s for s in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                _, df = future.result()
                fetched[symbol] = df
            except Exception as e:
                print(f"  {symbol}: fetch failed ({e})")

    results = []
    for symbol, df in fetched.items():
        try:
            r = classify_symbol(
                symbol, df,
                cache[symbol].get("composite_zones", []),
                cache[symbol].get("avg_daily_volume"),
            )
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"  {symbol}: classify failed ({e})")

    ranked_all = sorted(
        [r for r in results if r["rvol_pct"] is not None],
        key=lambda r: r["rvol_pct"], reverse=True,
    )
    top = ranked_all[:TOP_N_BY_RVOL]

    print(f"Classified {len(results)} symbols, {len(ranked_all)} with RVOL data.")
    message = format_message(top if top else ranked_all[:TOP_N_BY_RVOL])
    print("\n" + message)
    send_telegram(message, bot_token, chat_id)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# ONE-TIME SETUP NOTES (not executed -- just documentation)
#
# 1. Create a Telegram bot:
#    - Open Telegram, message @BotFather, send "/newbot", follow the
#      prompts. It replies with a token like "123456:ABC-DEF..." --
#      that's TELEGRAM_BOT_TOKEN.
#    - Send your new bot ANY message first (e.g. "hi") so it's allowed
#      to message you back.
#
# 2. Get your chat ID:
#    - Visit https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates in
#      a browser right after step 1's "hi" message.
#    - Look for "chat":{"id": 123456789, ...} in the JSON -- that number
#      is TELEGRAM_CHAT_ID.
#
# 3. Add three repo secrets (Settings -> Secrets and variables -> Actions
#    -> New repository secret): UPSTOX_ANALYTICAL_TOKEN (already exists
#    if scheduled-precompute.yml is set up), TELEGRAM_BOT_TOKEN,
#    TELEGRAM_CHAT_ID.
#
# 4. Add this file (github_premarket_scan.py) and the accompanying
#    workflow (.github/workflows/premarket-scan.yml) to the repo root.
# ---------------------------------------------------------------------------
