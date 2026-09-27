"""
app.py - Dr Yarapu Reddy Levels (cross-timeframe validated zones + alerts)

Builds on the sahi-key-levels module (vendored below, UNCHANGED) to add:
  1. Two independently-computed zone sets per symbol: a COMPOSITE profile
     (18 trading days) and an INTRADAY profile (today's session only).
  2. Cross-timeframe validation (see zone_validation.py): a zone only
     counts as a real level if its price range shows up in BOTH profiles.
     A zone with no multi-day backing, or a composite zone today's session
     hasn't touched at all, is treated as noise and dropped.
  3. BUY/SELL alerts, edge-triggered (only logged the moment a symbol's
     signal changes, not every refresh) and gated to market hours (same
     fix already applied in hvn-lvn-scanner: an after-hours refresh pulls
     Upstox's frozen post-close quotes, which must not get logged as a
     live signal).
  4. A Chart tab: candlesticks for today's session with composite,
     intraday, and validated zones overlaid (see candles_with_levels.py).

VENDORED FILES (copied unchanged from their source repos, per instruction
to leave the original logic untouched):
    hvn_lvn.py               <- from hvn-lvn-scanner
    sahi_style_key_levels.py <- from sahi-key-levels

THREE-TIER REFRESH MODEL (deliberate, not accidental complexity):
  - "Run Precompute" (slow, once/day): resolves instrument keys, fetches
    18 days of 5-min candles per symbol, computes the COMPOSITE zone set.
    This is the expensive step -- same reasoning as hvn-lvn-scanner's
    Precompute.
  - "Refresh Zones" (medium, every few minutes -- NOT on every quote tick):
    re-fetches ONLY today's 5-min candles per symbol (a much lighter
    historical-candle call than the 18-day Precompute fetch, but still one
    HTTP call per symbol, so this is not free -- don't wire it to run on
    every quote refresh across 200+ symbols). Recomputes the INTRADAY zone
    set, cross-validates against the cached COMPOSITE set, and logs any
    new BUY/SELL alerts.
  - "Refresh Quotes" (fast): single batch quote call for LTP/VWAP, same as
    hvn-lvn-scanner's existing fast refresh. Recomputes each symbol's
    signal against whatever zones were last computed by "Refresh Zones"
    (may be a few minutes stale) -- this keeps the Scanner table feeling
    responsive without re-fetching candles on every tick.

SETUP:
    pip install streamlit requests pandas numpy plotly --break-system-packages
    $env:UPSTOX_ACCESS_TOKEN = "your_token_here"
    streamlit run app.py
"""
import os
import re
import json
import socket
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone, time as dtime
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from hvn_lvn import build_volume_profile, find_hvn_lvn
from sahi_style_key_levels import sahi_style_key_levels
from zone_validation import cross_validated_zones, compute_zone_signal, compute_cvd_zone_signal
from candles_with_levels import (
    plot_candles_with_zones, build_cvd_chart, compute_cumulative_volume_delta,
    compute_recent_order_flow_imbalance_pct,
)
from ml_predict import predict_break_probability
from live_feed_reader import get_live_candles
from candle_aggregator import INTERVALS_SECONDS  # which intervals the live feed can serve ("1"/"5"/"15")

# Force IPv4 for every outbound connection this process makes. Confirmed
# necessary via a real debugging session: this app's network silently
# prefers IPv6 by default, which routes outbound requests from an address
# Upstox's static-IP allowlist doesn't recognize (Upstox's restriction is
# IPv4-only) -- Upstox's own error body (UDAPI1154) named the exact
# mismatched IPv6 address when this was diagnosed via place_real_order.py
# and feed_listener.py. Applied globally here (not just around the real-
# order call below) since it's a no-op on an IPv4-only network and every
# Upstox API call in this app benefits from it, not just order placement.
#
# GUARDED to run only once per process (via a marker on the socket module
# itself, not a local/global variable in this file) -- Streamlit reruns
# this entire script on every interaction, but reuses the SAME process and
# module namespace each time rather than starting fresh. Without the
# guard, the second rerun would read socket.getaddrinfo (already patched
# by the first rerun) into _orig_getaddrinfo, then redefine
# _getaddrinfo_ipv4_only -- and since that function looks up
# _orig_getaddrinfo BY NAME each call (not a frozen reference), it would
# end up resolving to itself and recurse forever the next time it's
# called. This is exactly what produced the RecursionError seen on
# Streamlit Cloud after the app had been running for a while.
if not getattr(socket, "_dryarapureddy_ipv4_patched", False):
    _orig_getaddrinfo = socket.getaddrinfo

    def _getaddrinfo_ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
        return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _getaddrinfo_ipv4_only
    socket._dryarapureddy_ipv4_patched = True

IST = timezone(timedelta(hours=5, minutes=30))

def now_ist():
    return datetime.now(IST)


def current_candle_boundary_key(now_dt):
    """Returns a string key identifying the most recently completed
    5-min candle boundary at or before now_dt, e.g. '2026-09-08 10:05'.
    Comparing this key across reruns lets the app detect exactly once
    per actual candle close, regardless of how often the underlying UI
    rerun timer ticks in between."""
    floored_minute = (now_dt.minute // CANDLE_INTERVAL_MINUTES) * CANDLE_INTERVAL_MINUTES
    boundary = now_dt.replace(minute=floored_minute, second=0, microsecond=0)
    return boundary.strftime("%Y-%m-%d %H:%M")


def seconds_to_next_candle_close(now_dt, interval_minutes):
    """Seconds remaining until the current interval_minutes-candle closes
    -- feeds the Dashboard tab's 'Current candle closes in' readout.
    Parameterized by interval_minutes (unlike current_candle_boundary_key
    above, which is hardcoded to CANDLE_INTERVAL_MINUTES for the scan
    cadence) so it works for whichever interval the Dashboard's switcher
    currently has selected."""
    floored_minute = (now_dt.minute // interval_minutes) * interval_minutes
    boundary = now_dt.replace(minute=floored_minute, second=0, microsecond=0)
    next_close = boundary + timedelta(minutes=interval_minutes)
    return max(0, int((next_close - now_dt).total_seconds()))

# ---------------- Config ----------------
INSTRUMENT_SEARCH_URL = "https://api.upstox.com/v2/instruments/search"
QUOTES_URL = "https://api.upstox.com/v2/market-quote/quotes"
UPSTOX_AUTHORIZE_URL = "https://api.upstox.com/v2/login/authorization/dialog"
UPSTOX_TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"
CACHE_PATH = "sahi_zones_cache.json"
ALERT_LOG_PATH = "alert_log.json"
FNO_LIVE_CANDLES_PATH = os.environ.get("FNO_LIVE_CANDLES_PATH", "fno_live_candles.json")
TRIPLE_CROSS_LOG_PATH = "triple_cross_log.json"  # written by tick_paper_trader.py, read-only here
# ^ same file feed_listener.py writes -- see live_ticks.py's original
# standalone version for the full explanation. Read-only here, no new
# dependency: just os/json, both already used elsewhere in this file.

DAILY_LOOKBACK_DAYS = 60         # needs enough history for RVOL_BASELINE_DAYS average --
                                  # was 30, but Upstox's historical-candle endpoint only
                                  # returns finalized/settled days (lags behind "now" by a
                                  # few trading days -- see candle_store.py's writeup on
                                  # algotrade-unified-v2 for the same underlying behavior),
                                  # so 30 calendar days plus that lag plus any holidays could
                                  # land under RVOL_BASELINE_DAYS trading days -- exactly what
                                  # was happening: avg_daily_volume ended up None for every
                                  # symbol, so RVOL% showed "None" everywhere despite LTP/VWAP
                                  # working fine. 60 calendar days leaves a comfortable margin.
COMPOSITE_LOOKBACK_DAYS = 18     # matches hvn-lvn-scanner's multi-day window
RVOL_BASELINE_DAYS = 20          # prior-N-day average full-day volume, same convention as hvn-lvn-scanner
TOP_N_RVOL = 5                   # only symbols in the top N by RVOL are eligible to alert

COMPOSITE_N_BINS = 50
INTRADAY_N_BINS = 45
MIN_PROMINENCE_PCT = 0.08
MIN_BIN_DISTANCE = 2
MAX_ZONES = 6
MIN_DISPLAY_PCT = 2.0
MIN_SIGNAL_DISTANCE_PCT = 0.5    # how far LTP must be from a validated zone to signal
MIN_VWAP_DISTANCE_PCT = 0.15     # how far LTP must be from VWAP before a bias counts as real
                                  # (found necessary live: without this, tiny VWAP wobbles of
                                  # 0.02-0.05% fired repeated BUY/SELL flips on the same symbol)
CANDLE_INTERVAL_MINUTES = 5
CANDLE_CLOSE_BUFFER_SECONDS = 5     # wait this long past each 5-min boundary before scanning, so
                                     # the quote has settled to reflect the just-closed candle
UI_RERUN_INTERVAL_SECONDS = 10      # how often Streamlit reruns to CHECK whether a candle just
                                     # closed -- NOT the scan cadence itself, which only fires
                                     # once per actual candle close regardless of this tick rate

MARKET_OPEN_TIME = dtime(9, 15)   # IST - no new alerts logged before this
MARKET_CLOSE_TIME = dtime(15, 30)  # IST - no new alerts logged at/after this

NEAR_ZONE_PCT = 0.3   # how close (%) LTP must be to a validated zone edge to count as "at" it

# "Premium" zone = high volume concentration (real trading interest, not
# a thin/noisy cluster) AND the ML model is confident it'll hold rather
# than break. Both thresholds intentionally reuse the same bar already
# used elsewhere (PAPER_TRADE_ML_RISK_THRESHOLD) for consistency -- a
# "premium" zone is exactly the kind of level the existing entry logic
# already trusts enough to trade against.
PREMIUM_MIN_ZONE_PCT = 20.0     # zone's share of session volume must be at least this high
PREMIUM_MAX_ML_RISK_PCT = 15.0  # ML-predicted break probability must be below this

CVD_QUALITY_LOOKBACK_CANDLES = 6  # recent-candle window for compute_recent_order_flow_imbalance_pct

# --- Paper trading (SIMULATED, no real orders) ---
PAPER_TRADE_LOG_PATH = "paper_trades.json"
HEARTBEAT_PATH = "daemon_heartbeat.json"
PAPER_TRADE_SIZE_RUPEES = 25000   # fixed rupee amount per simulated trade
PAPER_TRADE_ML_RISK_THRESHOLD = 15.0  # entry only if the crossed level's ML break-risk is BELOW this %
PAPER_TRADE_UNIVERSE_TOP_N = 10  # only the top-N Wide Range stocks (widest support-resistance gap) are eligible for entries

# --- Real order placement (REAL MONEY, Dashboard tab's "Real" trade mode
# only -- everywhere else in this app stays simulated). Same endpoint/
# payload shape proven working by the standalone place_real_order.py
# diagnostic script earlier in this project (api-hft.upstox.com is
# Upstox's dedicated low-latency order endpoint, separate from
# api.upstox.com which the rest of this app uses for quotes/candles). ---
REAL_ORDER_URL = "https://api-hft.upstox.com/v3/order/place"
REAL_ORDER_LOG_PATH = "real_orders_log.json"
# Saved once a login exchange succeeds, so API Key/Secret/Redirect URI
# don't need retyping into the sidebar on every restart -- same
# "typed once, persisted to disk, gitignored" pattern as
# upstox_token.txt for the token itself. Not committed (see .gitignore);
# the secret is stored in plaintext here, which is the same trust level
# this app already applies to the access token it saves the same way.
UPSTOX_OAUTH_CONFIG_PATH = "upstox_oauth_config.json"

# Full liquid NSE F&O universe (not restricted to Nifty 50 anymore) --
# same universe proven out across the other repos (hvn-lvn-scanner,
# fno-scanner-strategy-update). NIFTY/BANKNIFTY handled separately below
# as futures, not equity. This list drifts over time as NSE adds/removes
# F&O eligibility, so it's worth a periodic sanity check, not treated as
# permanently fixed.
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

# Sector grouping for the full F&O universe above -- used by the Sectors
# tab to render a grid of small charts for one sector at a time. Best-
# effort NSE-style categorization; a few names are genuinely borderline
# (e.g. Adani Enterprises is diversified, PFC/RECLTD are PSU financiers
# grouped under NBFC here) -- treat this as a practical scanning grouping,
# not a formal index classification.
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


def get_token():
    # Sidebar-entered token takes priority -- lets you paste a fresh
    # token here each day without touching the systemd service file or
    # restarting anything, and keeps this app's token independent from
    # whatever the other app/listener is using (avoids session clashes
    # from two connections sharing one token).
    sidebar_token = st.session_state.get("manual_token")
    if sidebar_token:
        return sidebar_token.strip()
    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if token:
        return token.strip()
    try:
        if "UPSTOX_ACCESS_TOKEN" in st.secrets:
            return st.secrets["UPSTOX_ACCESS_TOKEN"].strip()
    except Exception:
        pass
    if os.path.exists("upstox_token.txt"):
        with open("upstox_token.txt", "r") as f:
            t = f.read().strip()
        if t and t != "PASTE_YOUR_TOKEN_HERE":
            return t
    raise RuntimeError(
        "No token found. Enter one in the sidebar, set $env:UPSTOX_ACCESS_TOKEN, add "
        "UPSTOX_ACCESS_TOKEN to Streamlit secrets, or create upstox_token.txt."
    )


def build_upstox_login_url(api_key, redirect_uri):
    """The URL to send the user to in a browser to approve this app --
    step 1 of Upstox's OAuth authorization-code flow. Upstox redirects
    back to redirect_uri with ?code=... in the query string once
    approved (the redirect_uri itself doesn't need to resolve to
    anything real -- the code just needs to be visible in the browser's
    address bar to copy back out)."""
    return (
        f"{UPSTOX_AUTHORIZE_URL}?response_type=code&client_id={quote(api_key, safe='')}"
        f"&redirect_uri={quote(redirect_uri, safe='')}"
    )


def exchange_upstox_auth_code(auth_code, api_key, api_secret, redirect_uri):
    """Step 2: trades the short-lived authorization code for a real
    access_token. This is the ONLY thing that determines what
    permissions the resulting token carries -- whatever this app's
    trading/market-data scopes are set to in the Upstox developer
    console, a token minted this way inherits them, unlike a token
    grabbed some other way that may turn out read-only (UDAPI100067).
    Returns (ok, message_or_token_dict)."""
    try:
        resp = requests.post(
            UPSTOX_TOKEN_URL,
            headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            data={
                "code": auth_code.strip(),
                "client_id": api_key.strip(),
                "client_secret": api_secret.strip(),
                "redirect_uri": redirect_uri.strip(),
                "grant_type": "authorization_code",
            },
            timeout=20,
        )
    except requests.exceptions.RequestException as e:
        return False, f"Request failed before reaching Upstox: {e}"
    try:
        body = resp.json()
    except Exception:
        return False, f"Non-JSON response (HTTP {resp.status_code}): {resp.text[:300]}"
    if resp.status_code >= 400:
        return False, f"Rejected by Upstox (HTTP {resp.status_code}): {body}"
    if not body.get("access_token"):
        return False, f"No access_token in response: {body}"
    return True, body


def load_saved_oauth_config():
    """Whatever API Key/Secret/Redirect URI last worked, if anything --
    read once per script run to prefill the sidebar's login fields so
    they don't need retyping every restart. Missing/corrupt file just
    means "nothing saved yet", not an error."""
    if os.path.exists(UPSTOX_OAUTH_CONFIG_PATH):
        try:
            with open(UPSTOX_OAUTH_CONFIG_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_oauth_config(api_key, api_secret, redirect_uri):
    """Called only right after a login exchange actually succeeds --
    saving credentials that just proved they work, not whatever's
    sitting in the form. Best-effort: a write failure here shouldn't
    break the login that already succeeded."""
    try:
        with open(UPSTOX_OAUTH_CONFIG_PATH, "w") as f:
            json.dump({"api_key": api_key, "api_secret": api_secret, "redirect_uri": redirect_uri}, f)
    except OSError:
        pass


import gzip as _gzip
import io as _io
from datetime import date as _date

INSTRUMENT_MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"


def _get_with_backoff(url, headers=None, params=None, timeout=20, max_retries=5, base_delay=1.5):
    """requests.get wrapper that retries on 429 (honoring Retry-After if
    sent) AND on connection-level failures (DNS/network drops), instead
    of raising immediately on either. After max_retries, re-raises the
    last connection error if that's what kept failing, or returns the
    last 429 response (unchanged behavior: caller's .raise_for_status(),
    if any, surfaces that final failure)."""
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


def _equity_master_cache_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"instrument_master_{_date.today().isoformat()}.json")


def _load_equity_master_map():
    """Downloads Upstox's full instrument master once per calendar day
    (cached to disk), returns {TRADING_SYMBOL: instrument_key} for NSE
    cash-market equities. Replaces 220 individual /instruments/search
    calls with a single download."""
    path = _equity_master_cache_path()
    if os.path.exists(path):
        with open(path, "r") as f:
            master = json.load(f)
    else:
        resp = requests.get(INSTRUMENT_MASTER_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        resp.raise_for_status()
        with _gzip.GzipFile(fileobj=_io.BytesIO(resp.content)) as gz:
            master = json.load(gz)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(master, f)
        os.replace(tmp, path)
    return {
        inst["trading_symbol"].upper(): inst["instrument_key"]
        for inst in master
        if inst.get("segment") == "NSE_EQ" and inst.get("instrument_type") in ("EQ", "BE")
        and inst.get("trading_symbol")
    }


_FUTURES_MASTER_MAP = None


def _load_futures_master_map():
    """Reuses the SAME cached instrument master file (no extra download)
    to build {UNDERLYING_SYMBOL: (instrument_key, lot_size)} for NSE_FO
    index futures, nearest expiry per underlying. Fixes the persistent
    429s on NIFTY/BANKNIFTY, which always used to hit the search API."""
    path = _equity_master_cache_path()
    if os.path.exists(path):
        with open(path, "r") as f:
            master = json.load(f)
    else:
        resp = requests.get(INSTRUMENT_MASTER_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        resp.raise_for_status()
        with _gzip.GzipFile(fileobj=_io.BytesIO(resp.content)) as gz:
            master = json.load(gz)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(master, f)
        os.replace(tmp, path)

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


_EQUITY_MASTER_MAP = None


def resolve_equity_instrument_key(symbol, token):
    """Returns (instrument_key, lot_size) -- lot_size is always 1 for
    equities (quantity there just means share count), returned anyway so
    callers can treat equities and futures uniformly.

    PATCHED: looks up the bulk instrument master first (one download/day,
    zero per-symbol API calls) instead of hitting /instruments/search 220
    times per precompute run -- that loop was the source of the 429 flood.
    Falls back to the old search-based lookup only if the symbol isn't
    found in the master (e.g. a very recently listed stock)."""
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
    """No expiry filter - sorts client-side by expiry (same fix already
    applied in hvn-lvn-scanner: 'current_month' keyword returns zero
    results once that month's contract expires but the calendar hasn't
    rolled over yet).

    Returns (instrument_key, lot_size). lot_size is read live from
    Upstox's own instrument-search response (never hardcoded) since NSE
    periodically revises F&O lot sizes -- a stale hardcoded value would
    silently produce a wrong-sized real order once a revision happens."""
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


def resolve_commodity_instrument_key(name, token):
    """Nearest-expiry lookup for a commodity future, same pattern as
    resolve_futures_instrument_key. Returns (instrument_key, lot_size,
    exchange) -- the exchange string is included so the caller can show
    it for a human sanity check before ever placing a real order, and
    because it's the crux of a real live issue (see below).

    MCX orders are, as of testing, TEMPORARILY DISABLED platform-wide by
    Upstox itself (order placement returns UDAPI1161: "MCX API orders
    are temporarily disabled. Meanwhile, place commodity orders on NSE
    (NSCOM).") -- straight from Upstox, not this app. So this tries
    exchanges="NSCOM" first (the exact name Upstox's own error message
    uses), then falls back to exchanges="NSE" with a commodity segment
    filter if "NSCOM" isn't accepted as a query value (Upstox's public
    docs are inconsistent about which one the Instrument Search API
    itself expects). Either way, any candidate actually on MCX is
    filtered out here, since that's the exchange that's broken right
    now -- don't resolve to an instrument that's guaranteed to be
    rejected again.

    IMPORTANT -- do NOT reuse the F&O quantity math with this lot_size.
    Per Upstox's own Place Order API docs, the `quantity` field means
    something different by segment: for F&O/equities it's a unit count
    (must be a multiple of lot_size), but for commodities it's already
    the NUMBER OF LOTS -- lot_size here is informational only. Upstox's
    own developer forum has a report of this exact field's behavior
    being inconsistent in production, so treat lot_size from this
    function as "for display", not as a multiplier."""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}

    def _search(exchanges):
        params = {"query": name, "exchanges": exchanges, "segments": "COMM",
                  "instrument_types": "FUT", "page_number": 1, "records": 30}
        resp = _get_with_backoff(INSTRUMENT_SEARCH_URL, headers=headers, params=params)
        if resp.status_code >= 400:
            return []
        return resp.json().get("data", [])

    data = _search("NSCOM") or _search("NSE")
    candidates = [inst for inst in data
                  if inst.get("instrument_type") == "FUT"
                  and inst.get("underlying_symbol", "").upper() == name.upper()
                  and inst.get("exchange", "").upper() != "MCX"]
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda x: x["expiry"])
    nearest = candidates[0]
    lot_size = nearest.get("lot_size")
    return nearest["instrument_key"], (int(lot_size) if lot_size else None), nearest.get("exchange")


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


def fetch_5min_candles_ending(instrument_key, token, end_date, total_days):
    """Same as fetch_candles(unit='minutes', interval='5', ...) but anchored
    to an arbitrary past end_date instead of 'now' -- needed for the Replay
    tab, which looks at historical sessions, not today.

    Chunks into <=20-day windows and concatenates: Upstox's 5-min
    historical-candle endpoint rejects overly wide date ranges in one call
    (a 400 for ~30+ days -- same limit discovered and worked around in
    backtest_zone_formation.py)."""
    all_chunks = []
    remaining = total_days
    cursor_end = end_date
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}

    while remaining > 0:
        chunk_days = min(20, remaining)
        chunk_start = cursor_end - timedelta(days=chunk_days)
        url = (f"https://api.upstox.com/v3/historical-candle/{instrument_key}/minutes/5/"
               f"{cursor_end.strftime('%Y-%m-%d')}/{chunk_start.strftime('%Y-%m-%d')}")
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        candles = resp.json().get("data", {}).get("candles", [])
        if candles:
            all_chunks.append(pd.DataFrame(
                candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"]
            ))
        cursor_end = chunk_start
        remaining -= chunk_days

    if not all_chunks:
        return pd.DataFrame()

    df = pd.concat(all_chunks, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_replay_data(instrument_key, token, replay_date_str):
    """Cached per (symbol, date) -- the Replay tab's slider re-runs this
    function's CALLER on every drag, but the actual fetch only happens
    once per symbol/date pick, not once per slider position. Returns
    (composite_zones, day_df) or (None, None) if there's not enough
    history or no candles for that day."""
    replay_date = datetime.strptime(replay_date_str, "%Y-%m-%d").date()
    total_days = COMPOSITE_LOOKBACK_DAYS + 25  # buffer for weekends/holidays
    full_df = fetch_5min_candles_ending(instrument_key, token, replay_date, total_days)
    if full_df.empty:
        return None, None

    trading_days = sorted(full_df["date"].unique())
    if replay_date not in trading_days:
        return None, None
    day_idx = trading_days.index(replay_date)
    composite_days = trading_days[max(0, day_idx - COMPOSITE_LOOKBACK_DAYS):day_idx]
    if not composite_days:
        return None, None

    composite_df = full_df[full_df["date"].isin(composite_days)]
    composite_zones = compute_composite_zones(composite_df)
    day_df = full_df[full_df["date"] == replay_date].reset_index(drop=True)
    return composite_zones, day_df


def compute_composite_zones(intraday_df):
    """Composite zone set from the FULL multi-day intraday_df (no date
    filtering -- composite means across all fetched days)."""
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
    """Latest 200-period EMA value from a 5-min closing-price series.
    Needs at least `period` candles to be meaningful -- with 18 days of
    composite 5-min history (~1,350 candles), there's comfortably
    enough, but returns None defensively if there ever isn't (e.g. a
    newly-listed stock without 18 days of history yet).

    Computed ONCE per Precompute from the full composite series, then
    held static through the day -- a 200-period EMA is inherently slow-
    moving (smoothing over many days of data), so its value barely
    shifts within a single session. Recomputing it live each cycle
    would need the full multi-day candle series again (expensive,
    defeats the whole point of the cheap batch-quotes design), so this
    is a deliberate, honest tradeoff: accurate as of this morning's
    Precompute, not continuously live through the day."""
    if len(closes) < period:
        return None
    return float(pd.Series(closes).ewm(span=period, adjust=False).mean().iloc[-1])


def compute_intraday_zones(today_only_df):
    """Intraday zone set from a candle df already scoped to a single
    session (see fetch_today_candles below)."""
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


def fetch_intraday_candles(instrument_key, token, unit="minutes", interval="5"):
    """Upstox's historical-candle endpoint (fetch_candles above) NEVER
    includes the still-open trading day -- it only has data up through
    yesterday's final close. Today's still-forming candles require this
    separate intraday endpoint. Without this, "today's" fetch silently
    returns only yesterday's last candle, which looks like a frozen/stale
    chart rather than an obvious error.

    Wrapped defensively: a single slow/failed API call for one symbol
    (timeout, connection error, 5xx) must not crash the whole app when
    this runs inside a 200+-symbol grid (Sectors/By RVOL/Wide Range/Zone
    Watch) -- returns an empty DataFrame instead, same as the existing
    "no candles yet" case downstream already handles gracefully."""
    url = f"https://api.upstox.com/v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        candles = resp.json().get("data", {}).get("candles", [])
    except requests.exceptions.RequestException:
        return pd.DataFrame()
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    return df


def fetch_today_candles_for(instrument_key, token, unit="minutes", interval="5"):
    """Generalized version of fetch_today_candles, parameterized by
    unit/interval -- today's session candles via the intraday endpoint,
    falling back to the historical endpoint's last available day (e.g.
    before market open, when the intraday endpoint may return nothing
    yet) so the Chart/zone/Dashboard functions always get something to
    work with.

    The fallback call is wrapped defensively (unlike fetch_candles'
    other call site inside run_precompute, which already has its own
    try/except + logging and should keep raising so failures there stay
    visible) -- this call site feeds live chart rendering for 200+
    symbols at once (Sectors/By RVOL/Wide Range/Zone Watch), where one
    symbol's request failing (e.g. a 400 from asking the historical
    endpoint for a still-open trading day, which it doesn't support)
    must not crash the whole page."""
    df = fetch_intraday_candles(instrument_key, token, unit, interval)
    if not df.empty:
        return df

    try:
        df = fetch_candles(instrument_key, token, unit, interval, lookback_days=1)
    except requests.exceptions.RequestException:
        return pd.DataFrame()
    if df.empty:
        return df
    latest = df["date"].max()
    return df[df["date"] == latest]


def fetch_today_candles(instrument_key, token):
    """5-min convenience wrapper around fetch_today_candles_for -- every
    OTHER call site in this app (Chart/Sectors/Zone Watch/etc.) wants the
    standard CANDLE_INTERVAL_MINUTES granularity, so they keep calling
    this unchanged. Only the Dashboard tab's interval switcher calls
    fetch_today_candles_for directly with a different interval."""
    return fetch_today_candles_for(instrument_key, token, "minutes", "5")


@st.cache_data(ttl=45, show_spinner=False)
def fetch_today_candles_cached(instrument_key, token):
    """Same as fetch_today_candles, but memoized for 45s. The all-sectors
    scroll view renders 200+ small charts at once -- without this, every
    script rerun (including auto-refresh ticks) would re-fetch candles
    for all 200+ symbols, which is slow and hammers the Upstox API far
    harder than necessary for a view that's mostly just being scrolled,
    not actively refreshed every few seconds."""
    return fetch_today_candles(instrument_key, token)


@st.cache_data(ttl=45, show_spinner=False)
def fetch_today_candles_interval_cached(instrument_key, token, unit, interval):
    """Same 45s memoization as fetch_today_candles_cached, but for an
    arbitrary unit/interval -- backs the Dashboard tab's 1m/5m/15m
    switcher so flipping intervals (or an auto-refresh tick) doesn't
    re-hit the API more than once every 45s per (symbol, interval)."""
    return fetch_today_candles_for(instrument_key, token, unit, interval)


def get_today_candles(symbol, instrument_key, token):
    """Tries the live WebSocket feed first (fno-websocket-feed's shared
    JSON file) -- instant, no API call, genuinely live. Falls back to the
    existing REST fetch if the feed isn't running, is stale, or doesn't
    have this symbol yet (e.g. it's an index key the feed can't name, or
    the listener only just started). This fallback is what keeps
    Streamlit Cloud working exactly as before -- the live feed file will
    simply never exist there, so every call just uses REST, unchanged."""
    live_df = get_live_candles(symbol, interval=str(CANDLE_INTERVAL_MINUTES))
    if live_df is not None and not live_df.empty:
        return live_df
    return fetch_today_candles_cached(instrument_key, token)


def get_today_candles_for_interval(symbol, instrument_key, token, unit, interval):
    """Same live-feed-first / REST-fallback preference as
    get_today_candles, but for an arbitrary unit/interval -- used by the
    Dashboard tab's interval switcher. The live WebSocket feed aggregates
    every interval in candle_aggregator.INTERVALS_SECONDS ("1"/"5"/"15")
    from the same tick stream at once, so all three are live-feed-first
    now, not just the CANDLE_INTERVAL_MINUTES default; anything else
    (a unit other than minutes, or an interval the feed doesn't build)
    goes straight to the interval-aware REST path."""
    if unit == "minutes" and interval in INTERVALS_SECONDS:
        live_df = get_live_candles(symbol, interval=interval)
        if live_df is not None and not live_df.empty:
            return live_df
    return fetch_today_candles_interval_cached(instrument_key, token, unit, interval)


def run_precompute(token, progress_callback=None):
    cache = {}
    all_symbols = [(s, "futures") for s in FUTURES_SYMBOLS] + [(s, "equity") for s in EQUITY_SYMBOLS]
    for i, (symbol, kind) in enumerate(all_symbols):
        try:
            key, lot_size = (resolve_equity_instrument_key(symbol, token) if kind == "equity"
                              else resolve_futures_instrument_key(symbol, token))
            if key is None:
                continue
            daily_df = fetch_candles(key, token, "days", "1", DAILY_LOOKBACK_DAYS)
            intraday_df = fetch_candles(key, token, "minutes", "5", COMPOSITE_LOOKBACK_DAYS)

            prev_close = float(daily_df["close"].iloc[-1]) if not daily_df.empty else None
            # Upstox's daily candle API can lag a day behind (may not
            # include today's candle even after market close), while
            # the intraday API doesn't have that lag -- fall back to
            # today's last intraday close so Precompute run this
            # evening reflects TODAY's actual close, not yesterday's.
            if not intraday_df.empty:
                _intraday_last_date = intraday_df["timestamp"].iloc[-1].date()
                _daily_last_date = daily_df["timestamp"].iloc[-1].date() if not daily_df.empty else None
                if _daily_last_date is None or _intraday_last_date > _daily_last_date:
                    prev_close = float(intraday_df["close"].iloc[-1])
            avg_daily_volume = (float(daily_df["volume"].tail(RVOL_BASELINE_DAYS).mean())
                                 if len(daily_df) >= RVOL_BASELINE_DAYS else None)
            composite_zones = compute_composite_zones(intraday_df)
            intraday_zones = compute_intraday_zones(intraday_df)  # seed with today's slice of what we already have
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
        except Exception as e:
            st.warning(f"{symbol}: precompute failed ({e}), skipping.")
        if progress_callback:
            progress_callback(i + 1, len(all_symbols), symbol)
        time.sleep(0.15)

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    return cache


def run_zone_refresh(cache, token, progress_callback=None):
    """The 'medium' refresh tier: re-fetches TODAY's candles per symbol
    and recomputes intraday_zones. Composite zones are left untouched
    (those only change at the next Precompute)."""
    symbols = list(cache.keys())
    for i, symbol in enumerate(symbols):
        try:
            key = cache[symbol]["instrument_key"]
            today_df = fetch_today_candles(key, token)
            cache[symbol]["intraday_zones"] = compute_intraday_zones(today_df)
            cache[symbol]["zones_updated_at"] = now_ist().strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            st.warning(f"{symbol}: zone refresh failed ({e}), keeping previous zones.")
        if progress_callback:
            progress_callback(i + 1, len(symbols), symbol)
        time.sleep(0.1)

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    return cache


def fetch_batch_quotes(instrument_keys, token):
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    params = {"instrument_key": ",".join(instrument_keys)}
    resp = requests.get(QUOTES_URL, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json().get("data", {})


def build_manual_trade_candidate(direction, symbol, val_comp, chart_df):
    """Same structure-based logic as the algo's entries -- stop is the
    nearest zone against the trade, target is the nearest zone with it
    -- just triggered by a manual Buy/Sell click on the Chart tab
    instead of an automatic level-cross + VWAP-cross detection.

    Only requires a STOP-side zone (support for a long, resistance for
    a short) -- a trade can't be opened without a defined risk control.
    The TARGET side is optional: if there's no zone on that side yet,
    target is left as None (open-ended -- exits only via stop-loss or
    end-of-day, same as the app already handles elsewhere), rather than
    blocking the trade entirely just because profit-taking isn't
    pinned to a level yet."""
    if chart_df is None or chart_df.empty:
        return None
    day_open = float(chart_df["open"].iloc[0])
    ltp = float(chart_df["close"].iloc[-1])
    typical = (chart_df["high"] + chart_df["low"] + chart_df["close"]) / 3.0
    cum_vol = chart_df["volume"].cumsum()
    vwap_series = (typical * chart_df["volume"]).cumsum() / cum_vol.replace(0, pd.NA)
    vwap = float(vwap_series.ffill().iloc[-1]) if cum_vol.iloc[-1] > 0 else None

    support, _, resistance, _ = nearest_zones(ltp, val_comp)
    if direction == "long":
        stop_zone, target_zone = support, resistance
    else:
        stop_zone, target_zone = resistance, support
    if stop_zone is None:
        return None

    risk = None
    if vwap is not None:
        risk = predict_break_probability(
            stop_zone, ltp=ltp, vwap=vwap, day_open=day_open,
            session_start_time=MARKET_OPEN_TIME, session_end_time=MARKET_CLOSE_TIME,
            now_time=now_ist().time(), is_intraday_validated=True,  # stop_zone comes from val_comp, so it's validated by definition
        )

    return {
        "symbol": symbol, "direction": direction, "entry_price": ltp,
        "stop_loss": stop_zone["price_mode"],
        "target": target_zone["price_mode"] if target_zone is not None else None,
        "ml_risk_pct": round(risk * 100, 1) if risk is not None else None,
        "zone_pct": _pct_from_label_safe(stop_zone["label"]),
        "source": "manual",
    }


def nearest_zones(ltp, validated_zones):
    """Splits validated zones into support-side (price_mode <= ltp) and
    resistance-side (price_mode > ltp), and returns whichever of each is
    CLOSEST to ltp, along with the % distance from ltp to that zone's
    near edge (price_high for support, price_low for resistance -- the
    edge price would actually touch first)."""
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


def build_wide_range_df(cache, price_lookup):
    """Ranks every stock by the GAP between its nearest validated support
    and nearest validated resistance -- i.e. how much room price actually
    has to move before hitting a wall in either direction. A stock with a
    tight gap is already boxed in; a wide gap means real room for a move
    to develop (breakout continuation or range play) without immediately
    running into the next level. Only includes symbols with BOTH a
    validated support and resistance currently identified -- a stock with
    open air on one side has an undefined "gap" (not comparable)."""
    rows = []
    for symbol, c in cache.items():
        ltp = price_lookup.get(symbol)
        if ltp is None:
            continue
        val_comp, _, _ = cross_validated_zones(
            c.get("composite_zones", []), c.get("intraday_zones", [])
        )
        support, _, resistance, _ = nearest_zones(ltp, val_comp)
        if support is None or resistance is None:
            continue
        gap_price = resistance["price_mode"] - support["price_mode"]
        if gap_price <= 0:
            continue  # shouldn't happen given nearest_zones' split logic, but guard anyway
        gap_pct = gap_price / ltp * 100
        rows.append({
            "Symbol": symbol, "LTP": ltp,
            "Support": support["price_mode"], "Resistance": resistance["price_mode"],
            "Gap": round(gap_price, 2), "Gap %": round(gap_pct, 2),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("Gap %", ascending=False).reset_index(drop=True)


def get_top_wide_range_symbols(cache, price_lookup, top_n=10):
    """Same ranking as build_wide_range_df, but returns just the top_n
    symbol names -- used to restrict the paper-trading entry universe
    to stocks with genuine room to move, instead of scanning all 220+
    stocks every cycle. Reuses zones already in cache and prices
    already fetched this cycle, so the ranking itself costs no extra
    API calls."""
    df = build_wide_range_df(cache, price_lookup)
    if df.empty:
        return []
    return df["Symbol"].head(top_n).tolist()


def crossed_zones(prev_ltp, ltp, validated_zones):
    """Detects a zone LEVEL actually being crossed between the previous
    and current scan tick -- no VWAP condition, no 'near' threshold,
    fires the instant price crosses through a validated zone's price_mode.

    Direction determines the label, and it works out cleanly with no
    extra classification needed:
      - price FALLS through a level (prev_ltp >= level > ltp): that level
        is now above current price, i.e. it's acting as resistance going
        forward -- this is a "Resistance breakdown" (bearish).
      - price RISES through a level (prev_ltp < level <= ltp): that level
        is now below current price, i.e. it's acting as support going
        forward -- this is a "Support reclaim" (bullish).
    This matches the same after-the-fact support/resistance labeling the
    chart itself uses (zone vs. current price), so an alert fired here
    lines up with what you'd see if you opened the Chart tab right after.
    """
    breakdowns = []  # price fell through a level -> now resistance overhead
    reclaims = []     # price rose through a level -> now support underneath
    if prev_ltp is None or ltp is None:
        return breakdowns, reclaims
    for z in validated_zones:
        level = z["price_mode"]
        if prev_ltp >= level > ltp:
            breakdowns.append(z)
        elif prev_ltp < level <= ltp:
            reclaims.append(z)
    return breakdowns, reclaims


def build_setup_display_df(rows, zone_kind):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if zone_kind == "support":
        df = df[["symbol", "ltp", "vwap", "zone_level", "zone_pct", "distance_pct",
                  "room_level", "room_pct"]]
        df.columns = ["Symbol", "LTP", "VWAP", "Support level", "Zone %", "Distance %",
                      "Next resistance", "Room to run %"]
        # best risk/reward (most room before hitting resistance) floats to top;
        # symbols with no resistance overhead at all show blank Room and sort last
        return df.sort_values("Room to run %", ascending=False, na_position="last").reset_index(drop=True)
    else:
        df = df[["symbol", "ltp", "vwap", "zone_level", "zone_pct", "distance_pct",
                  "room_level", "room_pct"]]
        df.columns = ["Symbol", "LTP", "VWAP", "Resistance level", "Zone %", "Distance %",
                      "Next support", "Room to fall %"]
        return df.sort_values("Room to fall %", ascending=False, na_position="last").reset_index(drop=True)


def build_level_cross_display_df(rows, kind):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if kind == "breakdown":
        df = df[["symbol", "ltp", "level", "zone_pct", "next_level", "next_pct"]]
        df.columns = ["Symbol", "LTP", "Broke below", "Zone %", "Next support", "Room to fall %"]
        return df.sort_values("Room to fall %", ascending=False, na_position="last").reset_index(drop=True)
    else:
        df = df[["symbol", "ltp", "level", "zone_pct", "next_level", "next_pct"]]
        df.columns = ["Symbol", "LTP", "Broke above", "Zone %", "Next resistance", "Room to run %"]
        return df.sort_values("Room to run %", ascending=False, na_position="last").reset_index(drop=True)


MIN_ROOM_PCT_FOR_TOP_BOXES = 1.0  # only surface level-breaks with >1% room to move


def filter_by_room(rows, min_room_pct):
    """Keeps a level-break row if it has enough room to the next zone to be
    worth acting on, OR if there's no zone at all on that side (None = open
    air, which is arguably the best case, not a bad one -- so it passes
    through rather than getting filtered out for 'missing' data)."""
    return [r for r in rows if r.get("next_pct") is None or r["next_pct"] > min_room_pct]


def load_paper_trades():
    if os.path.exists(PAPER_TRADE_LOG_PATH):
        with open(PAPER_TRADE_LOG_PATH, "r") as f:
            return json.load(f)
    return {"trades": []}


def save_paper_trades(log):
    with open(PAPER_TRADE_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def has_open_paper_trade(paper_log, symbol):
    """One open paper position per symbol at a time -- a simple, sane
    guardrail against piling into the same name repeatedly."""
    return any(t["symbol"] == symbol and t["status"] == "open" for t in paper_log["trades"])


def open_paper_trade(paper_log, candidate):
    """candidate: {symbol, direction, entry_price, stop_loss, target,
    ml_risk_pct, zone_pct, source}. source is "algo" (level-cross +
    VWAP-cross + ML filter, opened automatically) or "manual" (opened by
    clicking Buy/Sell on a chart). Skips if the position would round to
    0 shares at the fixed rupee size (e.g. a very expensive stock)."""
    qty = int(PAPER_TRADE_SIZE_RUPEES // candidate["entry_price"])
    if qty < 1:
        return False
    paper_log["trades"].append({
        "symbol": candidate["symbol"], "direction": candidate["direction"],
        "entry_price": candidate["entry_price"],
        "entry_time": now_ist().strftime("%Y-%m-%d %H:%M:%S"),
        "stop_loss": candidate["stop_loss"], "target": candidate["target"], "qty": qty,
        "ml_risk_pct": candidate["ml_risk_pct"], "zone_pct": candidate["zone_pct"],
        "source": candidate.get("source", "algo"),
        "status": "open", "exit_price": None, "exit_time": None,
        "exit_reason": None, "pnl": None,
    })
    return True


def place_real_market_order(instrument_key, quantity, transaction_type, token, tag="dashboard", product="I"):
    """Places a REAL MARKET order via Upstox's order-placement API --
    REAL MONEY, not a simulation. Same payload shape and endpoint as the
    proven-working place_real_order.py diagnostic script (product "I" =
    intraday/MIS by default, price 0 as required for MARKET orders).
    Only the Dashboard tab's explicit "Real" trade mode calls this --
    every other trade path in this app (algo entries, manual Buy/Sell
    elsewhere, tick_paper_trader.py) stays simulated via open_paper_trade
    above.

    product: "I" (intraday/MIS) for the F&O trade section's default.
    Some scrips reject intraday outright with UDAPI100500 "Intraday (I)
    orders are not allowed on this scrip" -- confirmed live for the
    NSE-commodity CRUDEOIL contract, and Upstox's own community forum's
    standard advice for this exact error, across scrips generally, is
    "place a delivery order instead" -- i.e. product="D". The commodity
    test section passes "D" for that reason; the F&O section keeps "I".

    Returns (success: bool, message: str, order_ids: list). Never raises
    on an HTTP-level rejection (bad margin, market closed, etc) -- that
    comes back as success=False with Upstox's own error message, same as
    a network failure would, so the caller can show either one the same
    way without a bare exception surfacing in the UI."""
    payload = {
        "quantity": quantity, "product": product, "validity": "DAY", "price": 0,
        "tag": tag, "instrument_token": instrument_key, "order_type": "MARKET",
        "transaction_type": transaction_type, "disclosed_quantity": 0,
        "trigger_price": 0, "is_amo": False,
    }
    headers = {
        "Content-Type": "application/json", "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    try:
        resp = requests.post(REAL_ORDER_URL, json=payload, headers=headers, timeout=20)
    except requests.exceptions.RequestException as e:
        return False, f"Request failed before reaching Upstox: {e}", []

    try:
        body = resp.json()
    except Exception:
        return False, f"Non-JSON response (HTTP {resp.status_code}): {resp.text[:300]}", []

    if resp.status_code >= 400:
        return False, f"Rejected by Upstox (HTTP {resp.status_code}): {body}", []

    order_ids = body.get("data", {}).get("order_ids", [])
    return True, f"Order placed. Order ID(s): {order_ids}", order_ids


def load_real_orders_log():
    if os.path.exists(REAL_ORDER_LOG_PATH):
        with open(REAL_ORDER_LOG_PATH, "r") as f:
            return json.load(f)
    return {"orders": []}


def log_real_order(symbol, transaction_type, quantity, order_ids, instrument_key):
    """Append-only local record of every real order this app has placed
    -- Upstox's own Orders console is the actual source of truth, this
    is just so you have a record inside the app too without needing to
    cross-reference the console after the fact. Never used to decide
    anything (no dedup/exit logic reads this), purely informational."""
    log = load_real_orders_log()
    log["orders"].append({
        "symbol": symbol, "transaction_type": transaction_type, "quantity": quantity,
        "order_ids": order_ids, "instrument_key": instrument_key,
        "time": now_ist().strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(REAL_ORDER_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def check_paper_trade_exits(paper_log, price_lookup, force_eod=False):
    """Closes any OPEN paper trade whose stop-loss or target has been
    hit at the current LTP, or (if force_eod) closes everything still
    open at end of day. Computes P&L with the correct sign convention
    for both long and short."""
    now_str = now_ist().strftime("%Y-%m-%d %H:%M:%S")
    for t in paper_log["trades"]:
        if t["status"] != "open":
            continue
        ltp = price_lookup.get(t["symbol"])
        if ltp is None:
            continue
        exit_reason = None
        if t["direction"] == "long":
            if ltp <= t["stop_loss"]:
                exit_reason = "stop_loss"
            elif t["target"] is not None and ltp >= t["target"]:
                exit_reason = "target"
        else:  # short
            if ltp >= t["stop_loss"]:
                exit_reason = "stop_loss"
            elif t["target"] is not None and ltp <= t["target"]:
                exit_reason = "target"
        if exit_reason is None and force_eod:
            exit_reason = "end_of_day"
        if exit_reason:
            t["status"] = "closed"
            t["exit_price"] = ltp
            t["exit_time"] = now_str
            t["exit_reason"] = exit_reason
            if t["direction"] == "long":
                t["pnl"] = round((ltp - t["entry_price"]) * t["qty"], 2)
            else:
                t["pnl"] = round((t["entry_price"] - ltp) * t["qty"], 2)
    return paper_log


def render_daemon_status():
    """Shows whether the standalone paper_trader_daemon.py (which runs
    independently of any browser being open) is actually alive and
    succeeding, right where trades are checked -- no SSH or log-
    grepping needed to notice it's stopped working (e.g. an expired
    token)."""
    if not os.path.exists(HEARTBEAT_PATH):
        st.info(
            "No daemon heartbeat found yet -- either paper_trader_daemon.py "
            "isn't running, or it hasn't completed its first loop yet."
        )
        return

    try:
        with open(HEARTBEAT_PATH, "r") as f:
            hb = json.load(f)
    except Exception:
        st.warning("Daemon heartbeat file exists but couldn't be read.")
        return

    last_loop = hb.get("last_loop_time")
    last_success = hb.get("last_successful_scan")
    last_error = hb.get("last_error")
    consecutive_errors = hb.get("consecutive_errors", 0)

    if last_loop:
        last_loop_dt = datetime.strptime(last_loop, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
        minutes_since_loop = (now_ist() - last_loop_dt).total_seconds() / 60
    else:
        minutes_since_loop = None

    # process alive if it looped recently -- a generous multiple of the
    # idle-check interval (2 min) covers both market-hours (60s) and
    # after-hours (120s) cadence without false-alarming
    process_alive = minutes_since_loop is not None and minutes_since_loop < 10

    col1, col2 = st.columns(2)
    with col1:
        if process_alive:
            st.success(f"Daemon process: **alive** (last loop {minutes_since_loop:.0f} min ago)")
        else:
            st.error(
                f"Daemon process: **appears STOPPED** "
                f"(last loop {minutes_since_loop:.0f} min ago)" if minutes_since_loop is not None
                else "Daemon process: **no loop data**"
            )
    with col2:
        if consecutive_errors == 0:
            st.success(f"Last successful scan: {last_success or 'never'}")
        else:
            st.error(f"**{consecutive_errors} consecutive failures.** Last error: {last_error}")


def build_journal_analytics(closed_trades):
    """Computes hourly P&L/win-rate, long-vs-short, hold-time-by-outcome,
    and repeat-offender patterns from whatever closed trades currently
    exist -- recomputed fresh every time this is called (every page
    load), so the Trade Journal tab always reflects the CURRENT state
    of the log, not a frozen snapshot from whenever it was built."""
    if not closed_trades:
        return None
    df = pd.DataFrame(closed_trades)
    df["Entry Time"] = pd.to_datetime(df["entry_time"])
    df["Exit Time"] = pd.to_datetime(df["exit_time"])
    df["Hold Minutes"] = (df["Exit Time"] - df["Entry Time"]).dt.total_seconds() / 60
    df["Win"] = df["pnl"] > 0
    df["Entry Hour"] = df["Entry Time"].dt.hour

    hourly = df.groupby("Entry Hour").agg(
        trades=("pnl", "count"), pnl=("pnl", "sum"), win_rate=("Win", "mean")
    ).reset_index().to_dict("records")

    direction = df.groupby("direction").agg(
        trades=("pnl", "count"), pnl=("pnl", "sum"), win_rate=("Win", "mean")
    ).reset_index().to_dict("records")

    hold_by_outcome = df.groupby("Win")["Hold Minutes"].median().to_dict()

    symbol_counts = df["symbol"].value_counts()
    repeats = symbol_counts[symbol_counts > 1]
    repeat_data = []
    for sym in repeats.index:
        sub = df[df["symbol"] == sym]
        repeat_data.append({
            "symbol": sym, "trades": len(sub),
            "wins": int(sub["Win"].sum()), "pnl": round(sub["pnl"].sum(), 2),
        })
    repeat_data.sort(key=lambda x: x["pnl"])

    return {
        "total_trades": len(df),
        "total_pnl": round(df["pnl"].sum(), 2),
        "win_rate": round(df["Win"].mean() * 100, 1),
        "avg_win": round(df[df["Win"]]["pnl"].mean(), 2) if df["Win"].any() else None,
        "avg_loss": round(df[~df["Win"]]["pnl"].mean(), 2) if (~df["Win"]).any() else None,
        "hourly": hourly,
        "direction": direction,
        "hold_winners_median": round(hold_by_outcome.get(True, 0), 1),
        "hold_losers_median": round(hold_by_outcome.get(False, 0), 1),
        "repeat_offenders": repeat_data,
    }


def render_trade_journal(paper_log):
    """Renders the Trade Journal tab -- a ledger-styled, always-current
    review of every closed paper trade, recomputed fresh from whatever
    is in paper_log right now."""
    st.markdown("""
        <style>
        .journal-headline { font-size: 52px; font-weight: 600; font-family: 'Georgia', serif;
            line-height: 1; margin-bottom: 4px; }
        .journal-sub { color: #6b6b7a; font-size: 15px; margin-bottom: 20px; }
        .journal-ledger-row { display: flex; justify-content: space-between; padding: 8px 0;
            border-bottom: 1px solid rgba(0,0,0,0.08); font-size: 14px; }
        </style>
    """, unsafe_allow_html=True)

    closed = [t for t in paper_log.get("trades", []) if t["status"] == "closed"]
    analytics = build_journal_analytics(closed)

    if analytics is None:
        st.info("No closed trades yet -- the journal fills in as trades close.")
        return

    pnl_color = "#1B5E3F" if analytics["total_pnl"] >= 0 else "#8B2635"
    sign = "+" if analytics["total_pnl"] >= 0 else "\u2212"
    st.markdown(
        f'<div class="journal-headline" style="color:{pnl_color}">{sign}\u20b9{abs(analytics["total_pnl"]):.0f}</div>'
        f'<div class="journal-sub">net across {analytics["total_trades"]} closed trades</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Win rate", f"{analytics['win_rate']:.0f}%")
    col2.metric("Avg win", f"\u20b9{analytics['avg_win']:.0f}" if analytics["avg_win"] else "\u2014")
    col3.metric("Avg loss", f"\u20b9{analytics['avg_loss']:.0f}" if analytics["avg_loss"] else "\u2014")
    col4.metric("Trades", analytics["total_trades"])

    st.markdown("**P&L by hour of entry**")
    hourly_df = pd.DataFrame(analytics["hourly"])
    hourly_df["Hour"] = hourly_df["Entry Hour"].apply(lambda h: f"{h}:00")
    hourly_df["Color"] = hourly_df["pnl"].apply(lambda v: "Gain" if v >= 0 else "Loss")
    hourly_fig = go.Figure()
    hourly_fig.add_trace(go.Bar(
        x=hourly_df["Hour"], y=hourly_df["pnl"],
        marker_color=hourly_df["pnl"].apply(lambda v: "#1B5E3F" if v >= 0 else "#8B2635"),
        text=hourly_df["win_rate"].apply(lambda w: f"{w*100:.0f}% WR"),
        textposition="outside",
    ))
    hourly_fig.update_layout(
        height=280, margin=dict(l=40, r=20, t=20, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="P&L (\u20b9)"), showlegend=False,
    )
    st.plotly_chart(hourly_fig, use_container_width=True)

    st.markdown("**Long vs. short**")
    dcol1, dcol2 = st.columns(2)
    for d in analytics["direction"]:
        target_col = dcol1 if d["direction"] == "long" else dcol2
        color = "#1B5E3F" if d["pnl"] >= 0 else "#8B2635"
        target_col.markdown(
            f"**{d['direction'].capitalize()}**<br>"
            f"<span style='color:{color}'>\u20b9{d['pnl']:.0f}</span> \u00b7 {d['win_rate']*100:.0f}% win rate "
            f"({d['trades']} trades)",
            unsafe_allow_html=True,
        )

    st.markdown("**Median hold time**")
    hcol1, hcol2 = st.columns(2)
    hcol1.markdown(f"Winners: **{analytics['hold_winners_median']:.0f} min**")
    hcol2.markdown(f"Losers: **{analytics['hold_losers_median']:.0f} min**")

    if analytics["repeat_offenders"]:
        st.markdown("**Repeat names** (traded more than once)")
        for r in analytics["repeat_offenders"]:
            color = "#1B5E3F" if r["pnl"] >= 0 else "#8B2635"
            st.markdown(
                f'<div class="journal-ledger-row"><span>{r["symbol"]} '
                f'<span style="color:#6b6b7a">({r["trades"]} trades, {r["wins"]} wins)</span></span>'
                f'<span style="color:{color}">\u20b9{r["pnl"]:.0f}</span></div>',
                unsafe_allow_html=True,
            )


def render_dashboard_hero(paper_log):
    """A condensed, always-visible summary banner shown ABOVE the tab
    bar -- the first thing you see when opening the app, before
    clicking into anything. Same live analytics as the Trade Journal
    tab (build_journal_analytics), just a shorter version: headline
    P&L, key stats, and the hourly pattern at a glance. The full
    breakdown (long/short, hold times, repeat offenders) stays in the
    Trade Journal tab for anyone who wants to dig deeper."""
    st.markdown("""
        <style>
        .hero-headline { font-size: 44px; font-weight: 600; font-family: 'Georgia', serif;
            line-height: 1; margin-bottom: 2px; }
        .hero-sub { color: #6b6b7a; font-size: 14px; margin-bottom: 6px; }
        </style>
    """, unsafe_allow_html=True)

    closed = [t for t in paper_log.get("trades", []) if t["status"] == "closed"]
    open_count = sum(1 for t in paper_log.get("trades", []) if t["status"] == "open")
    analytics = build_journal_analytics(closed)

    with st.container(border=True):
        if analytics is None:
            st.markdown("#### Today's session")
            st.caption(f"No closed trades yet -- {open_count} open position(s) currently being tracked.")
            return

        hero_col, stats_col = st.columns([1, 2])
        with hero_col:
            pnl_color = "#1B5E3F" if analytics["total_pnl"] >= 0 else "#8B2635"
            sign = "+" if analytics["total_pnl"] >= 0 else "\u2212"
            st.markdown(
                f'<div class="hero-headline" style="color:{pnl_color}">{sign}\u20b9{abs(analytics["total_pnl"]):.0f}</div>'
                f'<div class="hero-sub">{analytics["total_trades"]} closed \u00b7 {open_count} open</div>',
                unsafe_allow_html=True,
            )
        with stats_col:
            c1, c2, c3 = st.columns(3)
            c1.metric("Win rate", f"{analytics['win_rate']:.0f}%")
            c2.metric("Avg win", f"\u20b9{analytics['avg_win']:.0f}" if analytics["avg_win"] else "\u2014")
            c3.metric("Avg loss", f"\u20b9{analytics['avg_loss']:.0f}" if analytics["avg_loss"] else "\u2014")

        hourly_df = pd.DataFrame(analytics["hourly"])
        hourly_df["Hour"] = hourly_df["Entry Hour"].apply(lambda h: f"{h}:00")
        hero_fig = go.Figure()
        hero_fig.add_trace(go.Bar(
            x=hourly_df["Hour"], y=hourly_df["pnl"],
            marker_color=hourly_df["pnl"].apply(lambda v: "#1B5E3F" if v >= 0 else "#8B2635"),
        ))
        hero_fig.update_layout(
            height=140, margin=dict(l=30, r=10, t=10, b=25),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, yaxis=dict(visible=False),
            xaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(hero_fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("P&L by hour \u00b7 full breakdown in the Trade Journal tab below")


def build_paper_trades_df(trades, status_filter):
    rows = [t for t in trades if t["status"] == status_filter]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "source" not in df.columns:
        df["source"] = "algo"
    df["source"] = df["source"].fillna("algo")
    if status_filter == "open":
        df = df[["symbol", "direction", "entry_price", "stop_loss", "target",
                  "qty", "ml_risk_pct", "zone_pct", "source", "entry_time"]]
        df.columns = ["Symbol", "Direction", "Entry", "Stop", "Target",
                      "Qty", "ML Risk %", "Zone %", "Source", "Entry Time"]
    else:
        df = df[["symbol", "direction", "entry_price", "exit_price", "qty",
                  "pnl", "exit_reason", "source", "entry_time", "exit_time"]]
        df.columns = ["Symbol", "Direction", "Entry", "Exit", "Qty",
                      "P&L (\u20b9)", "Exit Reason", "Source", "Entry Time", "Exit Time"]
        df = df.sort_values("Exit Time", ascending=False).reset_index(drop=True)
    return df


def run_live_scan(cache, token):
    """Fast tier: one batch quote call for LTP/VWAP, signal recomputed
    against whichever zones are currently cached (may be a few minutes
    stale if 'Refresh Zones' hasn't been run recently). Also computes RVOL
    (today's volume so far / prior-N-day average full-day volume) and
    ranks the top TOP_N_RVOL symbols -- only those are eligible to alert
    (see update_alert_log), since unusual volume is the conviction filter
    that keeps alerts to a handful of genuinely active names instead of
    every symbol that happens to tick across VWAP."""
    symbols = list(cache.keys())
    instrument_keys = [cache[s]["instrument_key"] for s in symbols]
    key_to_symbol = {cache[s]["instrument_key"]: s for s in symbols}
    try:
        quotes = fetch_batch_quotes(instrument_keys, token)
    except requests.exceptions.RequestException as e:
        st.warning(f"Live scan skipped this cycle (network error: {e}).")
        return (pd.DataFrame(), {}, set(), [], [], [], [], [], [], [])

    rows = []
    signals = {}
    bottom_setups = []       # near support + just crossed above VWAP
    top_setups = []          # near resistance + just closed below VWAP
    resistance_breakdowns = []  # pure level cross: price fell through a validated zone
    support_reclaims = []       # pure level cross: price rose through a validated zone
    paper_trade_candidates = []  # SIMULATED-only: level-cross + same-tick VWAP-cross + low ML risk
    support_watch = []       # EVERY symbol near a support edge, every cycle (not edge-triggered
                              # like bottom_setups above) -- feeds the persistent Zone Watch tab.
                              # "crossed" tags whether the same VWAP+zone confirmation ALSO fired
                              # this cycle, so Zone Watch can show it as confirmed vs still-watching.
    resistance_watch = []    # same idea, resistance side

    # Restrict paper-trading entries to the top Wide Range stocks --
    # genuine room to move, instead of scanning all 220+ stocks every
    # cycle. Ranking uses zones already cached and prices already
    # about to be parsed below, so this costs no extra API calls; the
    # quick first pass just extracts LTPs from the same quotes response
    # already fetched above.
    _price_lookup_for_ranking = {}
    for _q in quotes.values():
        _sym = key_to_symbol.get(_q.get("instrument_token"))
        if _sym and _q.get("last_price") is not None:
            _price_lookup_for_ranking[_sym] = _q["last_price"]
    top_wide_range_symbols = set(get_top_wide_range_symbols(
        cache, _price_lookup_for_ranking, top_n=PAPER_TRADE_UNIVERSE_TOP_N
    ))

    for quote_key, q in quotes.items():
        instrument_key = q.get("instrument_token")
        symbol = key_to_symbol.get(instrument_key)
        if not symbol:
            continue
        c = cache[symbol]
        ltp = q.get("last_price")
        vwap = q.get("average_price")
        today_volume = q.get("volume")
        prev_close = c.get("prev_close")
        avg_daily_volume = c.get("avg_daily_volume")

        change_pct = (round((ltp - prev_close) / prev_close * 100, 2)
                      if ltp is not None and prev_close else None)
        rvol_pct = (round(today_volume / avg_daily_volume * 100, 1)
                    if today_volume is not None and avg_daily_volume else None)
        signal = compute_zone_signal(
            ltp, vwap, c.get("composite_zones", []), c.get("intraday_zones", []),
            min_distance_pct=MIN_SIGNAL_DISTANCE_PCT,
            min_vwap_distance_pct=MIN_VWAP_DISTANCE_PCT,
        )
        signals[symbol] = {"signal": signal, "ltp": ltp, "vwap": vwap, "rvol_pct": rvol_pct}
        cache[symbol]["last_signal"] = signal

        # --- Setup detection: near support/resistance + VWAP cross ---
        # "Just crossed" is edge-triggered off the PREVIOUS scan's
        # above/below state, persisted in the cache (same pattern as
        # update_alert_log's edge-triggering below) so it survives
        # across reruns instead of re-firing on every refresh.
        if ltp is not None:
            val_comp, _, _ = cross_validated_zones(
                c.get("composite_zones", []), c.get("intraday_zones", [])
            )
            support, support_dist, resistance, resistance_dist = nearest_zones(ltp, val_comp)

            # defaults so downstream code can safely check these even
            # when vwap is None (crossed_up/crossed_down never set below)
            crossed_up = False
            crossed_down = False

            if vwap is not None:
                vwap_above_now = ltp > vwap
                prev_vwap_above = c.get("prev_vwap_above")
                crossed_up = prev_vwap_above is False and vwap_above_now
                crossed_down = prev_vwap_above is True and not vwap_above_now
                cache[symbol]["prev_vwap_above"] = vwap_above_now

                if support is not None and support_dist is not None and support_dist <= NEAR_ZONE_PCT and crossed_up:
                    bottom_setups.append({
                        "symbol": symbol, "ltp": ltp, "vwap": round(vwap, 2),
                        "zone_level": support["price_mode"],
                        "zone_pct": _pct_from_label_safe(support["label"]),
                        "distance_pct": round(support_dist, 2),
                        # "room to run": distance from LTP up to the next resistance
                        # overhead. None means no validated resistance zone was found
                        # above current price at all -- i.e. open air, arguably the
                        # BEST case for a bounce, not a bad one, so it's not filtered
                        # out, just shown blank and sorted to the bottom by default.
                        "room_level": resistance["price_mode"] if resistance is not None else None,
                        "room_pct": round(resistance_dist, 2) if resistance_dist is not None else None,
                    })

                if resistance is not None and resistance_dist is not None and resistance_dist <= NEAR_ZONE_PCT and crossed_down:
                    top_setups.append({
                        "symbol": symbol, "ltp": ltp, "vwap": round(vwap, 2),
                        "zone_level": resistance["price_mode"],
                        "zone_pct": _pct_from_label_safe(resistance["label"]),
                        "distance_pct": round(resistance_dist, 2),
                        # "room to fall": distance from LTP down to the next support
                        # floor. None means no validated support zone found below --
                        # i.e. open air on the downside if this rejection plays out.
                        "room_level": support["price_mode"] if support is not None else None,
                        "room_pct": round(support_dist, 2) if support_dist is not None else None,
                    })

            # --- Zone Watch: EVERY near-zone symbol, every cycle -- unlike
            # bottom_setups/top_setups above, this isn't edge-triggered on
            # the VWAP cross, so a symbol stays visible on the watch list
            # the whole time it's sitting near a zone, not just the one
            # cycle it happens to cross. crossed_up/crossed_down default to
            # False when vwap is None (set above), so this is safe to run
            # unconditionally regardless of whether vwap was available. ---
            _zw_day_open = (q.get("ohlc") or {}).get("open") or prev_close

            # --- CVD quality score inputs: recent order-flow imbalance is
            # the same regardless of which side (support/resistance) is
            # being scored, so it's fetched/computed ONCE per symbol here
            # (lazily -- only for symbols that actually have a near-zone
            # watch entry below, same "cheap by construction" design as
            # the paper-trade CVD fetch above) rather than twice. Uses
            # get_today_candles, which prefers the free live-feed snapshot
            # and only falls back to a 45s-cached REST call -- safe to
            # call for every near-zone symbol each cycle. ---
            _zw_near_support = support is not None and support_dist is not None and support_dist <= NEAR_ZONE_PCT
            _zw_near_resistance = resistance is not None and resistance_dist is not None and resistance_dist <= NEAR_ZONE_PCT
            _zw_flow_pct = None
            if _zw_near_support or _zw_near_resistance:
                _zw_candles = get_today_candles(symbol, c["instrument_key"], token)
                if not _zw_candles.empty:
                    _zw_flow_pct = compute_recent_order_flow_imbalance_pct(
                        _zw_candles, lookback=CVD_QUALITY_LOOKBACK_CANDLES
                    )

            if _zw_near_support:
                _sup_risk_pct = _zone_break_risk_pct(support, ltp, vwap, _zw_day_open)
                # room = distance from LTP up to the next validated zone
                # (resistance) -- the reward side of this BUY candidate's R:R.
                _sup_quality = compute_cvd_zone_signal(
                    ltp, vwap, side="support", distance_pct=support_dist,
                    room_pct=resistance_dist, order_flow_imbalance_pct=_zw_flow_pct,
                    rvol_pct=rvol_pct, ml_break_risk_pct=_sup_risk_pct,
                    min_vwap_distance_pct=MIN_VWAP_DISTANCE_PCT,
                )
                support_watch.append({
                    "symbol": symbol, "ltp": ltp,
                    "vwap": round(vwap, 2) if vwap is not None else None,
                    "zone_level": support["price_mode"],
                    "zone_pct": _pct_from_label_safe(support["label"]),
                    "distance_pct": round(support_dist, 2),
                    "crossed": bool(crossed_up),
                    "is_premium": _is_premium_zone(support, ltp, vwap, _zw_day_open, risk_pct=_sup_risk_pct),
                    "quality_score": _sup_quality["score"],
                    "quality_grade": _sup_quality["grade"],
                    "quality_components": _sup_quality["components"],
                })
            if _zw_near_resistance:
                _res_risk_pct = _zone_break_risk_pct(resistance, ltp, vwap, _zw_day_open)
                # room = distance from LTP down to the next validated zone
                # (support) -- the reward side of this SELL candidate's R:R.
                _res_quality = compute_cvd_zone_signal(
                    ltp, vwap, side="resistance", distance_pct=resistance_dist,
                    room_pct=support_dist, order_flow_imbalance_pct=_zw_flow_pct,
                    rvol_pct=rvol_pct, ml_break_risk_pct=_res_risk_pct,
                    min_vwap_distance_pct=MIN_VWAP_DISTANCE_PCT,
                )
                resistance_watch.append({
                    "symbol": symbol, "ltp": ltp,
                    "vwap": round(vwap, 2) if vwap is not None else None,
                    "zone_level": resistance["price_mode"],
                    "zone_pct": _pct_from_label_safe(resistance["label"]),
                    "distance_pct": round(resistance_dist, 2),
                    "crossed": bool(crossed_down),
                    "is_premium": _is_premium_zone(resistance, ltp, vwap, _zw_day_open, risk_pct=_res_risk_pct),
                    "quality_score": _res_quality["score"],
                    "quality_grade": _res_quality["grade"],
                    "quality_components": _res_quality["components"],
                })

            # --- Pure level-cross detection: no VWAP condition, no "near"
            # threshold. Fires the instant LTP actually crosses a validated
            # zone's price_mode, which catches moves as they start (e.g. a
            # breakdown at the open) rather than waiting for a VWAP
            # confirmation that might come minutes or hours later. ---
            prev_ltp = c.get("prev_ltp")
            level_breakdowns, level_reclaims = crossed_zones(prev_ltp, ltp, val_comp)
            cache[symbol]["prev_ltp"] = ltp

            # 200-EMA CROSSING filter (5-min, computed at Precompute from
            # the 18-day composite series -- see compute_ema_200 for why
            # this is a once-a-day snapshot, not continuously live).
            # This checks for an actual CROSSING EVENT on this exact
            # tick, not just "currently on the right side" -- price must
            # cross UP through the EMA on the same tick as a bullish
            # level-cross to confirm it, and cross DOWN on the same tick
            # for a bearish one. A stock that's simply been sitting
            # above/below the EMA for a while (no fresh cross right now)
            # does NOT count, matching the same "all conditions align on
            # THIS tick" discipline already used for the VWAP-cross.
            # Can't detect a crossing at all without a previous tick to
            # compare against, or without a computed EMA -- both cases
            # filter out entirely rather than assuming a pass.
            ema_200 = c.get("ema_200")
            if ema_200 is None or prev_ltp is None:
                level_breakdowns, level_reclaims = [], []
            else:
                ema_crossed_up = prev_ltp <= ema_200 and ltp > ema_200
                ema_crossed_down = prev_ltp >= ema_200 and ltp < ema_200
                if not ema_crossed_up:
                    level_reclaims = []
                if not ema_crossed_down:
                    level_breakdowns = []

            for z in level_breakdowns:
                # after breaking down through this level, the nearest
                # zone still below current price is the next support --
                # i.e. the next likely target/floor if the move continues.
                next_support, next_support_dist, _, _ = nearest_zones(ltp, val_comp)
                resistance_breakdowns.append({
                    "symbol": symbol, "ltp": ltp,
                    "level": z["price_mode"],
                    "zone_pct": _pct_from_label_safe(z["label"]),
                    "next_level": next_support["price_mode"] if next_support is not None else None,
                    "next_pct": round(next_support_dist, 2) if next_support_dist is not None else None,
                })

            for z in level_reclaims:
                # after breaking up through this level, the nearest zone
                # still above current price is the next resistance -- the
                # next hurdle if the move continues.
                _, _, next_resistance, next_resistance_dist = nearest_zones(ltp, val_comp)
                support_reclaims.append({
                    "symbol": symbol, "ltp": ltp,
                    "level": z["price_mode"],
                    "zone_pct": _pct_from_label_safe(z["label"]),
                    "next_level": next_resistance["price_mode"] if next_resistance is not None else None,
                    "next_pct": round(next_resistance_dist, 2) if next_resistance_dist is not None else None,
                })

            # --- Paper-trading candidate detection (SIMULATED only) ---
            # Deliberately stricter than the independent bottom_setups/
            # top_setups above: requires the level-cross AND the VWAP-
            # cross to fire on this SAME tick (not just both true at some
            # point), AND the just-crossed level (now in its FLIPPED
            # role -- a broken support becomes resistance, a broken
            # resistance becomes support) must show LOW ML break-risk,
            # i.e. the model thinks this level will actually hold if
            # retested -- real conviction behind the move, not a fakeout.
            # No candidate at all if there's no next zone to use as a
            # structure-based target (open air = no defined exit plan).
            # Restricted to the top Wide Range stocks (genuine room to
            # move) AND requires cumulative volume delta to agree with
            # the trade direction -- net buying pressure for a long, net
            # selling for a short. CVD needs actual candle data (not
            # available from the quotes response), so it's only fetched
            # here, lazily, for symbols that already passed every other
            # filter -- keeps this restricted-universe design cheap.
            day_open = (q.get("ohlc") or {}).get("open") or prev_close
            if day_open is not None and vwap is not None and symbol in top_wide_range_symbols:
                latest_cvd = None
                if level_reclaims or level_breakdowns:
                    cvd_candles = get_today_candles(symbol, c["instrument_key"], token)
                    if not cvd_candles.empty:
                        latest_cvd = compute_cumulative_volume_delta(cvd_candles).iloc[-1]

                for z in level_reclaims:  # bullish: broken level now acts as support
                    if not crossed_up:
                        continue
                    if latest_cvd is None or latest_cvd <= 0:
                        continue  # need net BUYING pressure to confirm a long
                    risk = predict_break_probability(
                        z, ltp=ltp, vwap=vwap, day_open=day_open,
                        session_start_time=MARKET_OPEN_TIME, session_end_time=MARKET_CLOSE_TIME,
                        now_time=now_ist().time(), is_intraday_validated=True,
                    )
                    if risk is None or risk * 100 >= PAPER_TRADE_ML_RISK_THRESHOLD:
                        continue
                    _, _, next_resistance, _ = nearest_zones(ltp, val_comp)
                    if next_resistance is None:
                        continue
                    paper_trade_candidates.append({
                        "symbol": symbol, "direction": "long", "entry_price": ltp,
                        "stop_loss": z["price_mode"], "target": next_resistance["price_mode"],
                        "ml_risk_pct": round(risk * 100, 1),
                        "zone_pct": _pct_from_label_safe(z["label"]),
                    })

                for z in level_breakdowns:  # bearish: broken level now acts as resistance
                    if not crossed_down:
                        continue
                    if latest_cvd is None or latest_cvd >= 0:
                        continue  # need net SELLING pressure to confirm a short
                    risk = predict_break_probability(
                        z, ltp=ltp, vwap=vwap, day_open=day_open,
                        session_start_time=MARKET_OPEN_TIME, session_end_time=MARKET_CLOSE_TIME,
                        now_time=now_ist().time(), is_intraday_validated=True,
                    )
                    if risk is None or risk * 100 >= PAPER_TRADE_ML_RISK_THRESHOLD:
                        continue
                    next_support, _, _, _ = nearest_zones(ltp, val_comp)
                    if next_support is None:
                        continue
                    paper_trade_candidates.append({
                        "symbol": symbol, "direction": "short", "entry_price": ltp,
                        "stop_loss": z["price_mode"], "target": next_support["price_mode"],
                        "ml_risk_pct": round(risk * 100, 1),
                        "zone_pct": _pct_from_label_safe(z["label"]),
                    })

        rows.append({
            "Symbol": symbol, "PrevClose": prev_close, "LTP": ltp,
            "Change%": change_pct, "VWAP": vwap, "RVOL%": rvol_pct, "Signal": signal,
        })

    ranked = sorted(
        [(s, d["rvol_pct"]) for s, d in signals.items() if d["rvol_pct"] is not None],
        key=lambda x: x[1], reverse=True,
    )
    top_n_symbols = set(s for s, _ in ranked[:TOP_N_RVOL])

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Top5RVOL"] = df["Symbol"].isin(top_n_symbols)
        df = df.sort_values("RVOL%", ascending=False, na_position="last").reset_index(drop=True)
        df.insert(0, "S.No", range(1, len(df) + 1))
    return (df, signals, top_n_symbols, bottom_setups, top_setups, resistance_breakdowns,
            support_reclaims, paper_trade_candidates, support_watch, resistance_watch)


def _pct_from_label_safe(label):
    m = re.search(r"[\d.]+", str(label))
    return float(m.group()) if m else 0.0


def _zone_break_risk_pct(zone, ltp, vwap, day_open):
    """Shared helper: ML-predicted break probability for `zone` right now,
    as a 0-100 percent (None if inputs are missing). Factored out of
    _is_premium_zone so the same call/risk value can also feed
    compute_cvd_zone_signal's composite score below, instead of predicting
    twice for the same zone on the same tick."""
    if ltp is None or vwap is None or day_open is None:
        return None
    risk = predict_break_probability(
        zone, ltp=ltp, vwap=vwap, day_open=day_open,
        session_start_time=MARKET_OPEN_TIME, session_end_time=MARKET_CLOSE_TIME,
        now_time=now_ist().time(), is_intraday_validated=True,
    )
    return None if risk is None else risk * 100


def _is_premium_zone(zone, ltp, vwap, day_open, risk_pct=None):
    """A 'premium' zone combines real trading volume behind it (high
    pct_of_session -- not a thin/noisy cluster) with the ML model being
    confident it'll hold rather than break. Computed fresh here rather
    than cached, since the ML risk component genuinely depends on
    current price/VWAP/time-of-day, not just the zone's own static
    properties -- a zone can be premium at one moment and not another
    as the session progresses. Returns False (not None/unknown) if
    inputs are missing, since an unproven zone shouldn't default to
    looking premium.

    risk_pct: pass an already-computed _zone_break_risk_pct() value to
    avoid a redundant model call when the caller needs both this bool
    AND the raw risk number (e.g. for compute_cvd_zone_signal) -- if
    omitted, this computes it itself."""
    zone_pct = _pct_from_label_safe(zone.get("label", ""))
    if zone_pct < PREMIUM_MIN_ZONE_PCT:
        return False
    if risk_pct is None:
        risk_pct = _zone_break_risk_pct(zone, ltp, vwap, day_open)
    if risk_pct is None:
        return False
    return risk_pct < PREMIUM_MAX_ML_RISK_PCT


def load_alert_log():
    if os.path.exists(ALERT_LOG_PATH):
        with open(ALERT_LOG_PATH, "r") as f:
            return json.load(f)
    return {"alerts": [], "last_eligible_symbols": []}


def save_alert_log(log):
    with open(ALERT_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def update_alert_log(alert_log, signals, eligible_symbols):
    """Edge-triggered: only logs a new entry the moment a symbol's signal
    changes to a fresh BUY/SELL state, not on every refresh it stays
    active. Gated to market hours -- an after-hours refresh pulls Upstox's
    frozen post-close LTP/VWAP, which must not get logged as a live
    signal (same bug already fixed in hvn-lvn-scanner's paper trader).

    Also gated to eligible_symbols (the current top TOP_N_RVOL by RVOL).
    "Newly entered" (just entered the top N this cycle) is tracked via
    alert_log["last_eligible_symbols"], persisted to disk -- NOT via
    st.session_state. Session state is per-browser-session, so a page
    reload or a Streamlit Cloud reconnect resets it to empty, making
    every currently-eligible symbol look "newly entered" again and
    re-logging duplicates seconds after the original (this exact bug
    was seen live: 5 symbols logged twice, 4 seconds apart). Persisting
    to the same file the dedup check already reads from survives
    reconnects correctly."""
    now = now_ist()
    prev_eligible = set(alert_log.get("last_eligible_symbols", []))
    newly_entered = eligible_symbols - prev_eligible
    alert_log["last_eligible_symbols"] = list(eligible_symbols)

    market_is_open = MARKET_OPEN_TIME <= now.time() < MARKET_CLOSE_TIME
    if not market_is_open:
        return alert_log

    last_signal = {a["symbol"]: a["signal"] for a in alert_log["alerts"] if a.get("is_latest")}
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    for symbol, data in signals.items():
        if symbol not in eligible_symbols:
            continue
        sig = data["signal"]
        if sig not in ("BUY", "SELL"):
            continue
        is_fresh_entry = symbol in newly_entered
        if not is_fresh_entry and last_signal.get(symbol) == sig:
            continue
        for a in alert_log["alerts"]:
            if a["symbol"] == symbol:
                a["is_latest"] = False
        alert_log["alerts"].append({
            "symbol": symbol, "signal": sig, "ltp": data["ltp"], "vwap": data["vwap"],
            "rvol_pct": data.get("rvol_pct"), "time": now_str, "is_latest": True,
        })
    return alert_log


def build_alert_display_df(alert_log, price_lookup=None):
    """price_lookup: {symbol: current LTP}, from the latest Scanner scan --
    used to mark-to-market each alert's PnL% against its entry price
    (the LTP captured at the moment the alert fired)."""
    if not alert_log["alerts"]:
        return pd.DataFrame()
    price_lookup = price_lookup or {}
    df = pd.DataFrame(alert_log["alerts"])
    if "rvol_pct" not in df.columns:
        df["rvol_pct"] = None

    def _current_price(row):
        return price_lookup.get(row["symbol"])

    def _pnl(row):
        cur = row["current_ltp"]
        entry = row["ltp"]
        if cur is None or entry is None:
            return None
        if row["signal"] == "BUY":
            return round((cur - entry) / entry * 100, 2)
        elif row["signal"] == "SELL":
            return round((entry - cur) / entry * 100, 2)
        return None

    df["current_ltp"] = df.apply(_current_price, axis=1)
    df["pnl_pct"] = df.apply(_pnl, axis=1)
    df = df[["symbol", "signal", "ltp", "current_ltp", "pnl_pct", "vwap", "rvol_pct", "time"]]
    df.columns = ["Symbol", "Signal", "EntryPrice", "LTP", "PnL%", "VWAP", "RVOL%", "Time"]
    df = df.sort_values("Time", ascending=False).reset_index(drop=True)
    df.insert(0, "S.No", range(1, len(df) + 1))
    return df


def build_zones_display_df(zones):
    if not zones:
        return pd.DataFrame()
    df = pd.DataFrame(zones)
    df = df[["price_mode", "label", "price_low", "price_high"]]
    df.columns = ["Level", "Zone %", "Range Low", "Range High"]
    return df.sort_values("Level", ascending=False).reset_index(drop=True)


def compute_zone_ml_risks(composite_zones, validated_zones, chart_df):
    """Core computation shared by build_ml_risk_df (the table) and
    build_ml_risk_lookup (the on-chart label values) -- returns a list of
    {price_mode, kind, is_validated, risk_pct} dicts, or [] if there's
    not enough data (empty chart_df, no zones)."""
    if not composite_zones or chart_df is None or chart_df.empty:
        return []

    day_open = float(chart_df["open"].iloc[0])
    ltp = float(chart_df["close"].iloc[-1])
    typical = (chart_df["high"] + chart_df["low"] + chart_df["close"]) / 3.0
    cum_vol = chart_df["volume"].cumsum()
    vwap_series = (typical * chart_df["volume"]).cumsum() / cum_vol.replace(0, pd.NA)
    vwap = float(vwap_series.ffill().iloc[-1]) if cum_vol.iloc[-1] > 0 else None

    now_time = now_ist().time()
    validated_keys = {round(z["price_mode"], 2) for z in validated_zones}

    results = []
    for z in composite_zones:
        is_validated = round(z["price_mode"], 2) in validated_keys
        prob = predict_break_probability(
            z, ltp=ltp, vwap=vwap, day_open=day_open,
            session_start_time=MARKET_OPEN_TIME, session_end_time=MARKET_CLOSE_TIME,
            now_time=now_time, is_intraday_validated=is_validated,
        )
        if prob is None:
            continue
        kind = "Support" if z["price_mode"] <= ltp else "Resistance"
        results.append({
            "price_mode": z["price_mode"], "kind": kind,
            "zone_pct": z["label"], "is_validated": is_validated,
            "risk_pct": round(prob * 100, 1),
        })
    return results


def build_ml_risk_df(composite_zones, validated_zones, chart_df):
    """ML break-risk probability for every composite zone, using today's
    already-fetched candle data (chart_df) for day_open/ltp/vwap -- no
    extra fetch needed. See ml_predict.py for the model's training
    details and honest limitations."""
    results = compute_zone_ml_risks(composite_zones, validated_zones, chart_df)
    if not results:
        return pd.DataFrame()
    rows = [{
        "Level": r["price_mode"], "Kind": r["kind"], "Zone %": r["zone_pct"],
        "Confirmed": "Yes" if r["is_validated"] else "No", "ML Break Risk %": r["risk_pct"],
    } for r in results]
    return pd.DataFrame(rows).sort_values("ML Break Risk %", ascending=False).reset_index(drop=True)


def build_ml_risk_lookup(composite_zones, validated_zones, chart_df):
    """Same computation as build_ml_risk_df, but returns a
    {rounded price_mode: risk_pct} dict for plot_candles_with_zones'
    ml_risk_lookup parameter -- shows the risk directly on the chart's
    zone label instead of (or alongside) the separate table."""
    results = compute_zone_ml_risks(composite_zones, validated_zones, chart_df)
    return {round(r["price_mode"], 2): r["risk_pct"] for r in results}


def get_session_x_range(df):
    """Pins the x-axis to the FULL known session hours (09:15-15:30 IST)
    for today's date, regardless of how many candles have actually formed
    so far. Without this, early in the session -- right after market
    open, when only a handful of 5-min candles exist -- Plotly's
    autorange fits tightly to just those few candles, stretching them to
    fill the entire chart width."""
    if df is None or df.empty:
        return None
    session_date = df["timestamp"].iloc[0].date()
    start = datetime.combine(session_date, MARKET_OPEN_TIME)
    end = datetime.combine(session_date, MARKET_CLOSE_TIME)
    if df["timestamp"].iloc[0].tzinfo is not None:
        start = start.replace(tzinfo=IST)
        end = end.replace(tzinfo=IST)
    return (start, end)


# ---------------- UI (four tabs: Scanner, Key Levels, Chart, Alerts) ----------------
st.set_page_config(page_title="Dr Yarapu Reddy Levels", layout="wide")

# --- Password gate: blocks the whole app until the right password is
# entered, since Streamlit Cloud's own "restrict viewers" setting is
# capped at one app per account and can't be relied on here. The
# password lives in Streamlit secrets (APP_PASSWORD), never in this
# file. Once entered correctly, session_state remembers it for the
# rest of this browser session -- no need to re-enter on every rerun. ---
def _check_password():
    st.write("DEBUG APP_PASSWORD =", repr(st.secrets.get("APP_PASSWORD")))
    st.write("DEBUG session_state _pw_input =", repr(st.session_state.get("_pw_input")))
    st.write("DEBUG session_state _pw_ok =", repr(st.session_state.get("_pw_ok")))
    st.write("DEBUG match? =", st.session_state.get("_pw_input") == st.secrets.get("APP_PASSWORD"))
    for _line in st.session_state.get("_pw_debug_log", []):
        st.write("DEBUG attempt:", _line)
    def _password_entered():
        _correct = None
        try:
            _correct = st.secrets.get("APP_PASSWORD")
        except Exception:
            _correct = None
        _typed = st.session_state.get("_pw_input")
        _matched = bool(_correct) and _typed == _correct
        _log = st.session_state.setdefault("_pw_debug_log", [])
        _log.append(
            f"typed={_typed!r} (len={len(_typed) if _typed is not None else 'N/A'}) "
            f"correct={_correct!r} (len={len(_correct) if _correct is not None else 'N/A'}) "
            f"matched={_matched}"
        )
        if _matched:
            st.session_state["_pw_ok"] = True
            del st.session_state["_pw_input"]
        else:
            st.session_state["_pw_ok"] = False

    if st.session_state.get("_pw_ok"):
        return True

    st.text_input(
        "Password", type="password", key="_pw_input", on_change=_password_entered,
    )
    if st.session_state.get("_pw_ok") is False:
        st.error("Incorrect password.")
    return False


if not _check_password():
    st.stop()

# Disable Streamlit's default full-page dim/fade effect during reruns.
# This app reruns often (multiple auto-refresh timers across tabs, all
# firing on the same underlying script rerun regardless of which tab is
# visible), so the default fade-in/fade-out became a distracting flicker
# rather than a helpful "updating" cue. NOT independently verified
# against Streamlit's current internals (their exact selector/attribute
# for this has changed across versions and wasn't confirmed via live
# docs) -- targets the two most commonly-cited current selectors as a
# best-effort fix. If this doesn't fully eliminate the dimming, the
# actual current selector needs inspecting directly (browser dev tools,
# right-click the dimmed page during a rerun -> Inspect -> look for
# which element gains a reduced-opacity style) rather than guessing a
# third selector blind.
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { opacity: 1 !important; transition: none !important; }
    div[data-stale="true"] { opacity: 1 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("Dr Yarapu Reddy Levels")

with st.sidebar:
    st.markdown("### Connect to Upstox")
    st.caption(
        "Logs in via Upstox's own OAuth flow so the token this app gets actually carries "
        "this app's real permissions (market data + trading). A token grabbed some other "
        "way is what produces 'read only token' order-placement errors (UDAPI100067)."
    )
    _saved_oauth = load_saved_oauth_config()
    upstox_api_key = st.text_input(
        "API Key (Client ID)",
        value=_saved_oauth.get("api_key") or os.environ.get("UPSTOX_API_KEY", ""),
        key="upstox_api_key",
    )
    upstox_api_secret = st.text_input(
        "API Secret",
        value=_saved_oauth.get("api_secret") or os.environ.get("UPSTOX_API_SECRET", ""),
        type="password", key="upstox_api_secret",
    )
    upstox_redirect_uri = st.text_input(
        "Redirect URI",
        value=_saved_oauth.get("redirect_uri") or os.environ.get("UPSTOX_REDIRECT_URI", "https://127.0.0.1"),
        key="upstox_redirect_uri",
        help="Must exactly match the Redirect URI registered on this app in the Upstox "
             "developer console -- it doesn't need to be a real, reachable page.",
    )
    if _saved_oauth:
        st.caption(
            "API Key/Secret/Redirect URI filled in from a previous successful login -- "
            f"edit above if they've changed, or delete {UPSTOX_OAUTH_CONFIG_PATH} to clear."
        )

    if upstox_api_key.strip() and upstox_redirect_uri.strip():
        st.markdown(
            f"[1. Log in to Upstox]({build_upstox_login_url(upstox_api_key.strip(), upstox_redirect_uri.strip())})"
        )
        st.caption(
            "Opens Upstox's login page. After you approve, it redirects to your Redirect "
            "URI with `?code=...` in the address bar -- the page itself may show an "
            "error/404, that's fine, just copy the code value out of the URL."
        )
    else:
        st.caption("Enter API Key and Redirect URI above to get a login link.")

    upstox_auth_code = st.text_input("2. Paste the code from the redirect URL", key="upstox_auth_code")

    if st.button("3. Exchange code for access token"):
        if not (upstox_api_key.strip() and upstox_api_secret.strip()
                 and upstox_redirect_uri.strip() and upstox_auth_code.strip()):
            st.error("Fill in API Key, API Secret, Redirect URI, and the pasted code first.")
        else:
            ok, result = exchange_upstox_auth_code(
                upstox_auth_code, upstox_api_key, upstox_api_secret, upstox_redirect_uri,
            )
            if ok:
                st.session_state["manual_token"] = result["access_token"]
                save_oauth_config(upstox_api_key.strip(), upstox_api_secret.strip(), upstox_redirect_uri.strip())
                st.success(
                    f"Logged in as {result.get('user_name', result.get('email', 'your Upstox account'))}. "
                    "Access token is active for this app now -- nothing else to paste below. "
                    "API Key/Secret/Redirect URI are saved for next time too."
                )
            else:
                st.error(result)

    st.divider()
    st.markdown("### Daily token")
    st.text_input(
        "Upstox access token (paste today's token here)",
        type="password", key="manual_token",
        help="Filled in automatically after a successful login above. You can also paste "
             "a token directly here instead, as a fallback -- replace it here each day "
             "instead of editing the systemd service.",
    )

st.caption(
    "Cross-timeframe validated zones: a level only counts if BOTH the 18-day composite "
    "profile and today's intraday profile independently show volume clustered there."
)

col1, col2, col3 = st.columns(3)
run_precompute_clicked = col1.button("Run Precompute (slow, once/day)")
refresh_zones_clicked = col2.button("Refresh Zones (medium, every few min)")
refresh_quotes_clicked = col3.button("Refresh Quotes (fast)")

auto_refresh_enabled = st.checkbox(
    "Auto-refresh (scans on every 5-min candle close) - only while this tab stays open",
    value=False,
)
# The underlying rerun timer ticks every UI_RERUN_INTERVAL_SECONDS just to
# CHECK whether a candle boundary has passed -- the actual scan (and the
# heavier zone refresh, now on the same cadence) only fires once per real
# candle close, tracked via last_scanned_candle_boundary, regardless of
# how often this rerun timer ticks in between.
st_autorefresh(interval=UI_RERUN_INTERVAL_SECONDS * 1000, key="auto_refresh_tick") if auto_refresh_enabled else None
if "last_scanned_candle_boundary" not in st.session_state:
    st.session_state["last_scanned_candle_boundary"] = None

auto_quotes_due = False
if auto_refresh_enabled and MARKET_OPEN_TIME <= now_ist().time() < MARKET_CLOSE_TIME:
    _now = now_ist()
    _boundary_key = current_candle_boundary_key(_now)
    _floored_minute = (_now.minute // CANDLE_INTERVAL_MINUTES) * CANDLE_INTERVAL_MINUTES
    _boundary_dt = _now.replace(minute=_floored_minute, second=0, microsecond=0)
    _seconds_past_boundary = (_now - _boundary_dt).total_seconds()
    if (_seconds_past_boundary >= CANDLE_CLOSE_BUFFER_SECONDS
            and _boundary_key != st.session_state["last_scanned_candle_boundary"]):
        auto_quotes_due = True
        st.session_state["last_scanned_candle_boundary"] = _boundary_key

auto_zone_due = auto_quotes_due  # zone refresh now rides the same candle-close cadence as the scan

if run_precompute_clicked:
    token = get_token()
    progress_bar = st.progress(0)
    status_text = st.empty()
    def _cb(i, total, symbol):
        progress_bar.progress(i / total)
        status_text.text(f"{i}/{total}: {symbol}")
    with st.spinner("Running precompute..."):
        cache = run_precompute(token, progress_callback=_cb)
    st.success(f"Precompute done. {len(cache)} symbols cached.")

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r") as f:
        cache = json.load(f)

    if refresh_zones_clicked or auto_zone_due:
        token = get_token()
        progress_bar = st.progress(0)
        status_text = st.empty()
        def _cb2(i, total, symbol):
            progress_bar.progress(i / total)
            status_text.text(f"{i}/{total}: {symbol}")
        with st.spinner("Refreshing intraday zones..."):
            cache = run_zone_refresh(cache, token, progress_callback=_cb2)
        st.success("Zone refresh done.")

    if refresh_quotes_clicked or refresh_zones_clicked or auto_quotes_due or auto_zone_due or "last_scan_df" not in st.session_state:
        token = get_token()
        (scan_df, signals, top_n_symbols, bottom_setups, top_setups, resistance_breakdowns,
         support_reclaims, paper_trade_candidates, support_watch, resistance_watch) = run_live_scan(cache, token)

        alert_log = load_alert_log()
        alert_log = update_alert_log(alert_log, signals, top_n_symbols)
        save_alert_log(alert_log)

        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)

        st.session_state["last_scan_df"] = scan_df
        st.session_state["last_refresh_time"] = now_ist().strftime("%H:%M:%S")
        st.session_state["alert_log"] = alert_log
        st.session_state["bottom_setups"] = bottom_setups
        st.session_state["top_setups"] = top_setups
        st.session_state["resistance_breakdowns"] = resistance_breakdowns
        st.session_state["support_reclaims"] = support_reclaims
        st.session_state["support_watch"] = support_watch
        st.session_state["resistance_watch"] = resistance_watch

        # --- Paper trading (SIMULATED, no real orders) ---
        # Check exits on EXISTING open positions FIRST, then open new
        # candidates AFTER -- critical ordering. Doing it the other way
        # (open then immediately check) would let a freshly-opened trade
        # get evaluated for exit at the exact same tick/price it just
        # opened at, which is meaningless (entry==exit, 0 P&L every
        # time) and was happening for anything opened after market
        # close, when force_eod is already true.
        paper_log = load_paper_trades()
        scan_price_lookup = dict(zip(scan_df["Symbol"], scan_df["LTP"])) if not scan_df.empty else {}
        market_closing_now = now_ist().time() >= MARKET_CLOSE_TIME
        paper_log = check_paper_trade_exits(paper_log, scan_price_lookup, force_eod=market_closing_now)
        for candidate in paper_trade_candidates:
            if not has_open_paper_trade(paper_log, candidate["symbol"]):
                open_paper_trade(paper_log, candidate)
        save_paper_trades(paper_log)
        st.session_state["paper_log"] = paper_log

    df = st.session_state.get("last_scan_df", pd.DataFrame())
    price_lookup = dict(zip(df["Symbol"], df["LTP"])) if not df.empty else {}
    rvol_lookup = dict(zip(df["Symbol"], df["RVOL%"])) if not df.empty else {}
    alert_log = st.session_state.get("alert_log") or load_alert_log()

    symbols_with_zones = [s for s in cache if cache[s].get("composite_zones") or cache[s].get("intraday_zones")]
    default_idx = symbols_with_zones.index("NIFTY") if "NIFTY" in symbols_with_zones else 0

    # The Dashboard tab's own symbol picker deliberately uses the FULL
    # resolved F&O universe (every symbol Precompute successfully found
    # an instrument_key for -- 231 equities + NIFTY/BANKNIFTY futures),
    # not symbols_with_zones above. It's a single-symbol live viewer, so
    # unlike Zone Watch/Scanner/etc. it doesn't actually need a symbol to
    # have validated (or even any) zones to be worth looking at -- a
    # freshly-added or thinly-traded symbol with no zones yet should
    # still be selectable here.
    dash_all_symbols = sorted(cache.keys())
    dash_futures_symbols = sorted(s for s in dash_all_symbols if s in FUTURES_SYMBOLS)
    dash_equity_symbols = sorted(s for s in dash_all_symbols if s not in FUTURES_SYMBOLS)

    # --- Dashboard tab settings, in the sidebar (viewer.py-style single-
    # symbol layout) -- added in a SECOND `with st.sidebar:` block here,
    # after symbols_with_zones/cache exist, rather than in the first
    # sidebar block up top (which runs before cache is loaded from disk).
    # Streamlit appends every `with st.sidebar:` block's content into the
    # same sidebar in script order, so this is safe and just as global as
    # the "Daily token" section already up there -- it shows regardless
    # of which tab is open, same as any other sidebar content would. ---
    if dash_all_symbols:
        with st.sidebar:
            st.markdown("### 🏠 Dashboard")
            dash_instrument_filter = st.radio(
                "Instrument type", ["All", "Futures", "Equities"], index=0,
                horizontal=True, key="dash_instrument_filter",
                help="Futures = NIFTY/BANKNIFTY only. Equities = the rest of "
                     "the F&O stock universe.",
            )
            if dash_instrument_filter == "Futures":
                dash_symbol_universe = dash_futures_symbols or dash_all_symbols
            elif dash_instrument_filter == "Equities":
                dash_symbol_universe = dash_equity_symbols or dash_all_symbols
            else:
                dash_symbol_universe = dash_all_symbols
            dash_default_idx = dash_symbol_universe.index("NIFTY") if "NIFTY" in dash_symbol_universe else 0
            # The Symbol selectbox's stored value can outlive a filter
            # switch (e.g. RELIANCE selected, then flipping to "Futures")
            # -- reset it BEFORE the widget renders so Streamlit doesn't
            # choke on a stored value that's no longer among the options.
            if st.session_state.get("dash_symbol") not in dash_symbol_universe:
                st.session_state["dash_symbol"] = dash_symbol_universe[dash_default_idx]
            st.selectbox(
                "Symbol", dash_symbol_universe, index=dash_default_idx, key="dash_symbol",
            )
            st.slider(
                "Candles to show", min_value=30, max_value=500, value=300, step=10,
                key="dash_candles_to_show",
                help="Caps how many of today's already-fetched candles are plotted -- "
                     "this app's Dashboard tab is single-session (today only), unlike "
                     "viewer.py's multi-day view, so this mostly matters on the 1m "
                     "interval where today alone can have 300+ candles.",
            )
            st.checkbox(
                "Auto-refresh this chart -- own timer, same pattern as Live Ticks' "
                "toggle below, off by default so opening this tab doesn't force the "
                "whole app to rerun every few seconds on its own (this was previously "
                "unconditional and a major cause of sluggishness, especially on mobile)",
                value=False, key="dash_autorefresh_on",
            )
            st.slider(
                "Auto-refresh every (seconds)", min_value=5, max_value=60, value=15,
                key="dash_autorefresh_secs",
            )
    else:
        dash_symbol_universe = []
        dash_default_idx = 0

    # --- Always-visible top summary: level breaks with real room to move ---
    # Sits above the tabs so it's visible no matter which tab is open --
    # the whole point is to avoid scrolling through 220 sector charts to
    # find what's actionable right now.
    top_breakdowns = filter_by_room(st.session_state.get("resistance_breakdowns", []), MIN_ROOM_PCT_FOR_TOP_BOXES)
    top_reclaims = filter_by_room(st.session_state.get("support_reclaims", []), MIN_ROOM_PCT_FOR_TOP_BOXES)

    box_col1, box_col2 = st.columns(2)
    with box_col1:
        st.markdown(f"**🔴 Resistance breakdown** (>{MIN_ROOM_PCT_FOR_TOP_BOXES:.0f}% room to fall)")
        bd_df = build_level_cross_display_df(top_breakdowns, "breakdown")
        if bd_df.empty:
            st.caption("None this cycle.")
        else:
            st.dataframe(bd_df, use_container_width=True, hide_index=True, height=150)
    with box_col2:
        st.markdown(f"**🟢 Support reclaim** (>{MIN_ROOM_PCT_FOR_TOP_BOXES:.0f}% room to run)")
        rc_df = build_level_cross_display_df(top_reclaims, "reclaim")
        if rc_df.empty:
            st.caption("None this cycle.")
        else:
            st.dataframe(rc_df, use_container_width=True, hide_index=True, height=150)

    render_dashboard_hero(st.session_state.get("paper_log") or load_paper_trades())

    st.divider()

    _trade_mode_col1, _trade_mode_col2 = st.columns([1, 3])
    with _trade_mode_col1:
        st.radio(
            "Trade mode", ["Paper", "Real"], horizontal=True, key="trade_mode_toggle",
            help="Real places actual orders on Upstox (requires a confirm click). "
                 "Paper is simulated, tracked by paper_trader_daemon.py.",
        )
    if st.session_state.get("trade_mode_toggle") == "Real":
        st.warning("REAL trade mode is active -- Buy/Sell buttons below will place actual orders after you confirm.")

    tab_dashboard, tab_scanner, tab_levels, tab_chart, tab_sectors, tab_rvol, tab_range, tab_breakout, tab_room_to_run, tab_tomorrow, tab_zonewatch, tab_alerts = st.tabs(
        ["Dashboard", "Scanner", "Key Levels", "Chart", "Sectors", "By RVOL", "Wide Range", "Breakout Watch", "Room to Run", "Tomorrow's Levels", "Zone Watch", "Alerts"]
    )

    with tab_dashboard:
        st.markdown("""
<style>
.algolab-strip{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.algolab-card{background:#11161f;border:1px solid #232b3a;border-radius:12px;padding:12px 18px;min-width:130px}
.algolab-card .k{color:#8fa1bd;font-size:11.5px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.algolab-card .v{font-size:19px;font-weight:700;margin-top:2px;color:#cdd6e4}
.algolab-pos{color:#34D399} .algolab-neg{color:#F87171}
</style>
""", unsafe_allow_html=True)

        _dash_paper_log = load_paper_trades()
        _open_trades = [t for t in _dash_paper_log["trades"] if t["status"] == "open"]
        _closed_trades = [t for t in _dash_paper_log["trades"] if t["status"] != "open"]

        _live_pnl = 0.0
        for _t in _open_trades:
            _ltp = price_lookup.get(_t["symbol"])
            if _ltp is None:
                continue
            if _t["direction"] == "long":
                _t["_live_pnl"] = round((_ltp - _t["entry_price"]) * _t["qty"], 2)
            else:
                _t["_live_pnl"] = round((_t["entry_price"] - _ltp) * _t["qty"], 2)
            _t["_ltp"] = _ltp
            _live_pnl += _t["_live_pnl"]
        _booked_pnl = sum(_t.get("pnl") or 0 for _t in _closed_trades)
        _total_pnl = _live_pnl + _booked_pnl

        def _pnl_class(v):
            return "algolab-pos" if v > 0 else ("algolab-neg" if v < 0 else "")

        def _rupee(v):
            return f"{'-' if v < 0 else ''}\u20b9{abs(v):,.0f}"

        st.markdown(
            f'<div class="algolab-strip">'
            f'<div class="algolab-card"><div class="k">Open trades</div><div class="v">{len(_open_trades)}</div></div>'
            f'<div class="algolab-card"><div class="k">Closed</div><div class="v">{len(_closed_trades)}</div></div>'
            f'<div class="algolab-card"><div class="k">Live PnL</div><div class="v {_pnl_class(_live_pnl)}">{_rupee(_live_pnl)}</div></div>'
            f'<div class="algolab-card"><div class="k">Booked PnL</div><div class="v {_pnl_class(_booked_pnl)}">{_rupee(_booked_pnl)}</div></div>'
            f'<div class="algolab-card"><div class="k">Total PnL</div><div class="v {_pnl_class(_total_pnl)}">{_rupee(_total_pnl)}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Paper trades only -- tracked continuously by paper_trader_daemon.py, "
            "which runs independently of this browser tab, so positions persist "
            "even after closing the terminal. Real orders are logged separately "
            "in real_orders_log.json; there's no Upstox positions API wired in "
            "yet for live real-order PnL, so check Upstox's own console for that."
        )

        if not _dash_paper_log["trades"]:
            st.info("No paper trades yet today.")
        else:
            _dash_rows = []
            for _t in sorted(_dash_paper_log["trades"], key=lambda x: 0 if x["status"] == "open" else 1):
                _dash_rows.append({
                    "Symbol": _t["symbol"],
                    "Direction": _t["direction"].upper(),
                    "Entry": _t["entry_price"],
                    "SL": _t["stop_loss"],
                    "Target": _t["target"],
                    "Qty": _t["qty"],
                    "LTP": _t.get("_ltp", _t.get("exit_price")),
                    "PnL": _t.get("_live_pnl", _t.get("pnl")),
                    "Status": _t["status"].upper() if _t["status"] == "open" else (_t.get("exit_reason") or _t["status"]).upper(),
                    "Source": _t.get("source", "manual"),
                })
            _dash_df = pd.DataFrame(_dash_rows)

            def _style_pnl(v):
                if v is None:
                    return ""
                return "color: #1a7a4c" if v > 0 else ("color: #b3261e" if v < 0 else "")

            st.dataframe(
                _dash_df.style.map(_style_pnl, subset=["PnL"]),
                use_container_width=True, hide_index=True,
            )

    with tab_scanner:
        st.caption(f"Last refreshed: {st.session_state.get('last_refresh_time', 'never')}")
        if df.empty:
            st.write("No data yet - click Refresh Quotes.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_levels:
        st.caption(
            "Composite = 18-day profile (updates on Precompute). Intraday = today's session "
            "(updates on Refresh Zones). Only overlapping ranges across both count as validated."
        )
        if not symbols_with_zones:
            st.write("No zones available yet - click 'Run Precompute'.")
        else:
            selected_symbol = st.selectbox("Symbol", symbols_with_zones, index=default_idx, key="levels_symbol")
            c = cache[selected_symbol]
            st.caption(f"Zones last updated: {c.get('zones_updated_at', 'never')}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Composite (18-day)**")
                st.dataframe(build_zones_display_df(c.get("composite_zones", [])),
                             use_container_width=True, hide_index=True)
            with col_b:
                st.markdown("**Intraday (today)**")
                st.dataframe(build_zones_display_df(c.get("intraday_zones", [])),
                             use_container_width=True, hide_index=True)

            val_comp, val_intra, _ = cross_validated_zones(
                c.get("composite_zones", []), c.get("intraday_zones", [])
            )
            st.markdown("**Validated (confirmed by both timeframes)**")
            validated_display = build_zones_display_df(val_comp)
            if validated_display.empty:
                st.write("No cross-validated zones yet.")
            else:
                st.dataframe(validated_display, use_container_width=True, hide_index=True)

    with tab_chart:
        if not symbols_with_zones:
            st.write("No zones available yet - click 'Run Precompute'.")
        else:
            chart_symbol = st.selectbox("Symbol", symbols_with_zones, index=default_idx, key="chart_symbol")
            c = cache[chart_symbol]
            token = get_token()

            chart_df = get_today_candles(chart_symbol, c["instrument_key"], token)

            if chart_df.empty:
                st.write("No candle data yet for today.")
            else:
                val_comp, val_intra, _ = cross_validated_zones(
                    c.get("composite_zones", []), c.get("intraday_zones", [])
                )
                ml_lookup = build_ml_risk_lookup(c.get("composite_zones", []), val_comp, chart_df)
                chart_rvol = rvol_lookup.get(chart_symbol)
                chart_title = (f"{chart_symbol} - price with key levels (RVOL {chart_rvol:.0f}%)"
                               if chart_rvol is not None else f"{chart_symbol} - price with key levels")
                _chart_padding = (chart_df["high"].max() - chart_df["low"].min()) * 0.05
                _chart_y_range = (chart_df["low"].min() - _chart_padding, chart_df["high"].max() + _chart_padding)
                fig = plot_candles_with_zones(
                    chart_df,
                    composite_zones=c.get("composite_zones", []),
                    intraday_zones=c.get("intraday_zones", []),
                    validated_zones=val_comp,
                    title=chart_title,
                    x_range=get_session_x_range(chart_df),
                    ml_risk_lookup=ml_lookup,
                    ema_200=c.get("ema_200"),
                    y_range=_chart_y_range,
                )
                st.plotly_chart(fig, use_container_width=True)

                cvd_fig = build_cvd_chart(chart_df, height=140, x_range=get_session_x_range(chart_df))
                st.plotly_chart(cvd_fig, use_container_width=True, key=f"cvd_chart_{chart_symbol}")
                st.caption(
                    "Cumulative volume delta -- an APPROXIMATION from candle color (up-close "
                    "candle = buying volume, down-close = selling volume), since Upstox's candle "
                    "API doesn't provide actual bid/ask tick data. Not tick-accurate order flow."
                )

                st.markdown("**ML break-risk per zone**")
                st.caption(
                    "A confidence FILTER on top of the zones above, not a replacement -- "
                    "trained on ~2,290 labeled zone tests across 20 symbols and 2.5 months of "
                    "history, held-out AUC 0.76. Small sample, single recent time window -- "
                    "treat high-risk flags as a reason to be more cautious, not a standalone "
                    "signal to act on."
                )
                ml_df = build_ml_risk_df(c.get("composite_zones", []), val_comp, chart_df)
                if ml_df.empty:
                    st.write("Not enough data to compute ML risk right now.")
                else:
                    st.dataframe(ml_df, use_container_width=True, hide_index=True)

                st.markdown("**Manual paper trade**")
                st.caption(
                    "One-click SIMULATED entry using the same structure-based logic as the "
                    "automatic trades -- stop is the nearest zone against you, target is the "
                    "nearest zone with you. Needs a stop-side zone to open at all (no trade "
                    "without a defined risk control); target is left open if there's no zone "
                    "on that side yet -- exits via stop-loss or end-of-day instead."
                )
                long_candidate = build_manual_trade_candidate("long", chart_symbol, val_comp, chart_df)
                short_candidate = build_manual_trade_candidate("short", chart_symbol, val_comp, chart_df)

                col_buy, col_sell = st.columns(2)
                with col_buy:
                    if long_candidate is None:
                        st.write("Buy: no validated support to use as a stop-loss.")
                    else:
                        target_str = f"{long_candidate['target']:.2f}" if long_candidate['target'] is not None else "open (no resistance yet)"
                        st.write(f"Buy @ {long_candidate['entry_price']:.2f} | "
                                 f"Stop {long_candidate['stop_loss']:.2f} | "
                                 f"Target {target_str}"
                                 + (f" | ML Risk {long_candidate['ml_risk_pct']:.1f}%" if long_candidate['ml_risk_pct'] is not None else ""))
                        if st.button("Buy (Long)", key="manual_buy_btn"):
                            manual_paper_log = load_paper_trades()
                            if has_open_paper_trade(manual_paper_log, chart_symbol):
                                st.warning(f"Already have an open position in {chart_symbol}.")
                            else:
                                opened = open_paper_trade(manual_paper_log, long_candidate)
                                if opened:
                                    save_paper_trades(manual_paper_log)
                                    st.session_state["paper_log"] = manual_paper_log
                                    st.success(f"Opened manual LONG on {chart_symbol}.")
                                else:
                                    st.warning("Position size rounds to 0 shares at this price -- not opened.")

                with col_sell:
                    if short_candidate is None:
                        st.write("Sell: no validated resistance to use as a stop-loss.")
                    else:
                        target_str = f"{short_candidate['target']:.2f}" if short_candidate['target'] is not None else "open (no support yet)"
                        st.write(f"Sell @ {short_candidate['entry_price']:.2f} | "
                                 f"Stop {short_candidate['stop_loss']:.2f} | "
                                 f"Target {target_str}"
                                 + (f" | ML Risk {short_candidate['ml_risk_pct']:.1f}%" if short_candidate['ml_risk_pct'] is not None else ""))
                        if st.button("Sell (Short)", key="manual_sell_btn"):
                            manual_paper_log = load_paper_trades()
                            if has_open_paper_trade(manual_paper_log, chart_symbol):
                                st.warning(f"Already have an open position in {chart_symbol}.")
                            else:
                                opened = open_paper_trade(manual_paper_log, short_candidate)
                                if opened:
                                    save_paper_trades(manual_paper_log)
                                    st.session_state["paper_log"] = manual_paper_log
                                    st.success(f"Opened manual SHORT on {chart_symbol}.")
                                else:
                                    st.warning("Position size rounds to 0 shares at this price -- not opened.")

    with tab_sectors:
        st.caption(
            "Scan by sector to see who's sitting at support, who's stuck at resistance, "
            "and who's in open air."
        )
        available_sectors = sorted(set(
            SECTOR_MAP[s] for s in symbols_with_zones if s in SECTOR_MAP
        ))
        if not available_sectors:
            st.write("No sector data available yet - click 'Run Precompute'.")
        else:
            view_mode = st.radio(
                "View", ["One sector at a time", "Scroll through all sectors"],
                horizontal=True, key="sector_view_mode",
            )

            def render_symbol_grid(symbols_list, token, key_prefix="sector", cols_per_row=2):
                # 2 charts per row by default, each an independent chart with
                # its own zone lines and ML risk labels visible directly on
                # the chart (confirmed working) -- see chat history if
                # revisiting the synced-crosshair subplot version later.
                # cols_per_row=1 renders one full-width chart per row instead
                # -- used when this function is called inside an already-
                # narrow half-width column (Zone Watch's side-by-side
                # support/resistance layout), where pairing 2 charts would
                # squeeze each into a quarter-width sliver.
                # key_prefix keeps chart widget keys unique across the
                # different tabs that all call this same function
                # (Sectors, By RVOL, Wide Range, Zone Watch) -- since
                # Streamlit runs every tab's code every rerun regardless of
                # which is visually active, the same symbol appearing in
                # two tabs' calls in the same run would otherwise collide
                # on an identical hardcoded key.
                for i in range(0, len(symbols_list), cols_per_row):
                    row_symbols = symbols_list[i:i + cols_per_row]
                    cols = st.columns(len(row_symbols))
                    for col, sym in zip(cols, row_symbols):
                        with col:
                            c = cache[sym]
                            grid_df = get_today_candles(sym, c["instrument_key"], token)
                            if grid_df.empty:
                                st.write(f"{sym}: no candle data yet.")
                                continue
                            val_comp, _, _ = cross_validated_zones(
                                c.get("composite_zones", []), c.get("intraday_zones", [])
                            )
                            ml_lookup = build_ml_risk_lookup(
                                c.get("composite_zones", []), val_comp, grid_df
                            )
                            grid_rvol = rvol_lookup.get(sym)
                            grid_title = f"{sym} (RVOL {grid_rvol:.0f}%)" if grid_rvol is not None else sym
                            _grid_padding = (grid_df["high"].max() - grid_df["low"].min()) * 0.05
                            _grid_y_range = (grid_df["low"].min() - _grid_padding, grid_df["high"].max() + _grid_padding)
                            fig = plot_candles_with_zones(
                                grid_df,
                                composite_zones=[],   # hide faint reference lines in grid view -- too busy at small size
                                intraday_zones=[],
                                validated_zones=val_comp,
                                title=grid_title,
                                height=260,
                                compact=True,
                                x_range=get_session_x_range(grid_df),
                                ml_risk_lookup=ml_lookup,
                                ema_200=c.get("ema_200"),
                                y_range=_grid_y_range,
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart_{sym}")

                            cvd_fig = build_cvd_chart(grid_df, height=80, compact=True, x_range=get_session_x_range(grid_df))
                            st.plotly_chart(cvd_fig, use_container_width=True, key=f"{key_prefix}_cvd_{sym}")

                            long_c = build_manual_trade_candidate("long", sym, val_comp, grid_df)
                            short_c = build_manual_trade_candidate("short", sym, val_comp, grid_df)
                            _trade_mode = st.session_state.get("trade_mode_toggle", "Paper")
                            bcol, scol = st.columns(2)
                            with bcol:
                                if long_c is not None:
                                    _long_sl = st.number_input(
                                        "SL", value=float(long_c["stop_loss"]), step=0.05, format="%.2f",
                                        key=f"{key_prefix}_longsl_{sym}",
                                    )
                                    _long_tgt_default = float(long_c["target"]) if long_c["target"] is not None else float(long_c["entry_price"]) * 1.01
                                    _long_tgt = st.number_input(
                                        "Target", value=_long_tgt_default, step=0.05, format="%.2f",
                                        key=f"{key_prefix}_longtgt_{sym}",
                                    )
                                    _long_c_edited = dict(long_c)
                                    _long_c_edited["stop_loss"] = _long_sl
                                    _long_c_edited["target"] = _long_tgt
                                    _buy_pending_key = f"pending_real_{key_prefix}_buy_{sym}"
                                    if _trade_mode == "Real":
                                        if st.session_state.get(_buy_pending_key):
                                            if st.button("Confirm BUY (real)", key=f"{key_prefix}_buyconfirm_{sym}"):
                                                _ok, _msg, _order_ids = place_real_market_order(
                                                    c["instrument_key"], long_c["qty"], "BUY", token,
                                                )
                                                st.session_state[_buy_pending_key] = False
                                                if _ok:
                                                    log_real_order(sym, "BUY", long_c["qty"], _order_ids, c["instrument_key"])
                                                    st.success(f"{_msg} (your SL {_long_sl:.2f} / Target {_long_tgt:.2f} -- market order only, not auto-placed)")
                                                else:
                                                    st.error(_msg)
                                            if st.button("Cancel", key=f"{key_prefix}_buycancel_{sym}"):
                                                st.session_state[_buy_pending_key] = False
                                        else:
                                            if st.button("Buy (REAL)", key=f"{key_prefix}_buy_{sym}"):
                                                st.session_state[_buy_pending_key] = True
                                    else:
                                        if st.button("Buy", key=f"{key_prefix}_buy_{sym}"):
                                            grid_paper_log = load_paper_trades()
                                            if has_open_paper_trade(grid_paper_log, sym):
                                                st.warning(f"Already open in {sym}.")
                                            elif open_paper_trade(grid_paper_log, _long_c_edited):
                                                save_paper_trades(grid_paper_log)
                                                st.session_state["paper_log"] = grid_paper_log
                                                st.success(f"LONG opened: {sym}")
                                            else:
                                                st.warning("Qty rounds to 0 -- not opened.")
                            with scol:
                                if short_c is not None:
                                    _short_sl = st.number_input(
                                        "SL", value=float(short_c["stop_loss"]), step=0.05, format="%.2f",
                                        key=f"{key_prefix}_shortsl_{sym}",
                                    )
                                    _short_tgt_default = float(short_c["target"]) if short_c["target"] is not None else float(short_c["entry_price"]) * 0.99
                                    _short_tgt = st.number_input(
                                        "Target", value=_short_tgt_default, step=0.05, format="%.2f",
                                        key=f"{key_prefix}_shorttgt_{sym}",
                                    )
                                    _short_c_edited = dict(short_c)
                                    _short_c_edited["stop_loss"] = _short_sl
                                    _short_c_edited["target"] = _short_tgt
                                    _sell_pending_key = f"pending_real_{key_prefix}_sell_{sym}"
                                    if _trade_mode == "Real":
                                        if st.session_state.get(_sell_pending_key):
                                            if st.button("Confirm SELL (real)", key=f"{key_prefix}_sellconfirm_{sym}"):
                                                _ok, _msg, _order_ids = place_real_market_order(
                                                    c["instrument_key"], short_c["qty"], "SELL", token,
                                                )
                                                st.session_state[_sell_pending_key] = False
                                                if _ok:
                                                    log_real_order(sym, "SELL", short_c["qty"], _order_ids, c["instrument_key"])
                                                    st.success(f"{_msg} (your SL {_short_sl:.2f} / Target {_short_tgt:.2f} -- market order only, not auto-placed)")
                                                else:
                                                    st.error(_msg)
                                            if st.button("Cancel", key=f"{key_prefix}_sellcancel_{sym}"):
                                                st.session_state[_sell_pending_key] = False
                                        else:
                                            if st.button("Sell (REAL)", key=f"{key_prefix}_sell_{sym}"):
                                                st.session_state[_sell_pending_key] = True
                                    else:
                                        if st.button("Sell", key=f"{key_prefix}_sell_{sym}"):
                                            grid_paper_log = load_paper_trades()
                                            if has_open_paper_trade(grid_paper_log, sym):
                                                st.warning(f"Already open in {sym}.")
                                            elif open_paper_trade(grid_paper_log, _short_c_edited):
                                                save_paper_trades(grid_paper_log)
                                                st.session_state["paper_log"] = grid_paper_log
                                                st.success(f"SHORT opened: {sym}")
                                            else:
                                                st.warning("Qty rounds to 0 -- not opened.")

            if view_mode == "One sector at a time":
                selected_sector = st.selectbox("Sector", available_sectors, key="sector_select")
                sector_symbols = [s for s in symbols_with_zones if SECTOR_MAP.get(s) == selected_sector]

                if not sector_symbols:
                    st.write("No symbols with zones in this sector yet.")
                else:
                    token = get_token()
                    render_symbol_grid(sector_symbols, token)

            else:
                total_symbols = len([s for s in symbols_with_zones if s in SECTOR_MAP])
                st.caption(
                    f"Renders all {total_symbols} symbols across {len(available_sectors)} sectors "
                    f"in one continuous scroll -- this fetches a lot of candle data at once, so it's "
                    f"gated behind this button rather than running automatically. Once loaded, "
                    f"results are cached for 45s so switching tabs or auto-refresh ticks won't "
                    f"immediately re-fetch everything."
                )
                if st.button("Load all sectors", key="load_all_sectors"):
                    st.session_state["show_all_sectors"] = True

                if st.session_state.get("show_all_sectors"):
                    token = get_token()
                    for sector in available_sectors:
                        sector_symbols = [s for s in symbols_with_zones if SECTOR_MAP.get(s) == sector]
                        if not sector_symbols:
                            continue
                        st.markdown(f"## {sector}")
                        render_symbol_grid(sector_symbols, token)
                        st.divider()

    with tab_rvol:
        st.caption(
            "Every stock with zones ranked by RVOL, highest first -- ignores sector "
            "grouping entirely, so the most active names across the WHOLE universe "
            "float to the top regardless of which sector they're in. Same charts, zone "
            "lines, and ML risk labels as the Sectors tab -- just a different ordering."
        )
        rvol_ranked_symbols = sorted(
            [s for s in symbols_with_zones if rvol_lookup.get(s) is not None],
            key=lambda s: rvol_lookup[s], reverse=True,
        )
        if not rvol_ranked_symbols:
            st.write("No RVOL data yet -- click 'Refresh Quotes'.")
        else:
            top_n_rvol_charts = st.slider(
                "Show top N by RVOL", min_value=5, max_value=len(rvol_ranked_symbols),
                value=min(10, len(rvol_ranked_symbols)), key="rvol_tab_top_n",
            )
            token = get_token()
            render_symbol_grid(rvol_ranked_symbols[:top_n_rvol_charts], token, key_prefix="rvol")

    with tab_range:
        st.caption(
            "Every stock ranked by the GAP between its nearest validated support "
            "and resistance -- i.e. how much room price actually has to move "
            "before hitting a wall in either direction. A wide gap means real "
            "room for a move to develop (breakout continuation or a range play) "
            "without immediately running into the next level. Only includes "
            "stocks with BOTH a support AND a resistance currently validated -- "
            "open air on one side makes the gap undefined, not comparable."
        )
        wide_range_df = build_wide_range_df(cache, price_lookup)
        if wide_range_df.empty:
            st.write("No stocks with both a validated support and resistance right now.")
        else:
            top_n_range = st.slider(
                "Show top N by gap", min_value=5, max_value=len(wide_range_df),
                value=min(10, len(wide_range_df)), key="range_tab_top_n",
            )
            shown_range_df = wide_range_df.head(top_n_range)
            st.dataframe(shown_range_df, use_container_width=True, hide_index=True)
            st.divider()
            token = get_token()
            render_symbol_grid(shown_range_df["Symbol"].tolist(), token, key_prefix="range")

    with tab_breakout:
        st.caption(
            "Two setups worth watching: 'Open Air' stocks have a validated zone "
            "on only ONE side (support or resistance) -- the other side has "
            "nothing to slow price down, which tends to produce sharp, "
            "one-directional moves. 'Testing Support Floor' flags stocks "
            "currently sitting right at the bottom edge of a support zone -- "
            "the setup stage, before a potential move through and past it, "
            "rather than flagging it only after the breakout already happened."
        )


        st.divider()
        st.markdown("### Pure CVD + Room Setups")
        st.caption(
            "Combines the room check above with a PURE (non-flipping) cumulative "
            "volume delta: 'Sell candidates' need the support side open AND CVD "
            "entirely red all session (never went positive). 'Buy candidates' need "
            "the resistance side open AND CVD entirely green all session (never "
            "went negative). Any flip either way disqualifies a stock -- mixed CVD "
            "means conviction isn't one-directional."
        )
        _CVD_ROOM_THRESHOLD_PCT = 2.0

        def _cvd_side_room(_prev_close, _side_zones):
            if not _side_zones:
                return True, "no zone this side"
            if len(_side_zones) == 1:
                return True, f"only zone {_side_zones[0]['price_mode']:.0f}"
            _gap_pct = abs(_side_zones[1]["price_mode"] - _side_zones[0]["price_mode"]) / _prev_close * 100
            return _gap_pct >= _CVD_ROOM_THRESHOLD_PCT, f"gap {_gap_pct:.1f}%"

        _sell_candidates = []
        _buy_candidates = []
        _cvd_token = get_token()
        for _sym in symbols_with_zones:
            _c = cache[_sym]
            _prev_close = _c.get("prev_close")
            _comp_zones = _c.get("composite_zones", [])
            if _prev_close is None or not _comp_zones:
                continue
            _support_side = sorted(
                [z for z in _comp_zones if z["price_mode"] <= _prev_close],
                key=lambda z: _prev_close - z["price_mode"],
            )
            _resistance_side = sorted(
                [z for z in _comp_zones if z["price_mode"] > _prev_close],
                key=lambda z: z["price_mode"] - _prev_close,
            )
            _sup_open, _ = _cvd_side_room(_prev_close, _support_side)
            _res_open, _ = _cvd_side_room(_prev_close, _resistance_side)
            if not (_sup_open or _res_open):
                continue
            _cvd_df = get_today_candles(_sym, _c["instrument_key"], _cvd_token)
            if _cvd_df.empty:
                continue
            _cvd_series = compute_cumulative_volume_delta(_cvd_df)
            _rvol = rvol_lookup.get(_sym)
            _rvol_rounded = round(_rvol, 0) if _rvol is not None else None
            if _sup_open and _cvd_series.max() < 0:
                _sell_candidates.append({"Symbol": _sym, "RVOL%": _rvol_rounded})
            if _res_open and _cvd_series.min() >= 0:
                _buy_candidates.append({"Symbol": _sym, "RVOL%": _rvol_rounded})

        _col_sell, _col_buy = st.columns(2)
        with _col_sell:
            st.markdown("**Sell candidates**")
            if not _sell_candidates:
                st.write("None right now.")
            else:
                _sell_df = pd.DataFrame(_sell_candidates).sort_values(
                    "RVOL%", ascending=False, na_position="last"
                ).reset_index(drop=True)
                st.dataframe(_sell_df, use_container_width=True, hide_index=True)
        with _col_buy:
            st.markdown("**Buy candidates**")
            if not _buy_candidates:
                st.write("None right now.")
            else:
                _buy_df = pd.DataFrame(_buy_candidates).sort_values(
                    "RVOL%", ascending=False, na_position="last"
                ).reset_index(drop=True)
                st.dataframe(_buy_df, use_container_width=True, hide_index=True)

        _cvd_room_symbols = [r["Symbol"] for r in _sell_candidates] + [r["Symbol"] for r in _buy_candidates]
        if _cvd_room_symbols:
            st.divider()
            render_symbol_grid(_cvd_room_symbols, _cvd_token, key_prefix="cvdroom")

        st.divider()
        st.markdown("### Testing Support Floor")
        _floor_threshold_pct = st.slider(
            "Flag when price is within this % of the support zone's bottom edge",
            min_value=0.1, max_value=2.0, value=0.5, step=0.1, key="floor_threshold_pct",
        )
        _floor_rows = []
        for _sym in symbols_with_zones:
            _c = cache[_sym]
            _ltp = price_lookup.get(_sym)
            if _ltp is None:
                continue
            _val_comp, _, _ = cross_validated_zones(_c.get("composite_zones", []), _c.get("intraday_zones", []))
            _best = None
            for _z in _val_comp:
                if _z["price_mode"] > _ltp:
                    continue
                _dist_to_floor = abs(_ltp - _z["price_low"]) / _ltp * 100
                if _dist_to_floor <= _floor_threshold_pct and (_best is None or _dist_to_floor < _best[1]):
                    _best = (_z, _dist_to_floor)
            if _best is not None:
                _floor_rows.append({"Symbol": _sym, "Distance to floor %": round(_best[1], 2)})

        if not _floor_rows:
            st.write("No stocks currently testing a support floor within that threshold.")
        else:
            _floor_df = pd.DataFrame(_floor_rows).sort_values("Distance to floor %").reset_index(drop=True)
            st.dataframe(_floor_df, use_container_width=True, hide_index=True)
            st.divider()
            _token2 = get_token()
            render_symbol_grid(_floor_df["Symbol"].tolist(), _token2, key_prefix="floor")

    with tab_room_to_run:
        st.markdown("### Room to Run (pre-market, based on prev close)")
        st.caption(
            "Uses PREVIOUS CLOSE (known before market open) as the reference "
            "point -- splits composite zones into support-side (below prev "
            "close) and resistance-side (above). A side counts as having room "
            "if it has no zone at all, only one zone, OR the gap to the "
            "second-nearest zone on that side is at least 2% -- real room "
            "once the near level is crossed. A tight cluster of levels close "
            "together does NOT qualify even if open air exists further out, "
            "since price has to fight through each one first."
        )
        _ROOM_THRESHOLD_PCT = st.slider(
            "Minimum gap to count as room (%)", min_value=0.5, max_value=5.0,
            value=2.0, step=0.5, key="room_threshold_pct",
        )

        def _side_room(_prev_close, _side_zones):
            """Returns (is_open, detail_str) for one side's sorted-by-
            distance zone list."""
            if not _side_zones:
                return True, "no zone this side"
            if len(_side_zones) == 1:
                return True, f"only zone {_side_zones[0]['price_mode']:.0f}"
            _gap_pct = abs(_side_zones[1]["price_mode"] - _side_zones[0]["price_mode"]) / _prev_close * 100
            _detail = f"{_side_zones[0]['price_mode']:.0f} -> {_side_zones[1]['price_mode']:.0f} gap {_gap_pct:.1f}%"
            return _gap_pct >= _ROOM_THRESHOLD_PCT, _detail

        _room_rows = []
        for _sym in symbols_with_zones:
            _c = cache[_sym]
            _prev_close = _c.get("prev_close")
            _comp_zones = _c.get("composite_zones", [])
            if _prev_close is None or not _comp_zones:
                continue
            _support_side = sorted(
                [z for z in _comp_zones if z["price_mode"] <= _prev_close],
                key=lambda z: _prev_close - z["price_mode"],
            )
            _resistance_side = sorted(
                [z for z in _comp_zones if z["price_mode"] > _prev_close],
                key=lambda z: z["price_mode"] - _prev_close,
            )
            _sup_open, _sup_detail = _side_room(_prev_close, _support_side)
            _res_open, _res_detail = _side_room(_prev_close, _resistance_side)
            if _sup_open or _res_open:
                _open_sides = []
                if _sup_open:
                    _open_sides.append(f"Support side ({_sup_detail})")
                if _res_open:
                    _open_sides.append(f"Resistance side ({_res_detail})")
                _rvol = rvol_lookup.get(_sym)
                _room_rows.append({
                    "Symbol": _sym,
                    "RVOL%": round(_rvol, 0) if _rvol is not None else None,
                    "Open side(s)": "; ".join(_open_sides),
                })

        if not _room_rows:
            st.write("No stocks meet the room threshold right now.")
        else:
            _room_df = pd.DataFrame(_room_rows).sort_values(
                "RVOL%", ascending=False, na_position="last"
            ).reset_index(drop=True)
            st.dataframe(_room_df, use_container_width=True, hide_index=True)
            st.divider()
            _token = get_token()
            render_symbol_grid(_room_df["Symbol"].tolist(), _token, key_prefix="roomcheck")

    with tab_tomorrow:
        st.caption(
            "Preview of tomorrow's zones using TODAY's close as the reference -- ready "
            "the moment Precompute finishes after today's close, no need to wait for "
            "market open. Same room rule as 'Room to Run' above. Charts are "
            "intentionally empty except for the zone lines and today's close -- "
            "tomorrow's candles don't exist yet."
        )
        _tmr_threshold_pct = st.slider(
            "Minimum gap to count as room (%)", min_value=0.5, max_value=5.0,
            value=2.0, step=0.5, key="tomorrow_threshold_pct",
        )

        def _tmr_side_room(_prev_close, _side_zones):
            if not _side_zones:
                return True
            if len(_side_zones) == 1:
                return True
            _gap_pct = abs(_side_zones[1]["price_mode"] - _side_zones[0]["price_mode"]) / _prev_close * 100
            return _gap_pct >= _tmr_threshold_pct

        _tmr_rows = []
        for _sym in symbols_with_zones:
            _c = cache[_sym]
            _prev_close = _c.get("prev_close")
            _comp_zones = _c.get("composite_zones", [])
            if _prev_close is None:
                continue
            _support_side = sorted(
                [z for z in _comp_zones if z["price_mode"] <= _prev_close],
                key=lambda z: _prev_close - z["price_mode"],
            )
            _resistance_side = sorted(
                [z for z in _comp_zones if z["price_mode"] > _prev_close],
                key=lambda z: z["price_mode"] - _prev_close,
            )
            _sup_open = _tmr_side_room(_prev_close, _support_side)
            _res_open = _tmr_side_room(_prev_close, _resistance_side)
            if _sup_open or _res_open:
                _rvol = rvol_lookup.get(_sym)
                _tmr_rows.append({
                    "Symbol": _sym,
                    "RVOL%": round(_rvol, 0) if _rvol is not None else None,
                    "Prev Close": round(_prev_close, 2),
                })

        if not _tmr_rows:
            st.write("No stocks meet the room threshold right now.")
        else:
            _tmr_df = pd.DataFrame(_tmr_rows).sort_values(
                "RVOL%", ascending=False, na_position="last"
            ).reset_index(drop=True)
            st.dataframe(_tmr_df, use_container_width=True, hide_index=True)
            st.divider()

            _tmr_date = now_ist().date() + timedelta(days=1)
            _tmr_t0 = pd.Timestamp(f"{_tmr_date} 09:15:00")
            _tmr_t1 = pd.Timestamp(f"{_tmr_date} 15:30:00")
            _tmr_symbols = _tmr_df["Symbol"].tolist()

            for _i in range(0, len(_tmr_symbols), 2):
                _row_syms = _tmr_symbols[_i:_i + 2]
                _cols = st.columns(len(_row_syms))
                for _col, _sym in zip(_cols, _row_syms):
                    with _col:
                        _c = cache[_sym]
                        _prev_close = _c.get("prev_close")
                        _comp_zones = _c.get("composite_zones", [])
                        _empty_df = pd.DataFrame({
                            "timestamp": [_tmr_t0, _tmr_t1],
                            "open": [_prev_close, _prev_close],
                            "high": [_prev_close, _prev_close],
                            "low": [_prev_close, _prev_close],
                            "close": [_prev_close, _prev_close],
                            "volume": [0, 0],
                        })
                        _tmr_fig = plot_candles_with_zones(
                            _empty_df,
                            composite_zones=_comp_zones,
                            intraday_zones=[],
                            validated_zones=[],
                            title=f"{_sym} - Tomorrow (prev close {_prev_close:.2f})",
                            height=260,
                            compact=True,
                            x_range=(_tmr_t0, _tmr_t1),
                        )
                        st.plotly_chart(_tmr_fig, use_container_width=True, key=f"tomorrow_chart_{_sym}")

    with tab_zonewatch:
        st.caption(
            f"Every symbol currently sitting within {NEAR_ZONE_PCT}% of a validated "
            f"support or resistance edge -- shown continuously, every cycle, unlike "
            f"the Setups tab (which only shows a symbol the ONE cycle its VWAP+zone "
            f"cross fires, then drops it). A row here moves from 'Watching' to "
            f"'Crossed' the instant that same confirmation happens, and stays "
            f"visible either way -- so you can watch a candidate the whole time "
            f"it's approaching a zone, not just catch the one instant it confirms."
        )
        token = get_token()

        support_watch = st.session_state.get("support_watch", [])
        resistance_watch = st.session_state.get("resistance_watch", [])

        support_crossed = [r["symbol"] for r in support_watch if r["crossed"]]
        support_watching = [r["symbol"] for r in support_watch if not r["crossed"]]
        resistance_crossed = [r["symbol"] for r in resistance_watch if r["crossed"]]
        resistance_watching = [r["symbol"] for r in resistance_watch if not r["crossed"]]

        # Quality score (weighted composite of ML-confidence + recent CVD
        # order-flow + RVOL + room-to-next-zone, 0-100 -- see
        # zone_validation.compute_cvd_zone_signal) sorts each list
        # highest-score-first and is called out explicitly, rather than
        # changing render_symbol_grid itself to understand it -- keeps
        # that shared function (also used by Sectors/By RVOL/Wide Range)
        # untouched. This is a ranking on top of the existing "near a
        # zone" membership, not an extra filter -- a low-scoring symbol
        # still shows up, just further down.
        def _sort_by_quality(symbols, watch_list):
            entry_by_symbol = {r["symbol"]: r for r in watch_list}
            ordered = sorted(
                symbols, key=lambda s: entry_by_symbol.get(s, {}).get("quality_score", 0.0),
                reverse=True,
            )
            graded = {s: entry_by_symbol[s] for s in symbols
                      if entry_by_symbol.get(s, {}).get("quality_grade") in ("BUY", "SELL", "WATCH")}
            return ordered, graded

        support_crossed, support_crossed_graded = _sort_by_quality(support_crossed, support_watch)
        support_watching, support_watching_graded = _sort_by_quality(support_watching, support_watch)
        resistance_crossed, resistance_crossed_graded = _sort_by_quality(resistance_crossed, resistance_watch)
        resistance_watching, resistance_watching_graded = _sort_by_quality(resistance_watching, resistance_watch)

        def _quality_caption(graded):
            """graded: {symbol: watch-entry dict}. A BUY/SELL-grade line
            (score >= CVD_SIGNAL_BUY_THRESHOLD) means every scored factor
            that had data agreed strongly; a WATCH-grade line
            (score >= CVD_SIGNAL_WATCH_THRESHOLD) is a real but imperfect
            setup -- shown, not filtered out, per the weighted-composite
            approach (a razor-perfect setup with one weak factor still
              surfaces here instead of vanishing behind a strict gate)."""
            if not graded:
                return
            strong = sorted((r for r in graded.values() if r["quality_grade"] in ("BUY", "SELL")),
                             key=lambda r: r["quality_score"], reverse=True)
            watching = sorted((r for r in graded.values() if r["quality_grade"] == "WATCH"),
                               key=lambda r: r["quality_score"], reverse=True)
            if strong:
                st.caption("🟢 " + ", ".join(f"{r['symbol']} ({r['quality_score']:.0f})" for r in strong))
            if watching:
                st.caption("🟡 " + ", ".join(f"{r['symbol']} ({r['quality_score']:.0f})" for r in watching))

        st.markdown("### Watching (side by side -- buy-side vs sell-side candidates)")
        st.caption("Sitting near a zone edge, hasn't confirmed a cross yet. Support side = "
                   "bullish candidates (bounce off a floor); Resistance side = bearish "
                   "candidates (rejection off a ceiling) -- same HUDCO-style setup, mirrored. "
                   "One chart per row on each side (rather than the usual 2) so both columns "
                   "stay readable at half width. Ranked by quality score, highest first -- "
                   "🟢 = strong composite confirmation (score ≥70), 🟡 = worth watching (45-69).")
        watch_col_support, watch_col_resistance = st.columns(2)
        with watch_col_support:
            st.markdown(f"**👀 Support ({len(support_watching)})**")
            _quality_caption(support_watching_graded)
            if not support_watching:
                st.caption("None right now.")
            else:
                render_symbol_grid(support_watching, token, key_prefix="zw_supwatch", cols_per_row=1)
        with watch_col_resistance:
            st.markdown(f"**👀 Resistance ({len(resistance_watching)})**")
            _quality_caption(resistance_watching_graded)
            if not resistance_watching:
                st.caption("None right now.")
            else:
                render_symbol_grid(resistance_watching, token, key_prefix="zw_reswatch", cols_per_row=1)

        st.divider()
        st.markdown("### Crossed this cycle (confirmed -- full chart + trade actions)")

        st.markdown(f"**🔥 Support reclaimed -- bullish ({len(support_crossed)})**")
        _quality_caption(support_crossed_graded)
        if not support_crossed:
            st.caption("None right now.")
        else:
            render_symbol_grid(support_crossed, token, key_prefix="zw_supcross")

        st.markdown(f"**🔥 Resistance broken down -- bearish ({len(resistance_crossed)})**")
        _quality_caption(resistance_crossed_graded)
        if not resistance_crossed:
            st.caption("None right now.")
        else:
            render_symbol_grid(resistance_crossed, token, key_prefix="zw_rescross")

    with tab_alerts:
        st.caption(
            f"Logged the moment a symbol's signal changes to a fresh BUY/SELL - not repeated "
            f"every refresh it stays active. Only the top {TOP_N_RVOL} symbols by RVOL are eligible "
            f"to alert. Only logged during market hours (9:15-15:30 IST)."
        )
        alert_df = build_alert_display_df(alert_log, price_lookup)
        if alert_df.empty:
            st.write("No alerts logged yet.")
        else:
            st.dataframe(alert_df, use_container_width=True, hide_index=True)
            csv = alert_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download alert log CSV", csv, "alert_log.csv", "text/csv")

else:
    st.info("No cache found yet - click 'Run Precompute' first.")
