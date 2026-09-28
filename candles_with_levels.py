"""
Candlestick chart with key levels + session VWAP, styled like a trading
platform (dark background, TradingView-style candle colors and gridlines).

THREE-TIER ZONE DISPLAY:
  - Composite zones (18-day history): known BEFORE the session even
    starts, so these get full support/resistance treatment (colored fill,
    dashed line, label) from the very first candle -- no waiting for
    today's intraday profile to "catch up" before showing a real,
    multi-day-backed level.
  - Validated zones (composite zones ALSO confirmed by today's intraday
    profile): same color coding, but bolder (thicker line, more opaque
    fill) -- the "extra confirmed" tier, layered on top of the matching
    composite zone at the same price.
  - Intraday-only zones (today's profile shows a cluster, but it's NOT
    backed by composite history): stay as faint, unlabeled dotted
    reference lines. These are the ones most prone to early-session
    noise (see backtest_zone_formation.py -- a single opening candle can
    trivially look like "the peak" of its own tiny profile), so they
    deliberately don't get the same visual weight as a real level.

Validated/composite zones are classified relative to the LAST close price:
  - zone below last close  -> SUPPORT   (green)
  - zone above last close  -> RESISTANCE (red)
Session VWAP is computed from the candle df itself (cumulative typical
price weighted by volume) -- no extra API call needed.

Matches the zone dict shape from sahi_style_key_levels():
    {"price_mode": float, "label": str (e.g. "31%"), "price_low": float, "price_high": float}
"""

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# TradingView-style dark theme palette
BG_COLOR = "#131722"
GRID_COLOR = "rgba(255, 255, 255, 0.06)"
TEXT_COLOR = "#D1D4DC"

CANDLE_UP = "#26A69A"
CANDLE_DOWN = "#EF5350"

SUPPORT_FILL = "rgba(38, 166, 154, 0.10)"
SUPPORT_LINE = "#26A69A"
RESISTANCE_FILL = "rgba(239, 83, 80, 0.10)"
RESISTANCE_LINE = "#EF5350"
VALIDATED_SUPPORT_FILL = "rgba(38, 166, 154, 0.22)"
VALIDATED_RESISTANCE_FILL = "rgba(239, 83, 80, 0.22)"
REFERENCE_LINE = "rgba(209, 212, 220, 0.25)"  # faint, no label -- intraday-only zones
VWAP_LINE = "#FF9800"  # orange, standard VWAP color on trading platforms
LTP_LINE = "#2962FF"


def _pct_from_label(label):
    m = re.search(r"[\d.]+", str(label))
    return float(m.group()) if m else 0.0


def _session_vwap(df):
    """Cumulative typical-price VWAP for a single session's candle df.
    Requires 'high','low','close','volume' columns already scoped to one day."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (typical * df["volume"]).cumsum()
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap.ffill()


def _zone_key(z):
    """Rounds price_mode for matching a composite zone against its
    validated counterpart -- validated_zones' price_mode comes from the
    same composite zone dict, so this should match exactly, but rounding
    guards against any float precision drift."""
    return round(z["price_mode"], 2)


def plot_candles_with_zones(df, composite_zones=None, intraday_zones=None,
                             validated_zones=None, title="Price with key levels",
                             show_vwap=True, x_range=None, height=500, compact=False,
                             tick_format=None, y_range=None, market_hours_breaks=False,
                             event_markers=None, ml_risk_lookup=None, ema_200=None):
    """
    df: OHLC(V) dataframe with columns ['timestamp','open','high','low','close']
        and ideally 'volume' (needed for the VWAP line).
    composite_zones: known before the session starts -- shown with full
        support/resistance treatment from the first candle.
    intraday_zones: today's still-forming profile -- shown as faint
        reference lines UNLESS a zone also appears in composite_zones
        (then it's covered by the composite/validated treatment instead).
    validated_zones: composite zones also confirmed by intraday -- drawn
        bolder on top of the matching composite zone.
    x_range: optional (start, end) tuple to PIN the x-axis to a fixed
        span regardless of how many candles are actually in df -- needed
        for the Replay tab, where df is a growing slice of a session and
        without this, Plotly's autorange fits tightly to whatever's
        plotted so far. With only 1-2 candles that means the axis spans
        just a few minutes, stretching a single candle to fill almost
        the entire chart width. Passing the FULL session's (start, end)
        here keeps candles properly sized from the very first step,
        simply "filling in" left to right as more data is revealed.
    height: chart height in px. Default 500 for a standalone Chart tab
        view; pass a smaller value (e.g. 260) for grid/multi-pane views
        like the Sectors tab.
    compact: True for small multi-pane grid cells (Sectors tab) -- hides
        the legend, tightens margins, and shrinks label/annotation fonts
        so the chart stays readable at a fraction of full size.
    y_range: optional (min, max) tuple to PIN the y-axis to a fixed price
        range -- needed to make two side-by-side panels for the same
        stock (e.g. 18-day composite + today) share an identical vertical
        scale, so the same price level lines up at the same height in
        both charts for direct visual comparison.
    market_hours_breaks: True to hide non-trading hours (overnight and
        weekend gaps) on the x-axis -- needed when df spans multiple
        calendar days at intraday (e.g. 5-min) granularity, since without
        this a continuous time axis shows huge flat gaps for every night
        and weekend, squeezing the actual trading data into thin slivers.
        Leave False for single-session charts (today's candles, Replay),
        where there's no multi-day gap to hide anyway.
    event_markers: optional list of {"time": datetime, "label": str,
        "color": optional hex str} dicts -- draws a vertical dotted line
        spanning the full chart height at each event's time, with a short
        label at the top. Ties Setups-tab alerts (a resistance breakdown,
        a support reclaim, a confirmation loss) to the exact candle where
        they fired, instead of leaving them only in a separate table.
    ml_risk_lookup: optional {rounded price_mode: risk_pct} dict -- when
        a zone's price matches a key here, its ML break-risk probability
        is appended directly to that zone's on-chart label (e.g.
        "Support 893 (19%) - ML 4%"), so the risk is visible right on
        the line itself, not just in a separate table.
    ema_200: optional float -- the 200-period EMA on 5-min candles,
        computed once at Precompute from 18 days of composite history
        (see compute_ema_200). Drawn as a single horizontal reference
        line rather than a true curve: a 200-period EMA is inherently
        slow-moving, and recomputing a full live curve would need the
        entire multi-day candle series again on every render. None
        means not enough history yet (e.g. a newly-listed stock) --
        simply omits the line rather than guessing.
    """
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["timestamp"],
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color=CANDLE_UP, increasing_fillcolor=CANDLE_UP,
        decreasing_line_color=CANDLE_DOWN, decreasing_fillcolor=CANDLE_DOWN,
        name="Price",
    ))

    if show_vwap and "volume" in df.columns and df["volume"].sum() > 0:
        vwap_series = _session_vwap(df)
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=vwap_series,
            mode="lines", name="VWAP",
            line=dict(color=VWAP_LINE, width=1.5, dash="solid"),
        ))

    # x0/x1 drive both the axis range AND how far zone/VWAP/LTP lines
    # stretch -- use the caller-supplied full-session span if given,
    # otherwise fall back to the actual plotted data's own span (the
    # original behavior, still correct for the Chart/Sectors tabs where
    # df already IS the full available session).
    if x_range is not None:
        x0, x1 = x_range
    else:
        x0, x1 = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]

    if ema_200 is not None:
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[ema_200, ema_200],
            mode="lines", name="EMA 200",
            line=dict(color="#B983FF", width=1.5, dash="dot"),
        ))
    last_close = float(df["close"].iloc[-1])

    composite_zones = composite_zones or []
    intraday_zones = intraday_zones or []
    validated_zones = validated_zones or []

    validated_keys = {_zone_key(z) for z in validated_zones}
    composite_keys = {_zone_key(z) for z in composite_zones}

    # Intraday-only zones (not backed by composite history at all) --
    # these are the ones most prone to early-session noise, so they stay
    # faint and unlabeled rather than getting full support/resistance
    # visual weight.
    for z in intraday_zones:
        if _zone_key(z) in composite_keys:
            continue  # already covered by the composite/validated treatment below
        try:
            zone_price = float(z["price_mode"])
        except (TypeError, ValueError):
            continue
        try:
            fig.add_shape(
                type="line", x0=x0, x1=x1, y0=zone_price, y1=zone_price,
                line=dict(color=REFERENCE_LINE, width=1, dash="dot"),
            )
        except Exception:
            continue

    # Composite zones -- known before the session starts, so these get
    # full support/resistance treatment from the very first candle.
    # Validated ones (also confirmed by intraday) get a bolder version,
    # layered at the same price.
    price_span = (df["high"].max() - df["low"].min()) if len(df) > 1 else 1
    min_gap = price_span * 0.04
    placed_y = []

    label_font_size = 11 if compact else 13
    line_width_base = 1.5 if compact else 1.8
    line_width_validated = 2.2 if compact else 2.8

    # Zones to draw = composite_zones UNION validated_zones, deduplicated
    # by price_mode. Drawing from composite_zones alone would miss cases
    # like the Sectors grid, which deliberately passes composite_zones=[]
    # (to avoid clutter at small size) and only passes validated_zones --
    # without this union, those validated levels would never render at all.
    _zones_by_key = {}
    for z in composite_zones:
        _zones_by_key[_zone_key(z)] = z
    for z in validated_zones:
        _zones_by_key.setdefault(_zone_key(z), z)
    zones_to_draw = sorted(_zones_by_key.values(), key=lambda z: z["price_mode"], reverse=True)
    for z in zones_to_draw:
        try:
            # Defensively coerce every numeric field to a plain Python
            # float up front. When one of these values arrives as a
            # numpy scalar, a 1-element array, or a pandas Series (this
            # has happened for at least BANKINDIA and FORTIS), a
            # comparison like `label_y != price_mode` stops returning a
            # plain bool and instead returns an array -- and Plotly's
            # `showarrow` validator then raises ValueError trying to
            # coerce that array to a bool, which crashed the WHOLE
            # chart over a single bad zone. Casting here makes every
            # downstream operation work on plain scalars again.
            price_mode = float(z["price_mode"])
            price_low = float(z["price_low"])
            price_high = float(z["price_high"])
        except (TypeError, ValueError):
            # Unusable price data on this one zone -- skip just this
            # zone instead of taking down the whole chart.
            continue

        try:
            is_resistance = price_mode >= last_close
            is_validated = _zone_key(z) in validated_keys

            if is_validated:
                fill = VALIDATED_RESISTANCE_FILL if is_resistance else VALIDATED_SUPPORT_FILL
                line_width = line_width_validated
                kind_label = "Resistance (confirmed)" if is_resistance else "Support (confirmed)"
            else:
                fill = RESISTANCE_FILL if is_resistance else SUPPORT_FILL
                line_width = line_width_base
                kind_label = "Resistance" if is_resistance else "Support"
            line_color = RESISTANCE_LINE if is_resistance else SUPPORT_LINE

            fig.add_hrect(
                y0=price_low, y1=price_high,
                fillcolor=fill, line_width=0, layer="below",
            )
            fig.add_shape(
                type="line", x0=x0, x1=x1, y0=price_mode, y1=price_mode,
                line=dict(color=line_color, width=line_width, dash="dash"),
            )

            ml_risk = (ml_risk_lookup or {}).get(_zone_key(z))
            ml_risk = float(ml_risk) if ml_risk is not None else None
            ml_suffix = f" - ML {ml_risk:.0f}%" if ml_risk is not None else ""

            if not compact:
                label_y = price_mode
                for py in placed_y:
                    if abs(label_y - py) < min_gap:
                        label_y = py - min_gap
                placed_y.append(label_y)

                fig.add_annotation(
                    x=x1, y=label_y,
                    text=f"{kind_label} {price_mode:.0f} ({_pct_from_label(z['label']):.0f}%){ml_suffix}",
                    showarrow=bool(label_y != price_mode),
                    arrowhead=0, arrowwidth=1, arrowcolor=line_color,
                    ax=45, ay=0,
                    xanchor="left", font=dict(size=label_font_size, color=line_color,
                                               family="Arial Black" if is_validated else "Arial"),
                    bgcolor="rgba(19, 23, 34, 0.9)",
                    bordercolor=line_color, borderwidth=2 if is_validated else 1, borderpad=3,
                )
            else:
                # Compact grid cells: a tiny price tag instead of the full
                # label+percent annotation, which would overwhelm a small chart.
                # ML risk still gets appended (short form) when available --
                # this is exactly the "visible on every sector chart" case.
                # xanchor="right" makes the box END at x1 and grow LEFTWARD
                # into the visible chart, instead of growing rightward past
                # the chart edge into the (too-narrow-for-this-text) margin.
                compact_ml = f" ML{ml_risk:.0f}%" if ml_risk is not None else ""
                fig.add_annotation(
                    x=x1, y=price_mode,
                    text=f"{price_mode:.0f}{compact_ml}",
                    showarrow=False, xanchor="right",
                    font=dict(size=label_font_size, color=line_color),
                    bgcolor="rgba(19, 23, 34, 0.85)",
                    bordercolor=line_color, borderwidth=1 if is_validated else 0, borderpad=1,
                )
        except Exception:
            # One malformed zone should never take down the whole chart --
            # skip just this zone's shape/annotation; every other zone
            # still renders normally.
            continue

    # marker for last close so it's obvious where "current price" sits
    fig.add_shape(
        type="line", x0=x0, x1=x1, y0=last_close, y1=last_close,
        line=dict(color=LTP_LINE, width=1, dash="solid"),
    )
    if not compact:
        fig.add_annotation(
            x=x0, y=last_close, text=f"LTP {last_close:.0f}",
            showarrow=False, xanchor="right", font=dict(size=12, color=LTP_LINE),
            bgcolor="rgba(19, 23, 34, 0.9)",
        )
    else:
        fig.add_annotation(
            x=x0, y=last_close, text=f"{last_close:.0f}",
            showarrow=False, xanchor="right", font=dict(size=11, color=LTP_LINE),
            bgcolor="rgba(19, 23, 34, 0.85)",
        )

    yaxis_config = dict(
        title="Price" if not compact else None,
        gridcolor=GRID_COLOR, gridwidth=1, showgrid=True,
        color=TEXT_COLOR, linecolor=GRID_COLOR, linewidth=1.5,
        tickfont=dict(size=13 if compact else 14),
    )
    if y_range is not None:
        yaxis_config["range"] = list(y_range)

    for ev in (event_markers or []):
        marker_color = ev.get("color", "#FFD54F")  # amber, distinct from support/resistance/VWAP/LTP
        fig.add_shape(
            type="line", xref="x", yref="paper",
            x0=ev["time"], x1=ev["time"], y0=0, y1=1,
            line=dict(color=marker_color, width=1.2, dash="dot"),
        )
        fig.add_annotation(
            x=ev["time"], y=1.0, yref="paper", yanchor="bottom",
            text=ev.get("label", ""), showarrow=False, textangle=-90 if compact else 0,
            font=dict(size=9 if compact else 10, color=marker_color),
            bgcolor="rgba(19, 23, 34, 0.85)",
        )

    xaxis_config = dict(
        title=None, gridcolor=GRID_COLOR, gridwidth=1, showgrid=True,
        rangeslider_visible=False, color=TEXT_COLOR, linecolor=GRID_COLOR, linewidth=1.5,
        range=[x0, x1],
        tickformat=tick_format or "%H:%M",
        tickfont=dict(size=12 if compact else 13),
    )
    if market_hours_breaks:
        # Hides 15:30 -> next day's 09:15 (overnight) and Sat/Mon (weekend)
        # so a multi-day intraday series stays visually compact instead of
        # mostly-blank-space with tiny clusters of real candles.
        xaxis_config["rangebreaks"] = [
            dict(bounds=["sat", "mon"]),
            dict(bounds=[15.5, 9.25], pattern="hour"),
        ]

    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=16 if not compact else 13), y=0.98),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        height=height,
        margin=(dict(l=45, r=55, t=35, b=25) if compact
                else dict(l=55, r=130, t=70, b=30)),
        showlegend=not compact,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.06,
            font=dict(color=TEXT_COLOR, size=10), bgcolor="rgba(0,0,0,0)",
        ) if not compact else None,
    )
    return fig


def build_synced_sector_grid(symbol_panels, cols=2, height_per_row=260):
    """
    Builds ONE combined figure containing every stock in a sector as a
    subplot grid (default 2 per row), with all x-axes linked together so
    hovering anywhere shows a synced vertical crosshair line across
    EVERY panel at once -- lets you see what the same moment in time
    looked like for every stock in the sector simultaneously.

    symbol_panels: list of dicts, each:
        {"symbol": str, "df": candle df (needs timestamp/open/high/low/close),
         "validated_zones": list of zone dicts (drawn bold/confirmed-style,
             same treatment as plot_candles_with_zones gives validated
             zones -- this grid only ever shows validated zones, same as
             the existing Sectors compact view),
         "x_range": (start, end) tuple to pin that panel's x-axis,
         "ml_risk_lookup": optional {rounded price_mode: risk_pct} dict}

    Returns one go.Figure. Caller renders it with a single st.plotly_chart.
    """
    from plotly.subplots import make_subplots

    n = len(symbol_panels)
    if n == 0:
        return go.Figure()
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[p["symbol"] for p in symbol_panels],
        vertical_spacing=min(0.12, 1.0 / max(rows, 1)),
        horizontal_spacing=0.06,
    )

    for i, panel in enumerate(symbol_panels):
        row = i // cols + 1
        col = i % cols + 1
        df = panel["df"]
        x_range = panel.get("x_range")
        x0, x1 = x_range if x_range is not None else (df["timestamp"].iloc[0], df["timestamp"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        ml_lookup = panel.get("ml_risk_lookup") or {}

        fig.add_trace(go.Candlestick(
            x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            increasing_line_color=CANDLE_UP, increasing_fillcolor=CANDLE_UP,
            decreasing_line_color=CANDLE_DOWN, decreasing_fillcolor=CANDLE_DOWN,
            showlegend=False, name=panel["symbol"],
        ), row=row, col=col)

        for z in (panel.get("validated_zones") or []):
            try:
                z_price_mode = float(z["price_mode"])
                z_price_low = float(z["price_low"])
                z_price_high = float(z["price_high"])
            except (TypeError, ValueError):
                continue

            try:
                is_resistance = z_price_mode >= last_close
                fill = VALIDATED_RESISTANCE_FILL if is_resistance else VALIDATED_SUPPORT_FILL
                line_color = RESISTANCE_LINE if is_resistance else SUPPORT_LINE

                fig.add_hrect(y0=z_price_low, y1=z_price_high, fillcolor=fill,
                              line_width=0, layer="below", row=row, col=col)
                fig.add_shape(type="line", x0=x0, x1=x1, y0=z_price_mode, y1=z_price_mode,
                              line=dict(color=line_color, width=1.8, dash="dash"), row=row, col=col)

                ml_risk = ml_lookup.get(round(z_price_mode, 2))
                ml_risk = float(ml_risk) if ml_risk is not None else None
                ml_suffix = f" ML{ml_risk:.0f}%" if ml_risk is not None else ""
                fig.add_annotation(
                    x=x1, y=z_price_mode, text=f"{z_price_mode:.0f}{ml_suffix}",
                    showarrow=False, xanchor="right", font=dict(size=10, color=line_color),
                    bgcolor="rgba(19, 23, 34, 0.85)", bordercolor=line_color, borderwidth=1, borderpad=1,
                    row=row, col=col,
                )
            except Exception:
                continue

        fig.add_shape(type="line", x0=x0, x1=x1, y0=last_close, y1=last_close,
                      line=dict(color=LTP_LINE, width=1, dash="solid"), row=row, col=col)

        fig.update_xaxes(range=[x0, x1], row=row, col=col)

    # Link every x-axis to the first one, and turn on cross-panel spike
    # lines -- this combination is what makes hovering on ANY panel draw
    # a synced vertical line on ALL panels at that same x position.
    fig.update_xaxes(
        matches="x", tickformat="%H:%M", gridcolor=GRID_COLOR, color=TEXT_COLOR,
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikethickness=1, spikecolor="#FFD54F", spikedash="dot",
    )
    fig.update_yaxes(gridcolor=GRID_COLOR, color=TEXT_COLOR)
    fig.update_annotations(font=dict(color=TEXT_COLOR, size=12))

    fig.update_layout(
        hovermode="x",
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR, font=dict(color=TEXT_COLOR),
        height=height_per_row * rows,
        showlegend=False,
        margin=dict(l=50, r=50, t=40, b=20),
    )
    return fig


def compute_cumulative_volume_delta(df):
    """Approximates volume delta per candle from plain OHLCV data --
    Upstox's REST candle API doesn't provide bid/ask-tagged tick data,
    so this uses the standard candle-color approximation instead of
    true order-flow delta: an up-close candle counts its full volume as
    buying pressure, a down-close candle as selling pressure, a flat
    candle as neutral. Returns the running cumulative sum across the
    session. This is a widely-used proxy when tick data isn't
    available, but it is an approximation, not tick-accurate CVD."""
    delta = np.where(df["close"] > df["open"], df["volume"],
                      np.where(df["close"] < df["open"], -df["volume"], 0))
    return pd.Series(delta, index=df.index).cumsum()


def compute_recent_order_flow_imbalance_pct(df, lookback=6):
    """Normalized order-flow confirmation over the last `lookback` candles:
    (sum of signed volume-delta) / (sum of raw volume) * 100 -- reads as
    a %-of-recent-volume net buying (positive) or net selling (negative)
    pressure.

    This is deliberately a DIFFERENT shape of number than
    compute_cumulative_volume_delta's running total: the cumulative sum
    scales with however much volume a stock has traded since the open
    (a high-volume large-cap's cumulative CVD dwarfs a lower-volume
    stock's, even with identical recent buying pressure), so it isn't
    comparable zone-to-zone or symbol-to-symbol. Normalizing by recent
    volume makes it directly comparable across stocks and directly
    usable as one input into a weighted composite score (see
    zone_validation.compute_cvd_zone_signal), the same way RVOL% already
    normalizes volume itself for cross-symbol comparison elsewhere in
    this app.

    Uses the SAME up/down-close approximation as
    compute_cumulative_volume_delta (see its docstring for the
    tick-data caveat -- this is not true bid/ask order flow).

    Returns None if there isn't enough data or the recent window has
    zero volume (can't normalize by zero)."""
    if df is None or df.empty or len(df) < 2:
        return None
    recent = df.tail(min(lookback, len(df)))
    total_vol = recent["volume"].sum()
    if total_vol <= 0:
        return None
    delta = np.where(recent["close"] > recent["open"], recent["volume"],
                      np.where(recent["close"] < recent["open"], -recent["volume"], 0))
    return float(delta.sum() / total_vol * 100)


def build_cvd_chart(df, height=100, compact=False, x_range=None):
    """A small bar chart of cumulative volume delta, meant to sit
    directly below a price chart from plot_candles_with_zones -- same
    time range, same color language (teal=net buying, red=net
    selling), so it reads as a companion panel even though it's a
    separate figure rather than a true shared-axis subplot.

    x_range should be the SAME value passed to plot_candles_with_zones
    for the price chart above it -- without this, this chart's x-axis
    auto-scales to just the candles that exist so far, while the price
    chart above is pinned to the full session, making these bars look
    stretched wider than the actual candles they correspond to."""
    if df is None or df.empty:
        return go.Figure()
    cvd = compute_cumulative_volume_delta(df)
    colors = [CANDLE_UP if v >= 0 else CANDLE_DOWN for v in cvd]

    # explicit bar width matching the actual candle interval, so bars
    # never render wider than their corresponding candle regardless of
    # how the x-axis range is scaled
    if len(df) > 1:
        interval_ms = df["timestamp"].diff().dropna().dt.total_seconds().median() * 1000
        bar_width_ms = interval_ms * 0.8
    else:
        bar_width_ms = None

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["timestamp"], y=cvd, marker_color=colors, showlegend=False,
                          width=bar_width_ms))
    xaxis_config = dict(showticklabels=not compact, gridcolor=GRID_COLOR, showgrid=not compact)
    if x_range is not None:
        xaxis_config["range"] = list(x_range)
    fig.update_layout(
        height=height,
        # left/right margins must match plot_candles_with_zones' EXACTLY
        # (compact: l=45,r=55 / non-compact: l=55,r=130) -- these are two
        # separate figures stacked visually, not true shared-axis
        # subplots, so the same timestamp only lands at the same pixel
        # position in both if their plot areas start/end at the same
        # horizontal offset. Top/bottom margins don't affect this and
        # can differ freely.
        margin=(dict(l=45, r=55, t=10, b=10) if compact
                else dict(l=55, r=130, t=10, b=20)),
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, size=9 if compact else 11),
        xaxis=xaxis_config,
        yaxis=dict(title=None if compact else "Cum. Vol Delta", gridcolor=GRID_COLOR),
    )
    return fig
