#!/usr/bin/env python3
"""TA Chart Agent v2 — daily technical-analysis briefing.

For each configured ticker this script:
  1. fetches daily OHLCV history from Yahoo Finance,
  2. computes SMA, RSI(14) and MACD(12,26,9),
  3. renders a four-panel chart (candles+SMAs / volume / RSI / MACD),
  4. composes an HTML email with a signal summary table and the charts inline,
  5. sends it over SMTP — unless --dry-run is given, in which case the email
     is written to <output_dir>/email_preview.html and nothing is sent.

The SMTP password is read from the TA_SMTP_PASSWORD environment variable and
is never stored in config.yaml.
"""

import argparse
import json
import os
import smtplib
import sys
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import yaml
import yfinance as yf

EASTERN = ZoneInfo("America/New_York")


def log(msg: str) -> None:
    print(f"[{datetime.now(EASTERN):%Y-%m-%d %H:%M:%S %Z}] {msg}", flush=True)


def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def fetch_history(ticker: str, lookback_days: int) -> pd.DataFrame:
    start = datetime.now(EASTERN) - timedelta(days=lookback_days)
    df = yf.Ticker(ticker).history(start=start.date(), interval="1d", auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"no price data returned for {ticker}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = df.index.tz_localize(None)
    return df


def add_indicators(df: pd.DataFrame, ind: dict) -> pd.DataFrame:
    close = df["Close"]
    df[f"SMA{ind['sma_fast']}"] = close.rolling(ind["sma_fast"]).mean()
    df[f"SMA{ind['sma_slow']}"] = close.rolling(ind["sma_slow"]).mean()
    if ind.get("sma_long"):
        df[f"SMA{ind['sma_long']}"] = close.rolling(ind["sma_long"]).mean()

    # Wilder's RSI
    period = ind["rsi_period"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)

    ema_fast = close.ewm(span=ind["macd_fast"], adjust=False).mean()
    ema_slow = close.ewm(span=ind["macd_slow"], adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACDSignal"] = df["MACD"].ewm(span=ind["macd_signal"], adjust=False).mean()
    df["MACDHist"] = df["MACD"] - df["MACDSignal"]
    return df


FIB_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)


def fib_levels(plot_df: pd.DataFrame) -> dict[float, float]:
    """Fibonacci retracement levels for the displayed window.

    Anchored on the window's swing high and swing low. If the high came after
    the low (up-leg) the retracements step down from the high; otherwise
    (down-leg) they step up from the low.
    """
    hi, lo = plot_df["High"].max(), plot_df["Low"].min()
    if plot_df["High"].idxmax() >= plot_df["Low"].idxmin():
        return {r: hi - (hi - lo) * r for r in FIB_RATIOS}
    return {r: lo + (hi - lo) * r for r in FIB_RATIOS}


def find_pivots(vals: np.ndarray, kind: str, order: int = 5) -> list[int]:
    """Indices of local swing highs ('high') or swing lows ('low')."""
    pivots = []
    for k in range(order, len(vals) - order):
        window = vals[k - order : k + order + 1]
        if kind == "high" and vals[k] >= window.max():
            pivots.append(k)
        elif kind == "low" and vals[k] <= window.min():
            pivots.append(k)
    return pivots


def detect_trendline(plot_df: pd.DataFrame, kind: str) -> tuple[int, float, float] | None:
    """Best-fitting resistance ('res') or support ('sup') trendline, if any.

    Tries lines through every pair of swing pivots and keeps the one with the
    most pivot touches that price has respected from the first anchor through
    the latest bar. Returns (start_index, start_price, end_price) where the
    end is the line's value at the last bar, or None when nothing qualifies.
    """
    vals = (plot_df["High"] if kind == "res" else plot_df["Low"]).to_numpy()
    n = len(vals)
    pivots = find_pivots(vals, "high" if kind == "res" else "low")
    if len(pivots) < 2:
        return None

    tol = (vals.max() - vals.min()) * 0.01
    last_close = float(plot_df["Close"].iloc[-1])
    best, best_score = None, (0, 0)
    for a, i in enumerate(pivots):
        for j in pivots[a + 1 :]:
            if j - i < n // 6:
                continue
            slope = (vals[j] - vals[i]) / (j - i)
            line = vals[i] + slope * (np.arange(i, n) - i)
            overshoot = vals[i:] - line if kind == "res" else line - vals[i:]
            if overshoot.max() > tol * 0.5:  # price broke through the line
                continue
            touch_idx = [
                p for p in pivots
                if p >= i and abs(vals[p] - (vals[i] + slope * (p - i))) < tol
            ]
            # the line must still be in play: touched recently and projecting
            # to somewhere near the current price, not drifting off-screen
            if (
                len(touch_idx) < 2
                or touch_idx[-1] < n - n // 3
                or abs(line[-1] - last_close) > last_close * 0.10
            ):
                continue
            score = (len(touch_idx), n - i)  # most touches, then earliest anchor
            if score > best_score:
                best_score = score
                best = (i, float(vals[i]), float(line[-1]))
    return best


def render_chart(ticker: str, df: pd.DataFrame, ind: dict, chart_days: int, out_dir: Path) -> Path:
    plot_df = df.dropna(subset=[f"SMA{ind['sma_slow']}", "RSI"]).tail(chart_days)
    if plot_df.empty:
        raise RuntimeError(f"{ticker}: not enough history to plot (increase lookback_days)")

    fast, slow = ind["sma_fast"], ind["sma_slow"]
    hist_colors = ["#2ebd85" if v >= 0 else "#f6465d" for v in plot_df["MACDHist"]]
    rsi_band = lambda level: pd.Series(level, index=plot_df.index)

    addplots = [
        mpf.make_addplot(plot_df[f"SMA{fast}"], panel=0, color="#e07ae0", width=1.0, label=f"SMA {fast}"),
        mpf.make_addplot(plot_df[f"SMA{slow}"], panel=0, color="#ffa726", width=1.0, label=f"SMA {slow}"),
        mpf.make_addplot(plot_df["RSI"], panel=2, color="#b39ddb", width=1.0, ylabel="RSI", ylim=(0, 100)),
        mpf.make_addplot(rsi_band(70), panel=2, color="#787b86", width=0.7, linestyle="--"),
        mpf.make_addplot(rsi_band(30), panel=2, color="#787b86", width=0.7, linestyle="--"),
        mpf.make_addplot(plot_df["MACD"], panel=3, color="#42a5f5", width=1.0, ylabel="MACD"),
        mpf.make_addplot(plot_df["MACDSignal"], panel=3, color="#ffa726", width=1.0),
        mpf.make_addplot(plot_df["MACDHist"], panel=3, type="bar", color=hist_colors, alpha=0.6),
    ]
    long = ind.get("sma_long")
    if long and f"SMA{long}" in plot_df and plot_df[f"SMA{long}"].notna().any():
        addplots.insert(
            2, mpf.make_addplot(plot_df[f"SMA{long}"], panel=0, color="#cdb24c", width=1.2, label=f"SMA {long}")
        )

    fib = fib_levels(plot_df) if ind.get("fibonacci", True) else {}

    # Auto-drawn trendlines: descending/ascending resistance through swing
    # highs and support through swing lows, only where price respected them.
    tline_segs, tline_colors = [], []
    if ind.get("trendlines", True):
        for kind, color in (("res", "#c478f0"), ("sup", "#4aa8ff")):
            tl = detect_trendline(plot_df, kind)
            if tl:
                i, y0, y1 = tl
                tline_segs.append([(plot_df.index[i], y0), (plot_df.index[-1], y1)])
                tline_colors.append(color)

    out_path = out_dir / f"{ticker.replace('^', '').replace('=', '_')}_4panel.png"
    last = plot_df.iloc[-1]
    title = f"\n{ticker}  {plot_df.index[-1]:%Y-%m-%d}  close {last['Close']:,.2f}"
    marketcolors = mpf.make_marketcolors(
        up="#2ebd85", down="#f6465d", edge="inherit", wick="inherit", volume="inherit"
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=marketcolors,
        facecolor="#131722",
        figcolor="#131722",
        edgecolor="#2a2e39",
        gridcolor="#2a2e39",
        gridstyle=":",
        rc={
            "font.size": 9,
            "text.color": "#d1d4dc",
            "axes.labelcolor": "#d1d4dc",
            "xtick.color": "#d1d4dc",
            "ytick.color": "#d1d4dc",
        },
    )
    line_kwargs = {}
    if fib:
        line_kwargs["hlines"] = dict(
            hlines=list(fib.values()),
            colors=["#e3b341"] * len(fib),
            linestyle="--",
            linewidths=0.7,
            alpha=0.5,
        )
    if tline_segs:
        line_kwargs["alines"] = dict(
            alines=tline_segs, colors=tline_colors, linewidths=1.3, alpha=0.9
        )
    fig, axes = mpf.plot(
        plot_df,
        type="candle",
        style=style,
        volume=True,
        addplot=addplots,
        panel_ratios=(6, 2, 2, 2),
        figsize=(12, 10),
        title=title,
        tight_layout=True,
        returnfig=True,
        **line_kwargs,
    )
    price_ax = axes[0]
    for ratio, level in fib.items():
        price_ax.text(
            0.995, level, f"{ratio:.1%}  {level:,.2f}",
            transform=price_ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color="#e3b341",
        )
    legend = price_ax.legend(loc="upper left", fontsize=8)
    legend.get_frame().set_alpha(0.3)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# Full index membership sources, keyed by the ETF that tracks the index.
# Yahoo only exposes a fund's top ~10 holdings, so broad indexes are resolved
# from these membership tables instead. Each entry is tried in order.
INDEX_SOURCES = {
    "SPY": [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"),
        ("https://stockanalysis.com/list/sp-500-stocks/", "Symbol"),
    ],
    "QQQ": [
        ("https://stockanalysis.com/list/nasdaq-100-stocks/", "Symbol"),
        ("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker"),
    ],
}


def index_members(etf: str) -> list[str]:
    """Full membership of the index tracked by `etf` (SPY/QQQ)."""
    import io

    import requests

    errors = []
    for url, column in INDEX_SOURCES[etf]:
        try:
            resp = requests.get(  # some sources 403 the default python agent
                url,
                headers={"User-Agent": "Mozilla/5.0 (TA-Chart-Agent briefing bot)"},
                timeout=30,
            )
            resp.raise_for_status()
            for table in pd.read_html(io.StringIO(resp.text)):
                if column in table.columns and len(table) > 50:
                    symbols = [
                        str(s).strip().replace(".", "-")  # BRK.B -> BRK-B (Yahoo style)
                        for s in table[column].tolist()
                    ]
                    return [s for s in symbols if s and s.lower() != "nan"]
            errors.append(f"{url}: no '{column}' table")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("; ".join(errors))


def resolve_universe(scan_cfg: dict) -> list[str]:
    """Build the scan universe.

    Union of: the configured ETFs' published top holdings (Yahoo exposes
    roughly ten per fund; non-US listings like 005930.KS are skipped), the
    full membership of the configured index universes (S&P 500 / Nasdaq-100
    via Wikipedia), and any extra_tickers. Falls back to the static
    `universe` list when nothing can be fetched.
    """
    tickers: dict[str, None] = {}  # insertion-ordered de-dup
    for etf in scan_cfg.get("etf_holdings", []):
        try:
            holdings = yf.Ticker(etf).funds_data.top_holdings
            symbols = [s for s in holdings.index if "." not in s]
            log(f"{etf}: {len(symbols)} US-listed holdings: {', '.join(symbols)}")
            tickers.update(dict.fromkeys(symbols))
        except Exception as exc:
            log(f"{etf}: holdings fetch failed — {exc}")
    for etf in scan_cfg.get("index_universes", []):
        try:
            members = index_members(etf)
            log(f"{etf}: {len(members)} index members")
            tickers.update(dict.fromkeys(members))
        except Exception as exc:
            log(f"{etf}: index membership fetch failed — {exc}")
    tickers.update(dict.fromkeys(scan_cfg.get("extra_tickers", [])))
    if not tickers:
        log("no ETF holdings resolved; falling back to the static universe")
        tickers.update(dict.fromkeys(scan_cfg.get("universe", [])))
    return list(tickers)


def bullish_score(df: pd.DataFrame, ind: dict) -> float:
    """Composite bullishness score (roughly 0-12, higher is more bullish).

    Rewards trend alignment (close > SMA fast > SMA slow, above SMA long),
    bullish MACD with a rising histogram, RSI in the 50-70 momentum zone
    (penalizing overbought >75), and 20-day momentum capped at +/-3.
    """
    last = df.iloc[-1]
    fast, slow = f"SMA{ind['sma_fast']}", f"SMA{ind['sma_slow']}"
    score = 0.0
    if last["Close"] > last[fast]:
        score += 2
    if last[fast] > last[slow]:
        score += 2
    long = ind.get("sma_long")
    if long and pd.notna(last.get(f"SMA{long}")) and last["Close"] > last[f"SMA{long}"]:
        score += 1
    if last["MACD"] > last["MACDSignal"]:
        score += 2
    hist = df["MACDHist"].tail(3)
    if len(hist) == 3 and hist.is_monotonic_increasing:
        score += 1
    rsi = last["RSI"]
    if 50 <= rsi <= 70:
        score += 1
    elif rsi > 75:
        score -= 1
    if len(df) > 21:
        ret20 = (last["Close"] / df["Close"].iloc[-21] - 1) * 100
        score += max(-3.0, min(3.0, ret20 * 0.1))
    return round(score, 2)


def support_level(df: pd.DataFrame, ind: dict) -> float | None:
    """Nearest meaningful support below the last close.

    Considers the moving averages, the Fibonacci retracement levels of the
    recent window, and the latest swing low — support is the highest of those
    sitting below price, i.e. the first floor the bulls need to defend.
    """
    last_close = float(df["Close"].iloc[-1])
    candidates = []
    for n in (ind["sma_fast"], ind["sma_slow"], ind.get("sma_long")):
        col = f"SMA{n}"
        if n and col in df and pd.notna(df[col].iloc[-1]) and df[col].iloc[-1] < last_close:
            candidates.append(float(df[col].iloc[-1]))
    window = df.tail(180)
    candidates += [lvl for lvl in fib_levels(window).values() if lvl < last_close]
    lows = window["Low"].to_numpy()
    pivots = find_pivots(lows, "low")
    if pivots and lows[pivots[-1]] < last_close:
        candidates.append(float(lows[pivots[-1]]))
    return max(candidates) if candidates else None


def fetch_research_notes(cfg: dict) -> list[dict]:
    """Recent research newsletters pulled from the Gmail inbox via IMAP.

    Uses the same account and app password as sending. The Gmail search query
    (config: research.gmail_search) selects the source — e.g. TMT Breakout
    issues — and the excerpts are handed to Claude as analysis context.
    Failures are logged and skipped; the briefing never blocks on this.
    """
    import html as html_lib
    import imaplib
    import re
    from email import message_from_bytes, policy

    research = cfg.get("research") or {}
    if not research.get("enabled"):
        return []
    password = os.environ.get("TA_SMTP_PASSWORD")
    if not password:
        log("research: TA_SMTP_PASSWORD not set; skipping inbox research")
        return []

    query = research.get("gmail_search", "TMT Breakout newer_than:10d")
    max_items = research.get("max_items", 3)
    max_chars = research.get("max_chars", 4000)
    notes = []
    try:
        imap = imaplib.IMAP4_SSL(research.get("imap_host", "imap.gmail.com"), 993)
        imap.login(cfg["email"]["username"], password)
        imap.select('"[Gmail]/All Mail"', readonly=True)
        _, data = imap.search(None, "X-GM-RAW", f'"{query}"')
        ids = (data[0] or b"").split()
        for mid in reversed(ids[-max_items:]):  # newest first
            _, msg_data = imap.fetch(mid, "(RFC822)")
            msg = message_from_bytes(msg_data[0][1], policy=policy.default)
            body = msg.get_body(preferencelist=("plain", "html"))
            text = body.get_content() if body else ""
            if body is not None and body.get_content_type() == "text/html":
                text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
                text = html_lib.unescape(re.sub(r"<[^>]+>", " ", text))
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                notes.append({
                    "subject": str(msg["Subject"]),
                    "date": str(msg["Date"]),
                    "excerpt": text[:max_chars],
                })
        imap.logout()
        log(f"research: loaded {len(notes)} note(s) matching '{query}'")
    except Exception as exc:
        log(f"research: inbox fetch failed — {exc}")
    return notes


def fetch_news(ticker: str, limit: int = 3) -> list[dict]:
    """Recent headlines from Yahoo Finance — potential catalysts for the move."""
    items = []
    try:
        for raw in (yf.Ticker(ticker).news or [])[:limit]:
            content = raw.get("content") or {}
            title = content.get("title")
            if not title:
                continue
            items.append({
                "title": title,
                "date": (content.get("pubDate") or "")[:10],
                "source": (content.get("provider") or {}).get("displayName", ""),
                "summary": (content.get("summary") or "")[:300],
            })
    except Exception as exc:
        log(f"{ticker}: news fetch failed — {exc}")
    return items


def technical_commentary(df: pd.DataFrame, ind: dict, support: float | None = None) -> str:
    """Deterministic commentary on what is driving the bullish setup."""
    last = df.iloc[-1]
    close = df["Close"]
    bits = []

    above = [
        str(n) for n in (ind["sma_fast"], ind["sma_slow"], ind.get("sma_long"))
        if n and pd.notna(last.get(f"SMA{n}")) and last["Close"] > last[f"SMA{n}"]
    ]
    if above:
        bits.append(f"price is holding above its {'/'.join(above)}-day moving averages")

    bull_streak = 0
    for positive in reversed((df["MACDHist"] > 0).tolist()):
        if not positive:
            break
        bull_streak += 1
    if bull_streak:
        bits.append(f"MACD has been in a bullish crossover for {bull_streak} session(s)")

    rsi = last["RSI"]
    if rsi > 70:
        bits.append(f"RSI at {rsi:.0f} signals strong momentum, though it is in overbought territory")
    elif rsi >= 50:
        bits.append(f"RSI at {rsi:.0f} shows momentum with room before overbought")

    if len(df) > 21:
        ret20 = (last["Close"] / close.iloc[-21] - 1) * 100
        if abs(ret20) >= 1:
            bits.append(f"the stock is {'up' if ret20 > 0 else 'down'} {abs(ret20):.1f}% over the past month")

    window_high = df["High"].tail(120).max()
    if last["Close"] >= window_high * 0.97:
        bits.append("price is within 3% of its recent high")

    vol_recent, vol_base = df["Volume"].tail(5).mean(), df["Volume"].tail(50).mean()
    if vol_base and vol_recent > vol_base * 1.3:
        bits.append(f"recent volume is running {vol_recent / vol_base - 1:.0%} above its 50-day average")

    if not bits:
        text = "Mixed technical picture; see chart for details."
    else:
        text = "; ".join(bits)
        text = text[0].upper() + text[1:] + "."
    if support:
        text += (
            f" Actionable: the bullish setup holds while price stays above support near"
            f" {support:,.2f} — a decisive close below that level would break the trend."
        )
    return text


def llm_commentary(picks: list[dict], research_notes: list[dict] | None = None) -> dict[str, str] | None:
    """Claude-written driver commentary, when ANTHROPIC_API_KEY is configured.

    Returns {ticker: commentary} or None, in which case the caller keeps the
    deterministic technical commentary.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        log("anthropic package not installed; using technical commentary")
        return None
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "commentary": {"type": "string"},
                    },
                    "required": ["ticker", "commentary"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=(
                "You write the commentary section of a daily technical-analysis "
                "briefing email. For each ticker, write 2-3 sentences on the "
                "plausible drivers of its bullish price action, weaving together "
                "the technical signals and the recent headlines provided. Then "
                "end with one actionable sentence using the provided "
                "support_level: frame the setup as constructive while price "
                "holds above that support (entries on strength or on pullbacks "
                "toward it), and state explicitly that a decisive close below "
                "the support level breaks the bullish trend. Use the exact "
                "support number provided. The input may include research_notes "
                "— excerpts from newsletters the user subscribes to (e.g. TMT "
                "Breakout). Treat them as an analyst's view: when a note "
                "discusses one of the tickers or a clearly relevant theme, "
                "weave that perspective in with attribution (e.g. 'TMT "
                "Breakout flags...'); silently ignore notes irrelevant to a "
                "pick. Ground every claim ONLY in the provided data — do not "
                "invent facts, numbers, or events. These are levels to watch, "
                "not personalized financial advice. Plain prose, no preamble, "
                "no hype words."
            ),
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{
                "role": "user",
                "content": json.dumps(
                    {"picks": picks, "research_notes": research_notes or []}
                ),
            }],
        )
        text = next(b.text for b in response.content if b.type == "text")
        result = {item["ticker"]: item["commentary"] for item in json.loads(text)["items"]}
        log(f"Claude commentary generated for {len(result)} tickers")
        return result
    except Exception as exc:
        log(f"Claude commentary unavailable, using technical commentary — {exc}")
        return None


def summarize(ticker: str, df: pd.DataFrame, ind: dict) -> dict:
    last, prev = df.iloc[-1], df.iloc[-2]
    slow = f"SMA{ind['sma_slow']}"
    rsi = last["RSI"]
    return {
        "ticker": ticker,
        "date": f"{df.index[-1]:%Y-%m-%d}",
        "close": f"{last['Close']:,.2f}",
        "change_pct": (last["Close"] / prev["Close"] - 1) * 100,
        "rsi": f"{rsi:.1f}",
        "rsi_state": "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral",
        "macd_state": "bullish" if last["MACD"] > last["MACDSignal"] else "bearish",
        "trend": "above" if last["Close"] > last[slow] else "below",
        "slow_name": slow,
    }


def build_html(summaries: list[dict], cids: dict[str, str], generated: str, heading: str) -> str:
    has_score = "score" in summaries[0]
    rows = []
    for s in summaries:
        chg = s["change_pct"]
        color = "#2e7d32" if chg >= 0 else "#c62828"
        score_cell = f"<td><b>{s['score']}</b></td>" if has_score else ""
        if has_score:
            support = s.get("support")
            score_cell += f"<td>{support:,.2f}</td>" if support else "<td>—</td>"
        rows.append(
            f"<tr><td><b>{s['ticker']}</b></td>{score_cell}<td>{s['close']}</td>"
            f"<td style='color:{color}'>{chg:+.2f}%</td>"
            f"<td>{s['rsi']} ({s['rsi_state']})</td><td>{s['macd_state']}</td>"
            f"<td>{s['trend']} {s['slow_name']}</td></tr>"
        )
    sections = []
    for s in summaries:
        ticker = s["ticker"]
        heading = ticker + (f" — bullish score {s['score']}" if has_score else "")
        block = f"<h3 style='font-family:sans-serif'>{heading}</h3>"
        if s.get("commentary"):
            block += (
                "<p style='max-width:860px;font-size:14px'>"
                f"<b>Potential drivers:</b> {s['commentary']}</p>"
            )
        if s.get("news"):
            headlines = "".join(
                f"<li>{n['title']} <span style='color:#777'>({n['source']}, {n['date']})</span></li>"
                for n in s["news"]
            )
            block += f"<ul style='font-size:13px;max-width:860px'>{headlines}</ul>"
        if ticker in cids:
            block += f"<img src='cid:{cids[ticker]}' width='860' style='max-width:100%'/>"
        sections.append(block)
    charts = "".join(sections)
    score_head = "<th>Score</th><th>Support</th>" if has_score else ""
    return f"""<html><body style="font-family:sans-serif">
<h2>{heading} — {summaries[0]['date']}</h2>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-size:14px">
<tr style="background:#eceff1"><th>Ticker</th>{score_head}<th>Close</th><th>1d %</th>
<th>RSI(14)</th><th>MACD</th><th>Trend</th></tr>
{''.join(rows)}
</table>
{charts}
<p style="color:#777;font-size:12px">Generated {generated} by TA Chart Agent v2.
Not investment advice.</p>
</body></html>"""


def build_email(
    cfg: dict, summaries: list[dict], chart_paths: dict[str, Path], heading: str
) -> EmailMessage:
    email_cfg = cfg["email"]
    msg = EmailMessage()
    msg["Subject"] = f"{heading} — {datetime.now(EASTERN):%Y-%m-%d}"
    msg["From"] = email_cfg["from_addr"]
    msg["To"] = ", ".join(email_cfg["to_addrs"])
    msg.set_content("Your email client does not support HTML. See attached charts.")

    cids = {t: make_msgid(domain="ta-chart-agent")[1:-1] for t in chart_paths}
    generated = f"{datetime.now(EASTERN):%Y-%m-%d %H:%M %Z}"
    msg.add_alternative(build_html(summaries, cids, generated, heading), subtype="html")
    for ticker, path in chart_paths.items():
        msg.get_payload()[1].add_related(
            path.read_bytes(), maintype="image", subtype="png", cid=f"<{cids[ticker]}>"
        )
    return msg


def send_email(cfg: dict, msg: EmailMessage) -> None:
    email_cfg = cfg["email"]
    password = os.environ.get("TA_SMTP_PASSWORD")
    if not password:
        raise RuntimeError(
            "TA_SMTP_PASSWORD is not set. Export it (or add it as a GitHub "
            "Actions secret) — see README.md."
        )
    host, port = email_cfg["smtp_host"], email_cfg["smtp_port"]
    if email_cfg.get("use_ssl"):
        server = smtplib.SMTP_SSL(host, port, timeout=30)
    else:
        server = smtplib.SMTP(host, port, timeout=30)
        server.starttls()
    with server:
        server.login(email_cfg["username"], password)
        server.send_message(msg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and email a daily TA briefing.")
    parser.add_argument("--config", default="config.yaml", type=Path)
    parser.add_argument("--tickers", nargs="+", help="override the configured ticker list")
    parser.add_argument(
        "--scan",
        action="store_true",
        help="scan the configured universe and email the most bullish charts",
    )
    parser.add_argument(
        "--heading",
        help="override the email subject/heading (e.g. 'Claude TA Morning Recap')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch data and render charts + email preview, but send nothing",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ind = cfg["indicators"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scan:
        heading = args.heading or "Claude TA Afternoon Recap"
        scan_cfg = cfg["scan"]
        universe = args.tickers or resolve_universe(scan_cfg)
        log(f"scanning {len(universe)} tickers for the {scan_cfg.get('top_n', 5)} most bullish...")
        scored = []
        for ticker in universe:
            try:
                df = add_indicators(fetch_history(ticker, cfg["lookback_days"]), ind)
                score = bullish_score(df, ind)
                scored.append((score, ticker, df))
                log(f"{ticker}: score {score}")
            except Exception as exc:
                log(f"{ticker}: skipped — {exc}")
        scored.sort(key=lambda item: item[0], reverse=True)
        picks = scored[: scan_cfg.get("top_n", 5)]
        log("top picks: " + ", ".join(f"{t} ({s})" for s, t, _ in picks))
        candidates = [(ticker, df, score) for score, ticker, df in picks]
    else:
        heading = args.heading or "Claude TA Morning Recap"
        candidates = [(t, None, None) for t in (args.tickers or cfg["tickers"])]

    summaries, chart_paths, failures = [], {}, []
    for ticker, df, score in candidates:
        try:
            if df is None:
                log(f"{ticker}: fetching {cfg['lookback_days']}d of history...")
                df = add_indicators(fetch_history(ticker, cfg["lookback_days"]), ind)
            chart_paths[ticker] = render_chart(ticker, df, ind, cfg["chart_days"], out_dir)
            summary = summarize(ticker, df, ind)
            if score is not None:
                summary["score"] = score
                summary["news"] = fetch_news(ticker)
                summary["support"] = support_level(df, ind)
                summary["commentary"] = technical_commentary(df, ind, summary["support"])
            summaries.append(summary)
            log(f"{ticker}: chart written to {chart_paths[ticker]}")
        except Exception as exc:  # one bad ticker must not kill the briefing
            failures.append(ticker)
            log(f"{ticker}: FAILED — {exc}")

    if not summaries:
        log("no tickers succeeded; aborting")
        return 1

    if args.scan:
        # Upgrade the deterministic commentary to Claude-written prose when an
        # API key is configured; otherwise keep the technical version.
        picks_context = [
            {
                "ticker": s["ticker"],
                "bullish_score": s.get("score"),
                "last_close": s["close"],
                "support_level": s.get("support"),
                "technicals": s.get("commentary"),
                "headlines": s.get("news", []),
            }
            for s in summaries
        ]
        research_notes = fetch_research_notes(cfg)
        for ticker, text in (llm_commentary(picks_context, research_notes) or {}).items():
            for s in summaries:
                if s["ticker"] == ticker:
                    s["commentary"] = text

    msg = build_email(cfg, summaries, chart_paths, heading)
    if args.dry_run or not cfg["email"].get("enabled", True):
        # Browser-viewable preview: same HTML, but images point at the local
        # PNGs instead of cid: attachments.
        preview = out_dir / "email_preview.html"
        cids = {t: p.name for t, p in chart_paths.items()}
        generated = f"{datetime.now(EASTERN):%Y-%m-%d %H:%M %Z}"
        preview.write_text(build_html(summaries, cids, generated, heading).replace("cid:", ""))
        log(f"DRY RUN — email not sent. Preview: {preview}")
        log(f"DRY RUN — would send to: {', '.join(cfg['email']['to_addrs'])} "
            f"with subject '{msg['Subject']}'")
    else:
        send_email(cfg, msg)
        log(f"email sent to {', '.join(cfg['email']['to_addrs'])}")

    if failures:
        log(f"completed with failures: {', '.join(failures)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
