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
    """Composite bullishness score on a 1-5 scale (5 = most bullish).

    Rewards trend alignment (close > SMA fast > SMA slow, above SMA long),
    bullish MACD with a rising histogram, RSI in the 50-70 momentum zone
    (penalizing overbought >75), and 20-day momentum. The raw signal sum
    spans roughly -4..12 and is mapped linearly onto 1..5.
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
    return round(min(5.0, max(1.0, 1 + (score + 4) / 4)), 1)


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


SECTOR_ETFS = {
    "XLK": "Tech", "XLC": "Comm Svcs", "XLY": "Cons Disc", "XLP": "Staples",
    "XLF": "Financials", "XLV": "Health Care", "XLI": "Industrials",
    "XLE": "Energy", "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real Estate",
}


def market_overview() -> dict:
    """Daily and 1-month moves for SPY, QQQ, and the S&P sector ETFs."""
    stats, frames = {}, {}
    for sym in ("SPY", "QQQ", "^VIX", *SECTOR_ETFS):
        try:
            df = fetch_history(sym, 90)
            close = df["Close"]
            stats[sym] = {
                "name": "VIX" if sym == "^VIX" else SECTOR_ETFS.get(sym, sym),
                "day_pct": round((close.iloc[-1] / close.iloc[-2] - 1) * 100, 2),
                "month_pct": round((close.iloc[-1] / close.iloc[-22] - 1) * 100, 2)
                if len(close) > 22 else None,
            }
            if sym == "^VIX":
                stats[sym]["level"] = round(float(close.iloc[-1]), 1)
            frames[sym] = df
        except Exception as exc:
            log(f"{sym}: market overview fetch failed — {exc}")
    return {"stats": stats, "frames": frames}


def render_market_chart(overview: dict, out_dir: Path) -> Path | None:
    """Two-panel bottom-of-email graph: SPY/QQQ month path + sector day moves."""
    frames, stats = overview["frames"], overview["stats"]
    if "SPY" not in frames or "QQQ" not in frames:
        return None
    fg, grid = "#d1d4dc", "#2a2e39"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="#131722")
    for ax in (ax1, ax2):
        ax.set_facecolor("#131722")
        ax.tick_params(colors=fg, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(grid)
        ax.grid(color=grid, linestyle=":", linewidth=0.5)

    for sym, color in (("SPY", "#4aa8ff"), ("QQQ", "#e07ae0")):
        close = frames[sym]["Close"].tail(22)
        ax1.plot(close.index, (close / close.iloc[0] - 1) * 100, label=sym, color=color, lw=1.4)
    ax1.axhline(0, color="#787b86", lw=0.7)
    ax1.set_title("SPY vs QQQ — past month (%)", color=fg, fontsize=10)
    ax1.legend(facecolor="#131722", edgecolor=grid, labelcolor=fg, fontsize=8)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
    for label in ax1.get_xticklabels():
        label.set_rotation(20)

    sectors = sorted(
        ((v["name"], v["day_pct"]) for k, v in stats.items() if k in SECTOR_ETFS),
        key=lambda item: item[1],
    )
    if sectors:
        names, moves = zip(*sectors)
        colors = ["#2ebd85" if m >= 0 else "#f6465d" for m in moves]
        ax2.barh(names, moves, color=colors)
        ax2.axvline(0, color="#787b86", lw=0.7)
        ax2.set_title("Sector ETFs — today (%)", color=fg, fontsize=10)

    fig.tight_layout()
    path = out_dir / "market_overview.png"
    fig.savefig(path, dpi=110, facecolor="#131722")
    plt.close(fig)
    return path


def technical_market_summary(overview: dict) -> str:
    """Deterministic fallback for the market summary paragraph."""
    stats = overview["stats"]
    if "SPY" not in stats or "QQQ" not in stats:
        return ""
    spy, qqq = stats["SPY"], stats["QQQ"]
    text = (
        f"SPY {'rose' if spy['day_pct'] >= 0 else 'fell'} {abs(spy['day_pct']):.2f}% today and "
        f"QQQ {'rose' if qqq['day_pct'] >= 0 else 'fell'} {abs(qqq['day_pct']):.2f}%."
    )
    sectors = [(v["name"], v["day_pct"]) for k, v in stats.items() if k in SECTOR_ETFS]
    if sectors:
        best = max(sectors, key=lambda s: s[1])
        worst = min(sectors, key=lambda s: s[1])
        text += (
            f" Sector leadership came from {best[0]} ({best[1]:+.2f}%), while "
            f"{worst[0]} lagged ({worst[1]:+.2f}%)."
        )
    if spy.get("month_pct") is not None and qqq.get("month_pct") is not None:
        text += (
            f" Over the past month SPY is {spy['month_pct']:+.1f}% and QQQ "
            f"{qqq['month_pct']:+.1f}%."
        )
    return text


def _gmail_notes(cfg: dict, query: str, max_items: int, max_chars: int) -> list[dict]:
    """Newest Gmail messages matching a search query, as plain-text excerpts.

    Uses the same account and app password as sending. Failures are logged
    and skipped; the briefing never blocks on the inbox.
    """
    import html as html_lib
    import imaplib
    import re
    from email import message_from_bytes, policy

    password = os.environ.get("TA_SMTP_PASSWORD")
    if not password:
        log(f"inbox search '{query}': TA_SMTP_PASSWORD not set; skipping")
        return []
    notes = []
    try:
        host = (cfg.get("research") or {}).get("imap_host", "imap.gmail.com")
        imap = imaplib.IMAP4_SSL(host, 993)
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
        log(f"inbox search '{query}': loaded {len(notes)} note(s)")
    except Exception as exc:
        log(f"inbox search '{query}' failed — {exc}")
    return notes


def fetch_research_notes(cfg: dict) -> list[dict]:
    """Recent research newsletters for analysis context.

    Sources are configured under research.sources, each with its own Gmail
    query (e.g. TMT Breakout, Citrini Research); a legacy single
    research.gmail_search is still honored when no source list exists.
    """
    research = cfg.get("research") or {}
    if not research.get("enabled"):
        return []
    sources = research.get("sources") or [{
        "gmail_search": research.get("gmail_search", "TMT Breakout newer_than:10d"),
        "max_items": research.get("max_items", 3),
    }]
    notes = []
    for src in sources:
        query = src.get("gmail_search")
        if not query:
            continue
        notes += _gmail_notes(
            cfg, query,
            src.get("max_items", 3),
            research.get("max_chars", 4000),
        )
    return notes


def fetch_podcast_notes(cfg: dict) -> list[dict]:
    """Latest investment-podcast highlights emailed to the inbox.

    Intended for output from the user's podcast-summarizing Claude project:
    that project emails its highlights to this Gmail account, and the newest
    matching message becomes the "Investment Podcast Highlights" section.
    """
    research = cfg.get("research") or {}
    query = research.get("podcast_search")
    if not research.get("enabled") or not query:
        return []
    return _gmail_notes(
        cfg, query,
        research.get("podcast_max_items", 2),
        research.get("podcast_max_chars", 6000),
    )


def _episode_links(entry, feed) -> tuple[str | None, str | None]:
    """(episode page or show page, transcript URL) for a feed entry.

    Transcripts come from the Podcasting 2.0 <podcast:transcript> tag, which
    only some shows publish; HTML/text transcripts are preferred over VTT.
    """
    url = entry.get("link")
    if not url:
        for link in entry.get("links", []) or []:
            if link.get("rel") == "alternate" and link.get("href"):
                url = link["href"]
                break
    if not url:
        url = (feed.feed or {}).get("link")  # fall back to the show page

    transcript = None
    raw = entry.get("podcast_transcript")
    candidates = raw if isinstance(raw, list) else ([raw] if raw else [])
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        href = cand.get("url") or cand.get("href")
        if not href:
            continue
        ctype = (cand.get("type") or "").lower()
        if "html" in ctype or "plain" in ctype or "text" in ctype:
            transcript = href
            break
        transcript = transcript or href
    return url, transcript


def fetch_podcast_feed_notes(cfg: dict) -> list[dict]:
    """Recent episodes from the configured podcast RSS network.

    Show notes usually name the tickers and themes discussed, so they feed
    the highlights and trade-setup extraction alongside inbox messages.
    Failing feeds are logged and skipped.
    """
    import html as html_lib
    import re
    import time

    import feedparser

    research = cfg.get("research") or {}
    feeds = research.get("podcast_feeds") or []
    if not research.get("enabled") or not feeds:
        return []
    max_age = research.get("podcast_feed_days", 7) * 86400
    max_chars = research.get("podcast_feed_chars", 2000)
    now = time.time()
    notes = []
    for url in feeds:
        try:
            parsed = feedparser.parse(url, agent="Mozilla/5.0 (TA-Chart-Agent)")
            show = (parsed.feed.get("title") or url)[:60]
            for entry in parsed.entries[:5]:
                stamp = entry.get("published_parsed") or entry.get("updated_parsed")
                if not stamp or now - time.mktime(stamp) > max_age:
                    continue
                text = entry.get("summary") or entry.get("description") or ""
                text = html_lib.unescape(re.sub(r"<[^>]+>", " ", text))
                text = re.sub(r"\s+", " ", text).strip()
                episode_url, transcript_url = _episode_links(entry, parsed)
                title = entry.get("title", "episode")
                notes.append({
                    "subject": f"{show}: {title}",
                    "show": show,
                    "title": title,
                    "date": (entry.get("published") or "")[:16],
                    "excerpt": text[:max_chars],
                    "url": episode_url,
                    "transcript": transcript_url,
                    "_ts": time.mktime(stamp),
                })
        except Exception as exc:
            log(f"podcast feed failed ({url}) — {exc}")
    notes.sort(key=lambda n: n["_ts"], reverse=True)
    # keep the network diverse: at most 2 episodes per show
    capped, per_show = [], {}
    for note in notes:
        show = note["subject"].split(":")[0]
        if per_show.get(show, 0) >= 2:
            continue
        per_show[show] = per_show.get(show, 0) + 1
        note.pop("_ts", None)
        capped.append(note)
        if len(capped) >= research.get("podcast_feed_max", 8):
            break
    log(f"podcast network: {len(capped)} recent episode note(s) from {len(feeds)} feeds")
    return capped


def llm_parse_trades(confirm_notes: list[dict]) -> list[dict]:
    """Executed trades parsed out of Fidelity confirmation emails.

    Privacy: contents are processed in-memory only; callers must never log
    or persist symbols, quantities, or prices.
    """
    if not confirm_notes or not os.environ.get("ANTHROPIC_API_KEY"):
        return []
    try:
        import anthropic
    except ImportError:
        return []
    schema = {
        "type": "object",
        "properties": {
            "trades": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["buy", "sell"]},
                        "symbol": {"type": "string"},
                        "quantity": {"type": "number"},
                        "price": {"type": "number"},
                        "date": {"type": "string"},
                    },
                    "required": ["action", "symbol", "quantity", "price", "date"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["trades"],
        "additionalProperties": False,
    }
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=(
                "Extract EXECUTED equity/ETF trades from these brokerage "
                "trade-confirmation emails. Include a trade only when the "
                "action (buy/sell), the US ticker symbol, and the share "
                "quantity are explicitly stated; use 0 for a missing price "
                "and the email's date if the trade date is absent. Ignore "
                "pending orders, cancellations, options, bonds, money-market "
                "sweeps, and marketing emails. Return an empty list if none."
            ),
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": json.dumps(confirm_notes)}],
        )
        return json.loads(next(b.text for b in response.content if b.type == "text"))["trades"]
    except Exception as exc:
        log(f"trade parsing failed — {exc}")
        return []


def build_positions(trades: list[dict]) -> dict[str, dict]:
    """Net open long positions (with average buy cost) from executed trades."""
    lots: dict[str, dict] = {}
    for trade in trades:
        symbol = (trade.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        rec = lots.setdefault(symbol, {"qty": 0.0, "cost_qty": 0.0, "cost_amt": 0.0})
        qty = trade["quantity"]
        if trade["action"] == "buy":
            rec["qty"] += qty
            price = trade.get("price") or 0
            if price > 0:
                rec["cost_qty"] += qty
                rec["cost_amt"] += qty * price
        else:
            rec["qty"] -= qty
    return {
        s: {
            "qty": r["qty"],
            "avg_cost": (r["cost_amt"] / r["cost_qty"]) if r["cost_qty"] else None,
        }
        for s, r in lots.items()
        if r["qty"] > 1e-4
    }


def analyze_positions(cfg: dict, ind: dict) -> list[dict]:
    """Portfolio rows for the email: each holding through the TA engine.

    Support is computed through *yesterday's* bar so that today's close can
    be tested against it — `breached` means today closed below the support
    the position had coming into the session.
    """
    portfolio = cfg.get("portfolio") or {}
    if not portfolio.get("enabled", True):
        return []
    confirm_notes = _gmail_notes(
        cfg,
        portfolio.get("gmail_search", "from:fidelity confirmation newer_than:90d"),
        portfolio.get("max_emails", 20),
        3000,
    )
    trades = llm_parse_trades(confirm_notes)
    positions = build_positions(trades)
    log(f"portfolio: parsed {len(trades)} trade(s) into {len(positions)} open position(s)")
    rows = []
    for symbol, lot in positions.items():
        try:
            df = add_indicators(fetch_history(symbol, cfg["lookback_days"]), ind)
            close = float(df["Close"].iloc[-1])
            support = support_level(df.iloc[:-1], ind) if len(df) > 2 else None
            avg_cost = lot.get("avg_cost")
            rows.append({
                "ticker": symbol,
                "qty": lot["qty"],
                "close": close,
                "avg_cost": avg_cost,
                "losing": bool(avg_cost) and close < avg_cost,
                "score": bullish_score(df, ind),
                "support": support,
                "breached": bool(support) and close < support,
                "below_200": below_200day(df, ind),
                "earnings_days": days_to_earnings(symbol),
            })
        except Exception as exc:
            log(f"portfolio: a position could not be analyzed — {exc}")
    rows.sort(key=lambda r: (not r["breached"], -r["score"]))
    return rows


def llm_extract_guests(podcast_notes: list[dict]) -> list[str]:
    """Distinct guest names interviewed in the recent episode notes."""
    if not podcast_notes or not os.environ.get("ANTHROPIC_API_KEY"):
        return []
    try:
        import anthropic
    except ImportError:
        return []
    schema = {
        "type": "object",
        "properties": {"guests": {"type": "array", "items": {"type": "string"}}},
        "required": ["guests"],
        "additionalProperties": False,
    }
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=(
                "From these podcast episode notes, list the distinct GUEST "
                "people interviewed — real person names only, most "
                "market-relevant first, at most 8. Exclude the shows' regular "
                "hosts and co-hosts (e.g. the person a show is named after), "
                "companies, and tickers. Return an empty list if none."
            ),
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": json.dumps(podcast_notes)}],
        )
        guests = json.loads(next(b.text for b in response.content if b.type == "text"))["guests"]
        log("guests identified: " + (", ".join(guests) or "none"))
        return guests
    except Exception as exc:
        log(f"guest extraction failed — {exc}")
        return []


def fetch_guest_crossovers(cfg: dict, podcast_notes: list[dict]) -> list[dict]:
    """Fresh interviews of the network's guests on OTHER shows.

    For each guest identified in the recent episodes, the free iTunes episode
    search finds their appearances across podcasts; episodes from shows not
    already in the network (within the recency window) join the intel.
    """
    import html as html_lib
    import re
    from datetime import timezone

    import requests

    research = cfg.get("research") or {}
    if not research.get("guest_crossover", True) or not podcast_notes:
        return []
    guests = llm_extract_guests(podcast_notes)[: research.get("guest_max", 5)]
    if not guests:
        return []
    known_shows = {n["subject"].split(":")[0].strip().lower() for n in podcast_notes}
    window = timedelta(days=research.get("crossover_days", 21))
    now = datetime.now(timezone.utc)
    notes, seen = [], set()
    for guest in guests:
        try:
            resp = requests.get(
                "https://itunes.apple.com/search",
                params={
                    "term": guest,
                    "media": "podcast",
                    "entity": "podcastEpisode",
                    "limit": 12,
                },
                timeout=15,
            )
            if resp.status_code != 200:
                continue
            for ep in resp.json().get("results", []):
                show = (ep.get("collectionName") or "").strip()
                title = (ep.get("trackName") or "").strip()
                if not show or not title or show.lower() in known_shows:
                    continue
                blob = f"{title} {ep.get('description') or ''}".lower()
                if guest.lower() not in blob:
                    continue
                try:
                    when = datetime.fromisoformat(
                        (ep.get("releaseDate") or "").replace("Z", "+00:00")
                    )
                except ValueError:
                    continue
                if now - when > window:
                    continue
                key = (show.lower(), title.lower())
                if key in seen:
                    continue
                seen.add(key)
                desc = re.sub(
                    r"\s+", " ",
                    html_lib.unescape(re.sub(r"<[^>]+>", " ", ep.get("description") or "")),
                ).strip()
                notes.append({
                    "subject": f"[guest crossover: {guest}] {show}: {title}",
                    "show": show,
                    "title": title,
                    "guest": guest,
                    "date": (ep.get("releaseDate") or "")[:10],
                    "excerpt": desc[: research.get("podcast_feed_chars", 2000)],
                    "url": ep.get("trackViewUrl") or ep.get("collectionViewUrl"),
                    "transcript": None,
                    "_ts": when.timestamp(),
                })
        except Exception as exc:
            log(f"crossover search failed for {guest} — {exc}")
    notes.sort(key=lambda n: n["_ts"], reverse=True)
    notes = notes[: research.get("crossover_max", 5)]
    for note in notes:
        note.pop("_ts", None)
    log(
        "guest crossovers: "
        + ("; ".join(n["subject"][:70] for n in notes) if notes else "none found")
    )
    return notes


def resistance_level(df: pd.DataFrame, ind: dict) -> float | None:
    """Nearest meaningful resistance above the last close.

    Mirror image of support_level: the lowest of the Fibonacci levels, the
    latest swing high, and any moving averages sitting above price — the
    first ceiling the bulls need to clear.
    """
    last_close = float(df["Close"].iloc[-1])
    candidates = []
    window = df.tail(180)
    candidates += [lvl for lvl in fib_levels(window).values() if lvl > last_close]
    highs = window["High"].to_numpy()
    pivots = find_pivots(highs, "high")
    if pivots and highs[pivots[-1]] > last_close:
        candidates.append(float(highs[pivots[-1]]))
    for n in (ind["sma_fast"], ind["sma_slow"], ind.get("sma_long")):
        col = f"SMA{n}"
        if n and col in df and pd.notna(df[col].iloc[-1]) and df[col].iloc[-1] > last_close:
            candidates.append(float(df[col].iloc[-1]))
    return min(candidates) if candidates else None


def vol_context(ticker: str, df: pd.DataFrame) -> dict:
    """Realized and implied volatility for options framing.

    HV20/HV60 are annualized realized vols from closes; IV is the at-the-money
    implied vol from the option chain expiring ~1 month out (None when the
    chain is unavailable).
    """
    rets = df["Close"].pct_change().dropna()
    out = {
        "hv20": round(float(rets.tail(20).std() * np.sqrt(252)) * 100, 1),
        "hv60": round(float(rets.tail(60).std() * np.sqrt(252)) * 100, 1),
        "iv": None,
        "iv_days": None,
    }
    try:
        tk = yf.Ticker(ticker)
        today = datetime.now(EASTERN).date()
        spot = float(df["Close"].iloc[-1])
        all_exps = []
        for e in tk.options or []:
            days = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
            if days >= 0:
                all_exps.append((days, e))
        all_exps.sort()
        chains = {}

        def get_chain(expiry):
            if expiry not in chains:
                chains[expiry] = tk.option_chain(expiry)
            return chains[expiry]

        # ATM implied vol from the expiry nearest ~35 days out
        iv_candidates = [(abs(d - 35), d, e) for d, e in all_exps if d >= 14]
        if iv_candidates:
            _, days, expiry = min(iv_candidates)
            chain = get_chain(expiry)
            ivs = []
            for side in (chain.calls, chain.puts):
                side = side[side["impliedVolatility"] > 0.01]
                if len(side):
                    ivs.append(float(
                        side.iloc[(side["strike"] - spot).abs().argmin()]["impliedVolatility"]
                    ))
            if ivs:
                out["iv"] = round(sum(ivs) / len(ivs) * 100, 1)
                out["iv_days"] = days

        # Day-level flow read across the two nearest expiries
        call_vol = put_vol = otm_call_vol = 0
        unusual = []
        for _, expiry in all_exps[:2]:
            chain = get_chain(expiry)
            calls, puts = chain.calls, chain.puts
            cv = calls["volume"].fillna(0)
            call_vol += int(cv.sum())
            put_vol += int(puts["volume"].fillna(0).sum())
            otm = calls[calls["strike"] > spot]
            otm_call_vol += int(otm["volume"].fillna(0).sum())
            oi = calls["openInterest"].fillna(0)
            hot = calls[(cv > oi) & (cv >= 200)]
            for _, row in hot.iterrows():
                unusual.append((
                    expiry, float(row["strike"]),
                    int(row["volume"]), int(row["openInterest"] or 0),
                ))
        if call_vol + put_vol >= 500:  # too illiquid to read below this
            unusual.sort(key=lambda u: -u[2])
            out["flow"] = {
                "cp_ratio": round(call_vol / max(put_vol, 1), 1),
                "call_vol": call_vol,
                "put_vol": put_vol,
                "otm_call_share": round(otm_call_vol / max(call_vol, 1) * 100),
                "unusual": unusual[:2],
            }
    except Exception:
        pass
    return out


def flow_note(flow: dict | None) -> str:
    """Human-readable options-flow tilt from day-level chain volume."""
    if not flow:
        return ""
    cp, otm = flow["cp_ratio"], flow["otm_call_share"]
    if cp >= 2 and otm >= 50:
        verdict = "flow leans clearly bullish"
    elif cp >= 1.5:
        verdict = "flow tilts bullish"
    elif cp <= 0.7:
        verdict = "flow tilts defensive (put-heavy)"
    else:
        verdict = "flow is balanced"
    bits = [f"call/put volume {cp}:1 across the nearest expiries"]
    if otm >= 60:
        bits.append(f"{otm}% of call volume in out-of-the-money strikes")
    for expiry, strike, vol, oi in flow.get("unusual", []):
        bits.append(
            f"heavy activity in the {expiry} {strike:g} calls "
            f"({vol:,} traded vs {oi:,} open interest — likely new positioning)"
        )
    return (
        f"{verdict} — " + "; ".join(bits)
        + ". (Day-level volume cannot distinguish buyers from sellers; treat as a tilt, not proof.)"
    )


def options_note(vol: dict, support: float | None, resistance: float | None,
                 earnings_days: int | None) -> str:
    """Rule-based options structure suggestion from the vol picture."""
    hv = vol.get("hv20")
    iv = vol.get("iv")
    if iv and hv:
        ratio = iv / hv if hv else None
        base = f"{vol['iv_days']}d ATM IV {iv:.0f}% vs 20d realized {hv:.0f}%"
        if ratio and ratio >= 1.25:
            stance = (
                "IV is rich — defined-risk premium selling (e.g. a bull put spread "
                + (f"struck below support {support:,.2f}" if support else "below support")
                + ") is favored over buying calls"
            )
        elif ratio and ratio <= 0.9:
            stance = (
                "IV is cheap — long calls or call debit spreads "
                + (f"toward resistance {resistance:,.2f}" if resistance else "")
                + " offer efficient upside with defined risk"
            )
        else:
            stance = "IV is roughly fair — call debit spreads balance cost and theta"
        note = f"{base}; {stance}"
    elif hv:
        note = f"20d realized vol {hv:.0f}% (option chain unavailable for IV)"
    else:
        return ""
    if earnings_days is not None and earnings_days <= 10:
        note += f"; earnings in {earnings_days}d — IV crush risk for long premium held through the print"
    return note


def reward_risk(close: float, support: float | None, resistance: float | None) -> float | None:
    """PTJ-style asymmetry: upside to first resistance per unit of downside to support."""
    if not support or not resistance or close <= support or resistance <= close:
        return None
    return round((resistance - close) / (close - support), 1)


def ptj_tag(below_200: bool) -> str:
    return " · ⛔ below 200-day" if below_200 else ""


def below_200day(df: pd.DataFrame, ind: dict) -> bool:
    long = ind.get("sma_long")
    col = f"SMA{long}" if long else None
    if not col or col not in df or pd.isna(df[col].iloc[-1]):
        return False
    return float(df["Close"].iloc[-1]) < float(df[col].iloc[-1])


def days_to_earnings(ticker: str, horizon: int = 45) -> int | None:
    """Days until the next earnings report, or None if unknown/far out."""
    try:
        cal = yf.Ticker(ticker).calendar
        dates = (cal or {}).get("Earnings Date") or []
        if not isinstance(dates, (list, tuple)):
            dates = [dates]
        today = datetime.now(EASTERN).date()
        best = None
        for d in dates:
            if hasattr(d, "date") and not isinstance(d, type(today)):
                d = d.date()
            delta = (d - today).days
            if 0 <= delta <= horizon and (best is None or delta < best):
                best = delta
        return best
    except Exception:
        return None


def earnings_tag(days: int | None, threshold: int = 10) -> str:
    """Warning suffix for headings/status cells when earnings are close."""
    if days is None or days > threshold:
        return ""
    return f" · ⚠️ earnings in {days}d"


LEDGER_PATH = Path("picks_ledger.csv")


def append_picks_ledger(summaries: list[dict], setups: list[dict]) -> None:
    """Record today's picks and setups for the weekly scorecard.

    The ledger holds only scan output (ticker, score, entry, support) —
    never portfolio data — so it is safe to commit to the public repo.
    """
    import csv

    existing = set()
    if LEDGER_PATH.exists():
        with LEDGER_PATH.open() as fh:
            existing = {
                (r["date"], r["kind"], r["ticker"]) for r in csv.DictReader(fh)
            }
    today = f"{datetime.now(EASTERN):%Y-%m-%d}"
    rows = []
    for s in summaries:
        if "score" in s:
            rows.append(
                ("pick", s["ticker"], s["score"],
                 float(str(s["close"]).replace(",", "")), s.get("support"))
            )
    for st in setups:
        rows.append(("setup", st["ticker"], st["score"], st["close"], st.get("support")))
    is_new = not LEDGER_PATH.exists()
    added = 0
    with LEDGER_PATH.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(["date", "kind", "ticker", "score", "close", "support"])
        for kind, ticker, score, close, support in rows:
            if (today, kind, ticker) in existing:
                continue
            writer.writerow([today, kind, ticker, score, f"{close:.2f}",
                             f"{support:.2f}" if support else ""])
            added += 1
    log(f"ledger: recorded {added} new entrie(s)")


SCORECARD_HORIZONS = (("1-Week", 5), ("1-Month", 21), ("1-Year", 252))


def build_scorecard(cfg: dict) -> str | None:
    """HTML for the Performance Scorecard across 1-week/1-month/1-year.

    Grades every ledger entry with at least a week of history: forward
    return at each horizon that has elapsed, the same-window SPY return
    (alpha), and whether the recorded support broke after entry. Returns
    None when there is nothing gradeable yet.
    """
    import csv

    if not LEDGER_PATH.exists():
        return None
    with LEDGER_PATH.open() as fh:
        rows = list(csv.DictReader(fh))[-5000:]
    today = datetime.now(EASTERN).date()
    try:
        spy = fetch_history("SPY", 560)
    except Exception:
        spy = None
    graded, cache = [], {}
    for r in rows:
        try:
            entry_date = datetime.strptime(r["date"], "%Y-%m-%d").date()
            age = (today - entry_date).days
            if age < 7:
                continue
            ticker = r["ticker"]
            if ticker not in cache:
                cache[ticker] = fetch_history(ticker, min(550, age + 80))
            df = cache[ticker]
            entry = float(r["close"])
            support = float(r["support"]) if r["support"] else None
            stamp = pd.Timestamp(entry_date)
            after = df[df.index > stamp]
            if len(after) < 5:
                continue
            rec = {
                "score": float(r["score"]),
                "rets": {},
                "alphas": {},
                "broke": bool(support) and bool((after["Close"] < support).any()),
            }
            spy_after, spy_entry = None, None
            if spy is not None and (spy.index <= stamp).any():
                spy_entry = float(spy[spy.index <= stamp]["Close"].iloc[-1])
                spy_after = spy[spy.index > stamp]
            for name, bars in SCORECARD_HORIZONS:
                if len(after) >= bars:
                    ret = float(after["Close"].iloc[bars - 1]) / entry - 1
                    rec["rets"][name] = ret
                    if spy_after is not None and len(spy_after) >= bars and spy_entry:
                        rec["alphas"][name] = ret - (
                            float(spy_after["Close"].iloc[bars - 1]) / spy_entry - 1
                        )
            if rec["rets"]:
                graded.append(rec)
        except Exception:
            continue
    if not graded:
        return None

    def horizon_row(label, items, horizon):
        sample = [g for g in items if horizon in g["rets"]]
        if not sample:
            return ""
        n = len(sample)
        rets = [g["rets"][horizon] for g in sample]
        alphas = [g["alphas"][horizon] for g in sample if horizon in g["alphas"]]
        hit = sum(1 for x in rets if x > 0) / n * 100
        avg = sum(rets) / n * 100
        alpha = f"{sum(alphas) / len(alphas) * 100:+.1f}%" if alphas else "—"
        broke = sum(1 for g in sample if g["broke"]) / n * 100
        return (
            f"<tr><td>{label}</td><td>{n}</td><td>{hit:.0f}%</td>"
            f"<td>{avg:+.1f}%</td><td>{alpha}</td><td>{broke:.0f}%</td></tr>"
        )

    header = (
        "<table border='1' cellpadding='6' cellspacing='0' "
        "style='border-collapse:collapse;font-size:14px'>"
        "<tr style='background:#eceff1'><th>{first}</th><th>N</th>"
        "<th>Hit rate</th><th>Avg return</th><th>Avg vs SPY</th>"
        "<th>Support broke</th></tr>"
    )
    block = (
        "<h3 style='font-family:sans-serif'>Performance Scorecard</h3>"
        "<p style='max-width:860px;font-size:14px'>How the ledger's signals "
        "performed at each horizon (graded on the close at signal time; "
        "'vs SPY' is the average excess return over SPY across the same "
        "windows):</p>"
    )
    block += header.format(first="Horizon")
    for name, _ in SCORECARD_HORIZONS:
        block += horizon_row(name, graded, name)
    block += "</table>"

    high = [g for g in graded if g["score"] >= 4.0]
    low = [g for g in graded if g["score"] < 4.0]
    if high and low:
        bucket_h = "1-Month" if any("1-Month" in g["rets"] for g in graded) else "1-Week"
        block += (
            f"<p style='max-width:860px;font-size:14px;margin-top:10px'>"
            f"By score bucket ({bucket_h}):</p>"
        )
        block += header.format(first="Bucket")
        block += horizon_row("Score ≥ 4.0", high, bucket_h)
        block += horizon_row("Score &lt; 4.0", low, bucket_h)
        block += "</table>"
    return block


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


def actionable_line(support: float | None, rr: float | None = None,
                    below_200: bool = False) -> str:
    """The bolded buy/support recommendation shown under each pick."""
    if not support:
        return ""
    text = (
        f"Actionable: the bullish setup holds while price stays above support near"
        f" {support:,.2f} — a decisive close below that level breaks the trend."
    )
    if rr:
        text += f" Reward:risk to first resistance ≈ {rr}:1" + (
            " — meets the ≥3:1 asymmetry bar." if rr >= 3 else
            " — thin asymmetry; better entries come on pullbacks toward support."
        )
    if below_200:
        text += " ⛔ Below the 200-day — treat as counter-trend."
    return text


def technical_commentary(df: pd.DataFrame, ind: dict) -> str:
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
        return "Mixed technical picture; see chart for details."
    text = "; ".join(bits)
    return text[0].upper() + text[1:] + "."


def fetch_tweets(cfg: dict, tickers: list[str]) -> dict[str, list[dict]]:
    """Recent X/Twitter chatter per ticker, via the official API v2.

    Requires an X_BEARER_TOKEN (paid X API tier with search access). Skips
    silently when no token is configured; stops on the first API error so a
    rate limit can't stall the briefing.
    """
    import requests

    token = os.environ.get("X_BEARER_TOKEN")
    research = cfg.get("research") or {}
    if not token or not research.get("twitter_enabled", True):
        return {}
    out = {}
    for ticker in tickers[: research.get("twitter_max_tickers", 6)]:
        try:
            resp = requests.get(
                "https://api.twitter.com/2/tweets/search/recent",
                params={
                    "query": f'("${ticker}" OR "{ticker} stock") lang:en -is:retweet',
                    "max_results": 10,
                    "tweet.fields": "created_at,public_metrics",
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=15,
            )
            if resp.status_code != 200:
                log(f"X search failed for {ticker} (HTTP {resp.status_code}); skipping the rest")
                break
            tweets = sorted(
                resp.json().get("data") or [],
                key=lambda t: (t.get("public_metrics") or {}).get("like_count", 0),
                reverse=True,
            )[:3]
            if tweets:
                out[ticker] = [
                    {
                        "text": t["text"][:280],
                        "likes": (t.get("public_metrics") or {}).get("like_count", 0),
                        "date": (t.get("created_at") or "")[:10],
                    }
                    for t in tweets
                ]
        except Exception as exc:
            log(f"X search failed for {ticker} — {exc}; skipping the rest")
            break
    if out:
        log(f"X chatter loaded for: {', '.join(out)}")
    return out


def llm_extract_podcast_ideas(podcast_notes: list[dict]) -> list[dict]:
    """Tickers with an explicit investment view voiced in the podcast notes."""
    if not podcast_notes or not os.environ.get("ANTHROPIC_API_KEY"):
        return []
    try:
        import anthropic
    except ImportError:
        return []
    schema = {
        "type": "object",
        "properties": {
            "ideas": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "thesis": {"type": "string"},
                        "sentiment": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
                    },
                    "required": ["ticker", "thesis", "sentiment"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["ideas"],
        "additionalProperties": False,
    }
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=(
                "Extract individual stock or ETF investment ideas from these "
                "podcast summaries. Include only names where a speaker voiced "
                "an explicit view. ticker: the US-listed Yahoo Finance symbol "
                "(resolve company names to tickers; skip non-US or ambiguous "
                "names). thesis: one sentence stating the speaker's view and "
                "why, attributed to the show when identifiable. sentiment: "
                "bullish, bearish, or neutral. Order by prominence. Return an "
                "empty list if there are none."
            ),
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": json.dumps(podcast_notes)}],
        )
        ideas = json.loads(next(b.text for b in response.content if b.type == "text"))["ideas"]
        log("podcast ideas extracted: " + (", ".join(i["ticker"] for i in ideas) or "none"))
        return ideas
    except Exception as exc:
        log(f"podcast idea extraction failed — {exc}")
        return []


def default_setup_plan(setup: dict) -> str:
    """Deterministic fallback trade plan with the computed levels."""
    support, resistance = setup.get("support"), setup.get("resistance")
    if support and resistance:
        plan = (
            f"Trade plan: constructive above support near {support:,.2f} with first "
            f"resistance near {resistance:,.2f}; a decisive close below "
            f"{support:,.2f} invalidates the setup."
        )
        rr = setup.get("rr")
        if rr:
            plan += f" Reward:risk to first resistance ≈ {rr}:1" + (
                " — meets the ≥3:1 asymmetry bar." if rr >= 3 else
                " — thin asymmetry; PTJ discipline says pass or wait for a pullback toward support."
            )
        if setup.get("below_200"):
            plan += " ⛔ Trades below its 200-day — nothing good happens below the 200-day; treat as counter-trend."
        return plan
    if support:
        return (
            f"Trade plan: constructive above support near {support:,.2f}; a decisive "
            f"close below that level invalidates the setup."
        )
    return "Trade plan: no clean support level below price — wait for a base to form."


def llm_commentary(
    picks: list[dict],
    research_notes: list[dict] | None = None,
    market_stats: dict | None = None,
    podcast_notes: list[dict] | None = None,
    podcast_setups: list[dict] | None = None,
    tweets: dict[str, list[dict]] | None = None,
) -> tuple[dict[str, dict], str, list[str], dict[str, dict], dict[str, dict]] | None:
    """Claude-written commentary, when ANTHROPIC_API_KEY is configured.

    Returns ({ticker: {commentary, actionable}}, market_summary,
    podcast_highlights, {ticker: {analysis, plan}},
    {note_id: {commentary, tickers}}) or None, in which case the caller
    keeps the deterministic versions.
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
                        "actionable": {"type": "string"},
                    },
                    "required": ["ticker", "commentary", "actionable"],
                    "additionalProperties": False,
                },
            },
            "market_summary": {"type": "string"},
            "podcast_highlights": {"type": "array", "items": {"type": "string"}},
            "podcast_episodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "commentary": {"type": "string"},
                        "tickers": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id", "commentary", "tickers"],
                    "additionalProperties": False,
                },
            },
            "setups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "analysis": {"type": "string"},
                        "plan": {"type": "string"},
                    },
                    "required": ["ticker", "analysis", "plan"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "items", "market_summary", "podcast_highlights", "podcast_episodes", "setups",
        ],
        "additionalProperties": False,
    }
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=(
                "You write the commentary sections of a daily technical-analysis "
                "briefing email. Bullish scores are on a 1-5 scale (5 = most "
                "bullish). For each ticker: 'commentary' is 2-3 sentences on the "
                "plausible drivers of its bullish price action, weaving together "
                "the technical signals and the recent headlines provided; "
                "'actionable' is one separate sentence using the provided "
                "support_level — frame the setup as constructive while price "
                "holds above that support (entries on strength or on pullbacks "
                "toward it), and state explicitly that a decisive close below "
                "the support level breaks the bullish trend, using the exact "
                "support number. The input may include research_notes — "
                "excerpts from newsletters the user subscribes to (e.g. TMT "
                "Breakout). Treat them as an analyst's view: when a note "
                "discusses one of the tickers or a clearly relevant theme, "
                "weave that perspective in with attribution (e.g. 'TMT "
                "Breakout flags...'); silently ignore notes irrelevant to a "
                "pick. Separately, write 'market_summary': a 4-6 sentence "
                "paragraph on today's SPY and QQQ moves using the market_stats "
                "provided, the sector rotation visible in the sector data "
                "(which groups led and lagged, and what that rotation "
                "suggests), and macro factors likely to influence stocks over "
                "a one-month horizon — drawing only on the provided stats, "
                "headlines, and research notes; if market_stats is empty, "
                "return an empty market_summary. Also fill 'podcast_highlights': "
                "if podcast_notes are provided (inbox summaries plus recent "
                "episodes from a podcast RSS network), distill them into 4-8 "
                "short bullet highlights an investor would want to remember — "
                "the key theses, tickers, and calls made, each attributed to "
                "the show/episode; prefer concrete investment views over "
                "episode descriptions; if podcast_notes is empty, "
                "return an empty array. Also fill 'podcast_episodes': one "
                "entry per podcast note that carries investment substance, "
                "using that note's exact 'id' — 'commentary' is 2-3 sentences "
                "on what an investor should take from that specific episode "
                "(the guest's thesis, the mechanism, the risk, and how it "
                "connects to current market conditions or the picks above), "
                "and 'tickers' lists any US-listed symbols discussed in it "
                "(empty when none). Skip notes that are purely promotional "
                "or have no investment content. For each entry in "
                "podcast_trade_candidates (research_notes may span multiple "
                "letters — e.g. TMT Breakout, Citrini Research — attribute "
                "each view to its letter by name), add a 'setups' item: 'analysis' is "
                "1-2 sentences connecting the podcast thesis to the current "
                "technical posture (score, trend, levels); 'plan' is ONE "
                "sentence with the exact numbers provided — the constructive "
                "zone above the support level, the first resistance to watch, "
                "and that a decisive close below the support invalidates the "
                "setup (for a bearish podcast view, frame it as a name to "
                "avoid or de-risk rather than a long setup). Wherever "
                "days_to_earnings is 7 or less, the commentary or plan MUST "
                "flag the imminent earnings report as event risk that can gap "
                "through technical levels. Apply Paul Tudor Jones risk "
                "principles throughout: play great defense first; frame every "
                "idea by its asymmetry using the provided reward_risk (upside "
                "to first resistance per unit of downside to support) and "
                "treat >=3:1 as the bar worth taking; respect the 200-day "
                "moving average — when below_200day is true, say plainly that "
                "nothing good happens below the 200-day and frame any long as "
                "counter-trend; never suggest adding to a losing position "
                "(losers average losers); when signals conflict, recommend "
                "cutting risk rather than rationalizing. Use the volatility "
                "data (realized HV vs ATM IV) and the VIX in market_stats to "
                "set options context: rich IV favors defined-risk premium "
                "selling, cheap IV favors long calls or debit spreads — "
                "always defined-risk, never naked. The volatility data may "
                "include 'flow' (call/put volume ratio, OTM call share, and "
                "strikes where volume exceeded open interest): treat bullish "
                "flow as corroborating evidence when it agrees with the "
                "technical setup and note the divergence when it doesn't, but "
                "never as proof — day-level volume cannot distinguish buyers "
                "from sellers. If tweets are "
                "provided, treat them as unverified crowd chatter: you may "
                "note notable sentiment as 'X chatter...', never as fact, and "
                "ignore anything spammy. Ground every claim ONLY in "
                "the provided data — do not invent facts, numbers, or events. "
                "These are levels to watch, not personalized financial advice. "
                "Plain prose, no preamble, no hype words."
            ),
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{
                "role": "user",
                "content": json.dumps({
                    "picks": picks,
                    "research_notes": research_notes or [],
                    "market_stats": market_stats or {},
                    "podcast_notes": podcast_notes or [],
                    "podcast_trade_candidates": podcast_setups or [],
                    "tweets": tweets or {},
                }),
            }],
        )
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        per_ticker = {
            item["ticker"]: {"commentary": item["commentary"], "actionable": item["actionable"]}
            for item in data["items"]
        }
        log(f"Claude commentary generated for {len(per_ticker)} tickers")
        setups_map = {
            item["ticker"]: {"analysis": item["analysis"], "plan": item["plan"]}
            for item in data.get("setups", [])
        }
        episodes_map = {
            item["id"]: {
                "commentary": item["commentary"],
                "tickers": item.get("tickers", []),
            }
            for item in data.get("podcast_episodes", [])
        }
        return (
            per_ticker,
            data.get("market_summary", ""),
            data.get("podcast_highlights", []),
            setups_map,
            episodes_map,
        )
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


def build_html(
    summaries: list[dict],
    cids: dict[str, str],
    generated: str,
    heading: str,
    market: dict | None = None,
    podcast: dict | None = None,
    setups: list[dict] | None = None,
    positions: list[dict] | None = None,
    scorecard: str | None = None,
) -> str:
    owned = {p["ticker"] for p in positions} if positions else set()
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
    if positions:
        breached = [p["ticker"] for p in positions if p["breached"]]
        block = "<h3 style='font-family:sans-serif'>Your Positions</h3>"
        if breached:
            block += (
                "<p style='max-width:860px;font-size:14px'><b>⚠️ Support broken: "
                + ", ".join(breached)
                + " closed below the support carried into today — bullish structure is broken.</b></p>"
            )
        block += (
            "<table border='1' cellpadding='6' cellspacing='0' "
            "style='border-collapse:collapse;font-size:14px'>"
            "<tr style='background:#eceff1'><th>Ticker</th><th>Shares</th><th>Avg Cost</th>"
            "<th>Last</th><th>Score</th><th>Support</th><th>Status</th></tr>"
        )
        for p in positions:
            status = "⚠️ below support" if p["breached"] else "✅ above support"
            status += earnings_tag(p.get("earnings_days"))
            status += ptj_tag(p.get("below_200", False))
            if p.get("losing"):
                status += " · 🔻 below cost — losers average losers: do not add"
            support_cell = f"{p['support']:,.2f}" if p.get("support") else "—"
            cost_cell = f"{p['avg_cost']:,.2f}" if p.get("avg_cost") else "—"
            block += (
                f"<tr><td><b>{p['ticker']}</b></td><td>{p['qty']:g}</td>"
                f"<td>{cost_cell}</td>"
                f"<td>{p['close']:,.2f}</td><td>{p['score']}</td>"
                f"<td>{support_cell}</td><td>{status}</td></tr>"
            )
        block += "</table>"
        sections.append(block)
    for s in summaries:
        ticker = s["ticker"]
        heading_line = ticker + (f" — bullish score {s['score']}" if has_score else "")
        if ticker in owned:
            heading_line += " · in your portfolio"
        heading_line += earnings_tag(s.get("earnings_days"))
        heading_line += ptj_tag(s.get("below_200", False))
        block = f"<h3 style='font-family:sans-serif'>{heading_line}</h3>"
        if s.get("commentary"):
            block += (
                "<p style='max-width:860px;font-size:14px'>"
                f"<b>Potential drivers:</b> {s['commentary']}</p>"
            )
        if s.get("actionable"):
            block += (
                "<p style='max-width:860px;font-size:14px'>"
                f"<b>{s['actionable']}</b></p>"
            )
        if s.get("options_note"):
            block += (
                "<p style='max-width:860px;font-size:13px'>"
                f"<i>Options: {s['options_note']}</i></p>"
            )
        if s.get("flow_note"):
            block += (
                "<p style='max-width:860px;font-size:13px'>"
                f"<i>Options flow: {s['flow_note']}</i></p>"
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
    if market and (market.get("text") or market.get("cid")):
        block = "<h3 style='font-family:sans-serif'>Market Summary — SPY &amp; QQQ</h3>"
        if market.get("text"):
            block += f"<p style='max-width:860px;font-size:14px'>{market['text']}</p>"
        if market.get("cid"):
            block += f"<img src='cid:{market['cid']}' width='860' style='max-width:100%'/>"
        sections.append(block)
    if podcast and (
        podcast.get("highlights") or podcast.get("episodes") or podcast.get("text")
    ):
        block = "<h3 style='font-family:sans-serif'>Investment Podcast Highlights</h3>"
        if podcast.get("highlights"):
            block += (
                "<p style='max-width:860px;font-size:14px'><b>Across this week's "
                "shows:</b></p><ul style='max-width:860px;font-size:14px'>"
                + "".join(f"<li>{h}</li>" for h in podcast["highlights"])
                + "</ul>"
            )
        elif podcast.get("text"):
            block += f"<p style='max-width:860px;font-size:14px'>{podcast['text']}</p>"

        for ep in podcast.get("episodes") or []:
            title = ep.get("title") or ep.get("subject", "Episode")
            show = ep.get("show")
            label = f"{show} — {title}" if show else title
            if ep.get("guest"):
                label = f"🔁 {label}"
            block += (
                "<p style='max-width:860px;font-size:14px;margin-bottom:2px'>"
                f"<b>{label}</b>"
            )
            if ep.get("date"):
                block += f" <span style='color:#777'>({ep['date']})</span>"
            links = []
            if ep.get("url"):
                links.append(f"<a href='{ep['url']}'>Episode ↗</a>")
            if ep.get("transcript"):
                links.append(f"<a href='{ep['transcript']}'>Transcript ↗</a>")
            if links:
                block += " · " + " · ".join(links)
            block += "</p>"
            if ep.get("commentary"):
                block += (
                    "<p style='max-width:860px;font-size:14px;margin-top:2px'>"
                    f"{ep['commentary']}</p>"
                )
            if ep.get("tickers"):
                block += (
                    "<p style='max-width:860px;font-size:13px;color:#555;"
                    "margin-top:2px'><i>Discussed: "
                    + ", ".join(ep["tickers"])
                    + "</i></p>"
                )
        if podcast.get("episodes"):
            block += (
                "<p style='color:#777;font-size:12px'>🔁 marks an episode found "
                "by following a guest to a show outside the regular network. "
                "Transcript links appear when the show publishes one.</p>"
            )
        elif podcast.get("source"):
            block += f"<p style='color:#777;font-size:12px'>Source: {podcast['source']}</p>"
        sections.append(block)
    if setups:
        block = "<h3 style='font-family:sans-serif'>Podcast Trade Setups</h3>"
        for st in setups:
            title = f"{st['ticker']} — bullish score {st['score']}"
            if st.get("sentiment"):
                title += f" · podcast view: {st['sentiment']}"
            if st["ticker"] in owned:
                title += " · in your portfolio"
            title += earnings_tag(st.get("earnings_days"))
            title += ptj_tag(st.get("below_200", False))
            block += f"<h4 style='font-family:sans-serif;margin-bottom:4px'>{title}</h4>"
            if st.get("thesis"):
                block += (
                    "<p style='max-width:860px;font-size:14px'>"
                    f"<i>Podcast thesis:</i> {st['thesis']}</p>"
                )
            if st.get("analysis"):
                block += f"<p style='max-width:860px;font-size:14px'>{st['analysis']}</p>"
            if st.get("plan"):
                block += (
                    "<p style='max-width:860px;font-size:14px'>"
                    f"<b>{st['plan']}</b></p>"
                )
            if st.get("options_note"):
                block += (
                    "<p style='max-width:860px;font-size:13px'>"
                    f"<i>Options: {st['options_note']}</i></p>"
                )
            if st.get("flow_note"):
                block += (
                    "<p style='max-width:860px;font-size:13px'>"
                    f"<i>Options flow: {st['flow_note']}</i></p>"
                )
            if st["ticker"] in cids:
                block += (
                    f"<img src='cid:{cids[st['ticker']]}' width='860' style='max-width:100%'/>"
                )
        sections.append(block)
    if scorecard:
        sections.append(scorecard)
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
    cfg: dict,
    summaries: list[dict],
    chart_paths: dict[str, Path],
    heading: str,
    market: dict | None = None,
    podcast: dict | None = None,
    setups: list[dict] | None = None,
    positions: list[dict] | None = None,
    scorecard: str | None = None,
) -> EmailMessage:
    email_cfg = cfg["email"]
    msg = EmailMessage()
    msg["Subject"] = f"{heading} — {datetime.now(EASTERN):%Y-%m-%d}"
    msg["From"] = email_cfg["from_addr"]
    msg["To"] = ", ".join(email_cfg["to_addrs"])
    msg.set_content("Your email client does not support HTML. See attached charts.")

    cids = {t: make_msgid(domain="ta-chart-agent")[1:-1] for t in chart_paths}
    market_html = None
    if market:
        market_html = {"text": market.get("text", "")}
        if market.get("path"):
            market_html["cid"] = make_msgid(domain="ta-chart-agent")[1:-1]
    generated = f"{datetime.now(EASTERN):%Y-%m-%d %H:%M %Z}"
    msg.add_alternative(
        build_html(
            summaries, cids, generated, heading, market_html, podcast, setups,
            positions, scorecard,
        ),
        subtype="html",
    )
    for ticker, path in chart_paths.items():
        msg.get_payload()[1].add_related(
            path.read_bytes(), maintype="image", subtype="png", cid=f"<{cids[ticker]}>"
        )
    if market_html and market_html.get("cid"):
        msg.get_payload()[1].add_related(
            market["path"].read_bytes(),
            maintype="image",
            subtype="png",
            cid=f"<{market_html['cid']}>",
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
                summary["resistance"] = resistance_level(df, ind)
                summary["rr"] = reward_risk(
                    float(df["Close"].iloc[-1]), summary["support"], summary["resistance"]
                )
                summary["below_200"] = below_200day(df, ind)
                summary["earnings_days"] = days_to_earnings(ticker)
                summary["commentary"] = technical_commentary(df, ind)
                summary["actionable"] = actionable_line(
                    summary["support"], summary["rr"], summary["below_200"]
                )
                vol = vol_context(ticker, df)
                summary["vol"] = vol
                summary["options_note"] = options_note(
                    vol, summary["support"], summary["resistance"], summary["earnings_days"]
                )
                summary["flow_note"] = flow_note(vol.get("flow"))
            summaries.append(summary)
            log(f"{ticker}: chart written to {chart_paths[ticker]}")
        except Exception as exc:  # one bad ticker must not kill the briefing
            failures.append(ticker)
            log(f"{ticker}: FAILED — {exc}")

    if not summaries:
        log("no tickers succeeded; aborting")
        return 1

    market = None
    podcast = None
    podcast_setups = []
    positions = []
    scorecard = None
    if args.scan:
        positions = analyze_positions(cfg, ind)
        overview = market_overview()
        market_text = technical_market_summary(overview)
        market_path = None
        try:
            market_path = render_market_chart(overview, out_dir)
        except Exception as exc:
            log(f"market chart failed — {exc}")

        # Upgrade the deterministic commentary to Claude-written prose when an
        # API key is configured; otherwise keep the technical version.
        picks_context = [
            {
                "ticker": s["ticker"],
                "bullish_score": s.get("score"),
                "last_close": s["close"],
                "support_level": s.get("support"),
                "resistance_level": s.get("resistance"),
                "reward_risk": s.get("rr"),
                "below_200day": s.get("below_200"),
                "volatility": s.get("vol"),
                "days_to_earnings": s.get("earnings_days"),
                "technicals": s.get("commentary"),
                "headlines": s.get("news", []),
            }
            for s in summaries
        ]
        research_notes = fetch_research_notes(cfg)
        podcast_notes = fetch_podcast_notes(cfg) + fetch_podcast_feed_notes(cfg)
        podcast_notes += fetch_guest_crossovers(cfg, podcast_notes)
        for idx, note in enumerate(podcast_notes, 1):
            note["id"] = f"ep{idx}"
        if podcast_notes:
            sources = "; ".join(n["subject"] for n in podcast_notes[:3])
            if len(podcast_notes) > 3:
                sources += f"; +{len(podcast_notes) - 3} more"
            podcast = {
                "source": sources,
                "text": podcast_notes[0]["excerpt"][:1500],
                "highlights": None,
                "episodes": [],
            }

        # Turn podcast-mentioned tickers into full trade setups: extract the
        # ideas, run each through the same TA engine, chart them.
        podcast_setups = []
        if podcast_notes:
            picked = {s["ticker"] for s in summaries}
            max_setups = (cfg.get("research") or {}).get("podcast_max_setups", 3)
            for idea in llm_extract_podcast_ideas(podcast_notes):
                if len(podcast_setups) >= max_setups:
                    break
                tick = (idea.get("ticker") or "").upper().strip()
                if not tick or tick in picked:
                    continue
                try:
                    df = add_indicators(fetch_history(tick, cfg["lookback_days"]), ind)
                    chart_paths[tick] = render_chart(tick, df, ind, cfg["chart_days"], out_dir)
                    setup = {
                        "ticker": tick,
                        "thesis": idea.get("thesis", ""),
                        "sentiment": idea.get("sentiment", ""),
                        "score": bullish_score(df, ind),
                        "close": round(float(df["Close"].iloc[-1]), 2),
                        "support": support_level(df, ind),
                        "resistance": resistance_level(df, ind),
                        "earnings_days": days_to_earnings(tick),
                        "below_200": below_200day(df, ind),
                    }
                    setup["rr"] = reward_risk(
                        setup["close"], setup["support"], setup["resistance"]
                    )
                    vol = vol_context(tick, df)
                    setup["vol"] = vol
                    setup["options_note"] = options_note(
                        vol, setup["support"], setup["resistance"], setup["earnings_days"]
                    )
                    setup["flow_note"] = flow_note(vol.get("flow"))
                    setup["plan"] = default_setup_plan(setup)
                    podcast_setups.append(setup)
                    picked.add(tick)
                    log(f"{tick}: podcast trade setup prepared (score {setup['score']})")
                except Exception as exc:
                    log(f"{tick}: podcast setup skipped — {exc}")

        tweets = fetch_tweets(
            cfg, [s["ticker"] for s in summaries] + [st["ticker"] for st in podcast_setups]
        )
        llm_result = llm_commentary(
            picks_context,
            research_notes,
            overview["stats"],
            podcast_notes,
            [{k: v for k, v in st.items() if k != "plan"} for st in podcast_setups],
            tweets,
        )
        if llm_result:
            per_ticker, llm_market, llm_podcast, llm_setups, llm_episodes = llm_result
            for s in summaries:
                item = per_ticker.get(s["ticker"])
                if item:
                    s["commentary"] = item["commentary"]
                    s["actionable"] = item["actionable"]
            if llm_market:
                market_text = llm_market
            if llm_podcast and podcast:
                podcast["highlights"] = llm_podcast
            if podcast and llm_episodes:
                # keep episodes Claude found substantive, in feed order
                episodes = []
                for note in podcast_notes:
                    item = llm_episodes.get(note.get("id"))
                    if not item:
                        continue
                    episodes.append({**note, **item})
                podcast["episodes"] = episodes
                log(f"podcast section: {len(episodes)} episode write-up(s)")
            for st in podcast_setups:
                item = llm_setups.get(st["ticker"])
                if item:
                    st["analysis"] = item.get("analysis", "")
                    if item.get("plan"):
                        st["plan"] = item["plan"]
        if market_text or market_path:
            market = {"text": market_text, "path": market_path}

        # Weekly Scorecard: Friday afternoon recaps grade the ledger.
        now_et = datetime.now(EASTERN)
        if (now_et.weekday() == 4 and now_et.hour >= 12) or os.environ.get(
            "TA_FORCE_SCORECARD"
        ):
            try:
                scorecard = build_scorecard(cfg)
                if scorecard:
                    log("scorecard: graded ledger included in this recap")
                else:
                    log("scorecard: no gradeable ledger entries yet")
            except Exception as exc:
                log(f"scorecard failed — {exc}")

    msg = build_email(
        cfg, summaries, chart_paths, heading, market, podcast, podcast_setups,
        positions, scorecard,
    )
    if args.dry_run or not cfg["email"].get("enabled", True):
        # Browser-viewable preview: same HTML, but images point at the local
        # PNGs instead of cid: attachments.
        preview = out_dir / "email_preview.html"
        cids = {t: p.name for t, p in chart_paths.items()}
        preview_market = None
        if market:
            preview_market = {"text": market.get("text", "")}
            if market.get("path"):
                preview_market["cid"] = market["path"].name
        generated = f"{datetime.now(EASTERN):%Y-%m-%d %H:%M %Z}"
        preview.write_text(
            build_html(
                summaries, cids, generated, heading, preview_market, podcast,
                podcast_setups, positions, scorecard,
            ).replace("cid:", "")
        )
        log(f"DRY RUN — email not sent. Preview: {preview}")
        log(f"DRY RUN — would send to: {', '.join(cfg['email']['to_addrs'])} "
            f"with subject '{msg['Subject']}'")
    else:
        send_email(cfg, msg)
        log(f"email sent to {', '.join(cfg['email']['to_addrs'])}")
        if args.scan:
            try:
                append_picks_ledger(summaries, podcast_setups)
            except Exception as exc:
                log(f"ledger append failed — {exc}")

    if failures:
        log(f"completed with failures: {', '.join(failures)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
