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

import mplfinance as mpf
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


def render_chart(ticker: str, df: pd.DataFrame, ind: dict, chart_days: int, out_dir: Path) -> Path:
    plot_df = df.dropna(subset=[f"SMA{ind['sma_slow']}", "RSI"]).tail(chart_days)
    if plot_df.empty:
        raise RuntimeError(f"{ticker}: not enough history to plot (increase lookback_days)")

    fast, slow = ind["sma_fast"], ind["sma_slow"]
    hist_colors = ["#2e7d32" if v >= 0 else "#c62828" for v in plot_df["MACDHist"]]
    rsi_band = lambda level: pd.Series(level, index=plot_df.index)

    addplots = [
        mpf.make_addplot(plot_df[f"SMA{fast}"], panel=0, color="#1565c0", width=1.0, label=f"SMA{fast}"),
        mpf.make_addplot(plot_df[f"SMA{slow}"], panel=0, color="#ef6c00", width=1.0, label=f"SMA{slow}"),
        mpf.make_addplot(plot_df["RSI"], panel=2, color="#6a1b9a", width=1.0, ylabel="RSI", ylim=(0, 100)),
        mpf.make_addplot(rsi_band(70), panel=2, color="#9e9e9e", width=0.7, linestyle="--"),
        mpf.make_addplot(rsi_band(30), panel=2, color="#9e9e9e", width=0.7, linestyle="--"),
        mpf.make_addplot(plot_df["MACD"], panel=3, color="#1565c0", width=1.0, ylabel="MACD"),
        mpf.make_addplot(plot_df["MACDSignal"], panel=3, color="#ef6c00", width=1.0),
        mpf.make_addplot(plot_df["MACDHist"], panel=3, type="bar", color=hist_colors, alpha=0.6),
    ]

    out_path = out_dir / f"{ticker.replace('^', '').replace('=', '_')}_4panel.png"
    last = plot_df.iloc[-1]
    title = f"\n{ticker}  {plot_df.index[-1]:%Y-%m-%d}  close {last['Close']:,.2f}"
    style = mpf.make_mpf_style(base_mpf_style="yahoo", gridstyle=":", rc={"font.size": 9})
    mpf.plot(
        plot_df,
        type="candle",
        style=style,
        volume=True,
        addplot=addplots,
        panel_ratios=(6, 2, 2, 2),
        figsize=(12, 10),
        title=title,
        tight_layout=True,
        savefig=dict(fname=str(out_path), dpi=110),
    )
    return out_path


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


def build_html(summaries: list[dict], cids: dict[str, str], generated: str) -> str:
    rows = []
    for s in summaries:
        chg = s["change_pct"]
        color = "#2e7d32" if chg >= 0 else "#c62828"
        rows.append(
            f"<tr><td><b>{s['ticker']}</b></td><td>{s['close']}</td>"
            f"<td style='color:{color}'>{chg:+.2f}%</td>"
            f"<td>{s['rsi']} ({s['rsi_state']})</td><td>{s['macd_state']}</td>"
            f"<td>{s['trend']} {s['slow_name']}</td></tr>"
        )
    charts = "".join(
        f"<h3 style='font-family:sans-serif'>{t}</h3>"
        f"<img src='cid:{cid}' width='860' style='max-width:100%'/>"
        for t, cid in cids.items()
    )
    return f"""<html><body style="font-family:sans-serif">
<h2>Daily TA Briefing — {summaries[0]['date']}</h2>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-size:14px">
<tr style="background:#eceff1"><th>Ticker</th><th>Close</th><th>1d %</th>
<th>RSI(14)</th><th>MACD</th><th>Trend</th></tr>
{''.join(rows)}
</table>
{charts}
<p style="color:#777;font-size:12px">Generated {generated} by TA Chart Agent v2.
Not investment advice.</p>
</body></html>"""


def build_email(cfg: dict, summaries: list[dict], chart_paths: dict[str, Path]) -> EmailMessage:
    email_cfg = cfg["email"]
    msg = EmailMessage()
    msg["Subject"] = f"{email_cfg['subject_prefix']} {summaries[0]['date']}"
    msg["From"] = email_cfg["from_addr"]
    msg["To"] = ", ".join(email_cfg["to_addrs"])
    msg.set_content("Your email client does not support HTML. See attached charts.")

    cids = {t: make_msgid(domain="ta-chart-agent")[1:-1] for t in chart_paths}
    generated = f"{datetime.now(EASTERN):%Y-%m-%d %H:%M %Z}"
    msg.add_alternative(build_html(summaries, cids, generated), subtype="html")
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
        "--dry-run",
        action="store_true",
        help="fetch data and render charts + email preview, but send nothing",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    tickers = args.tickers or cfg["tickers"]
    ind = cfg["indicators"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, chart_paths, failures = [], {}, []
    for ticker in tickers:
        try:
            log(f"{ticker}: fetching {cfg['lookback_days']}d of history...")
            df = add_indicators(fetch_history(ticker, cfg["lookback_days"]), ind)
            chart_paths[ticker] = render_chart(ticker, df, ind, cfg["chart_days"], out_dir)
            summaries.append(summarize(ticker, df, ind))
            log(f"{ticker}: chart written to {chart_paths[ticker]}")
        except Exception as exc:  # one bad ticker must not kill the briefing
            failures.append(ticker)
            log(f"{ticker}: FAILED — {exc}")

    if not summaries:
        log("no tickers succeeded; aborting")
        return 1

    msg = build_email(cfg, summaries, chart_paths)
    if args.dry_run or not cfg["email"].get("enabled", True):
        # Browser-viewable preview: same HTML, but images point at the local
        # PNGs instead of cid: attachments.
        preview = out_dir / "email_preview.html"
        cids = {t: p.name for t, p in chart_paths.items()}
        generated = f"{datetime.now(EASTERN):%Y-%m-%d %H:%M %Z}"
        preview.write_text(build_html(summaries, cids, generated).replace("cid:", ""))
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
