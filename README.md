# TA Chart Agent v2

Daily technical-analysis briefing: fetches OHLCV data from Yahoo Finance,
renders a four-panel chart per ticker (candlesticks + SMAs, volume, RSI,
MACD), and emails an HTML summary with the charts inline at 6:30pm ET on
weekdays.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## config.yaml

| Key | What it does |
|---|---|
| `tickers` | Yahoo Finance symbols to analyze (stocks, ETFs, `^GSPC`, `BTC-USD`, ...) |
| `lookback_days` | Calendar days of history fetched; keep well above the slowest indicator window |
| `chart_days` | Trading days shown on each chart |
| `indicators.*` | SMA fast/slow windows, RSI period, MACD fast/slow/signal |
| `indicators.fibonacci` | Draw Fibonacci retracement levels (23.6/38.2/50/61.8/78.6%) on the price panel, anchored on the charted window's swing high/low |
| `indicators.trendlines` | Auto-detect and draw support/resistance trendlines through swing pivots, only where price has respected the line |
| `output_dir` | Where PNG charts and `email_preview.html` are written |
| `email.smtp_host` / `smtp_port` / `use_ssl` | SMTP server; `587` + `use_ssl: false` = STARTTLS, `465` + `use_ssl: true` = implicit SSL |
| `email.username` / `from_addr` / `to_addrs` | Login user, sender, recipients |
| `email.enabled` | Set `false` to always behave like a dry run |

**The SMTP password is never stored in config.yaml.** It is read from the
`TA_SMTP_PASSWORD` environment variable at send time.

### Gmail credentials

Gmail blocks plain passwords for SMTP, so you need an App Password:

1. Enable 2-Step Verification on the account (required for app passwords).
2. Go to <https://myaccount.google.com/apppasswords>, create an app password
   named e.g. `ta-chart-agent`, and copy the 16-character code.
3. Locally: `export TA_SMTP_PASSWORD="abcd efgh ijkl mnop"` (spaces are fine).
4. For the scheduled GitHub Actions run: repo **Settings → Secrets and
   variables → Actions → New repository secret**, name `TA_SMTP_PASSWORD`.

## Usage

```bash
# Dry run: fetch data, render charts and output/email_preview.html, send nothing
python ta_briefing_v2.py --dry-run

# Real run (requires TA_SMTP_PASSWORD)
python ta_briefing_v2.py

# Ad-hoc ticker override
python ta_briefing_v2.py --dry-run --tickers TSLA AMD
```

## Scheduling — 6:30pm ET weekdays

The committed workflow `.github/workflows/ta_briefing.yml` runs the briefing
automatically. GitHub cron is UTC-only, so it fires at both 22:30 and 23:30
UTC on weekdays and a guard step keeps only the run that lands in the 6pm
hour in `America/New_York`, which handles daylight-saving transitions.
Add the `TA_SMTP_PASSWORD` secret (above) and it is live; you can also
trigger it manually from the Actions tab (with an optional dry-run flag).

To run it from your own machine instead, add this crontab entry
(`crontab -e`):

```cron
CRON_TZ=America/New_York
30 18 * * 1-5 cd /path/to/TA-Chart-Agent && .venv/bin/python ta_briefing_v2.py >> briefing.log 2>&1
```

(If your cron daemon lacks `CRON_TZ` support, set the schedule in your
machine's local equivalent of 6:30pm ET.)

---
*Generated briefings are informational only — not investment advice.*
