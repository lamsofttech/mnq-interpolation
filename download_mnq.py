"""Download MNQ=F 1-minute bars from Yahoo Finance into data/mnq_1m_raw.csv.

Yahoo keeps 1-minute bars for only about 30 days and serves at most 8 days per
request, so this pulls the last 29 days in 7-day chunks. Rerunning it later adds
any new days to the saved file without dropping the old ones, so the sample can
only grow. Every run is recorded in data/download_log.json for reproducibility.

Usage (from this folder):  python download_mnq.py
"""
import json
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKER = "MNQ=F"
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_FILE = DATA_DIR / "mnq_1m_raw.csv"
LOG_FILE = DATA_DIR / "download_log.json"


def download_chunks(days_back=29, chunk_days=7):
    end = datetime.now(timezone.utc) + timedelta(days=1)
    earliest = end - timedelta(days=days_back + 1)
    frames = []
    chunk_end = end
    while chunk_end > earliest:
        chunk_start = max(chunk_end - timedelta(days=chunk_days), earliest)
        bars = yf.download(
            TICKER,
            start=chunk_start.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
            interval="1m",
            auto_adjust=False,
            progress=False,
        )
        print(f"  {chunk_start:%Y-%m-%d} to {chunk_end:%Y-%m-%d}: {len(bars)} bars")
        if len(bars):
            if isinstance(bars.columns, pd.MultiIndex):
                bars.columns = bars.columns.get_level_values(0)
            bars.index = bars.index.tz_convert("UTC")
            frames.append(bars[["Open", "High", "Low", "Close", "Volume"]])
        chunk_end = chunk_start
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Downloading {TICKER} 1-minute bars from Yahoo Finance...")
    new = download_chunks()
    if new.empty:
        print("No data came back. Check your internet connection and try again in a few minutes.")
        sys.exit(1)

    if RAW_FILE.exists():
        old = pd.read_csv(RAW_FILE, index_col="Datetime")
        old.index = pd.to_datetime(old.index, utc=True)
        combined = pd.concat([old, new])
    else:
        combined = new
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_csv(RAW_FILE, index_label="Datetime")

    log = json.loads(LOG_FILE.read_text()) if LOG_FILE.exists() else []
    log.append({
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": TICKER,
        "interval": "1m",
        "bars_downloaded": int(len(new)),
        "bars_in_file": int(len(combined)),
        "first_bar_utc": str(combined.index.min()),
        "last_bar_utc": str(combined.index.max()),
        "yfinance": yf.__version__,
        "pandas": pd.__version__,
        "python": platform.python_version(),
    })
    LOG_FILE.write_text(json.dumps(log, indent=2))

    print(f"Saved {len(combined)} bars to {RAW_FILE}")
    print(f"Covers {combined.index.min()} to {combined.index.max()} (UTC)")


if __name__ == "__main__":
    main()
