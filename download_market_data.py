"""Download the small public stock-price sample used by the real-data example."""

from __future__ import annotations

import argparse
import csv
import io
import urllib.request
from pathlib import Path

SOURCE_BASE = "https://raw.githubusercontent.com/taiwaich/stocks/main/"
FILES = {
    "AAPL": "apple_daily.csv",
    "FB": "facebook_daily.csv",
    "NASDAQ": "nasdaq_daily.csv",
    "NFLX": "netflix_daily.csv",
    "TWTR": "twitter_daily.csv",
    "YHOO": "yahoo_daily.csv",
}


def parse_adjusted_close(text: str) -> dict[str, float]:
    """Read one source CSV and return its adjusted close by date."""
    reader = csv.DictReader(io.StringIO(text))
    required = {"Date", "Adj Close"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("source CSV must contain Date and Adj Close columns")

    prices = {}
    for row_number, row in enumerate(reader, start=2):
        try:
            prices[row["Date"]] = float(row["Adj Close"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid source row {row_number}") from error
    if not prices:
        raise ValueError("source CSV contains no prices")
    return prices


def build_price_csv(raw_files: dict[str, str], output: Path) -> int:
    """Write aligned adjusted-close prices and return the shared row count."""
    if set(raw_files) != set(FILES):
        raise ValueError(f"expected source symbols: {tuple(FILES)}")

    series = {symbol: parse_adjusted_close(text) for symbol, text in raw_files.items()}
    common_dates = sorted(set.intersection(*(set(values) for values in series.values())))
    if len(common_dates) < 2:
        raise ValueError("source files do not share at least two dates")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("date", *FILES))
        for date in common_dates:
            writer.writerow((date, *(series[symbol][date] for symbol in FILES)))
    return len(common_dates)


def download_prices(output: Path = Path("data/market_prices.csv")) -> int:
    """Fetch the source files and write the aligned local price CSV."""
    raw_files = {}
    for symbol, filename in FILES.items():
        with urllib.request.urlopen(SOURCE_BASE + filename, timeout=30) as response:
            raw_files[symbol] = response.read().decode("utf-8-sig")
    return build_price_csv(raw_files, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the public sample price data")
    parser.add_argument("--output", type=Path, default=Path("data/market_prices.csv"))
    args = parser.parse_args()
    rows = download_prices(args.output)
    print(f"wrote {rows} shared dates to {args.output}")
