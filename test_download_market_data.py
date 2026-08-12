"""Offline tests for the public-data download and alignment helper."""

import csv
import tempfile
import unittest
from pathlib import Path

from download_market_data import FILES, build_price_csv, parse_adjusted_close


class TestDownloadMarketData(unittest.TestCase):
    def test_aligns_and_sorts_source_prices(self):
        raw = {
            symbol: "Date,Adj Close\n2020-01-02,2\n2020-01-01,1\n"
            for symbol in FILES
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prices.csv"
            rows = build_price_csv(raw, output)
            with output.open(newline="", encoding="utf-8") as handle:
                records = list(csv.reader(handle))

        self.assertEqual(rows, 2)
        self.assertEqual(records[0], ["date", *FILES])
        self.assertEqual(records[1][0], "2020-01-01")
        self.assertEqual(records[2][0], "2020-01-02")
        self.assertEqual(records[1][1:], ["1.0"] * len(FILES))

    def test_parser_reads_adjusted_close(self):
        prices = parse_adjusted_close("Date,Open,Adj Close\n2020-01-01,10,9.5\n")
        self.assertEqual(prices, {"2020-01-01": 9.5})

    def test_parser_rejects_wrong_columns(self):
        with self.assertRaises(ValueError):
            parse_adjusted_close("Date,Close\n2020-01-01,10\n")


if __name__ == "__main__":
    unittest.main()
