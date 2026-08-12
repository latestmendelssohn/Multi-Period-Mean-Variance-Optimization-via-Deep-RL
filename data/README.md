# public sample market data

`download_market_data.py` fetches six daily CSV files from the public GitHub repository [taiwaich/stocks](https://github.com/taiwaich/stocks): Apple, Facebook, Nasdaq, Netflix, Twitter, and Yahoo. The source files contain OHLC prices, volume, and `Adj Close`. The script keeps only the dividend-adjusted close and aligns the dates into `data/market_prices.csv`.

The six files share 419 dates from 2013-11-07 through 2015-07-09. This is enough for the current 60-day lookback and 150-period example, but it is not a ten-year dataset. The source repository does not display a license or a stronger redistribution note, so the raw files and generated CSV are not committed here. Check the source terms before redistributing the data.

From the project directory:

```text
python download_market_data.py
python backtest.py --csv data/market_prices.csv --kind prices --periods 150 --cost-bps 10
```
