import polars as pl

import utils.data_handler_polars
import volatility_trading_strategy

data_root = "/Users/gustavebesacier/Library/Mobile Documents/com~apple~CloudDocs/Documents/HEC/EPFL MA III/Financial big data/project/data/clean/APA/2004/02_bbo_trade.csv"

TICKER = "ABT"
YEARS = [2004]
MONTHS = "*"

# TODO: HANDLE MSFT
get_data = False
vol_strat = True

params = {
    "short_window": 5, # IN SECONDS!
    "long_window": 100, # IN SECONDS!
    "volatility_threshold": 2.0,
    "plot": True
}

if get_data:
    for ticker in ["MMM", "COP", "S", "TWX", "UPS"]:
        files_bbo, files_trade = utils.data_handler_polars.handle_files(ticker=ticker, year=YEARS, month=MONTHS)
        concatenated_df = utils.data_handler_polars.read_data(files_bbo=files_bbo, files_trade=files_trade, ticker=ticker)

if vol_strat:
    df = pl.scan_csv(data_root)
    returns = volatility_trading_strategy.volatility_trading_strategy(df, parameters=params)

    print(returns.collect())
    print(f"Return over the period: {returns.collect()['return'].mean()*100:.2f}%.")