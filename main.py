import polars as pl

import momentum
import utils.data_handler_polars

YEARS = [2004]
MONTHS = "*"
TICKERS = ["MSFT", "PEP", "APA"]
data_root = "/Users/gustavebesacier/Library/Mobile Documents/com~apple~CloudDocs/Documents/HEC/EPFL MA III/Financial big data/project/data/clean/APA/2004/02_bbo_trade.csv"

load_data = False
mom = True

if load_data:
    print(f"Loading data for {", ".join(TICKERS)}")

    for ticker in TICKERS:
        files_bbo, files_trade = utils.data_handler_polars.handle_files(ticker=ticker, year=YEARS, month=MONTHS)
        concatenated_df = utils.data_handler_polars.read_data(files_bbo=files_bbo, files_trade=files_trade, ticker=ticker)

parameters_mom = {"short_window": 100,"long_window": 500,"plot":False}

if mom:
    # for
    df = pl.scan_csv(data_root)
    daily_returns = momentum.momentum_strat2(df, parameters=parameters_mom)
    print(daily_returns.collect())