import os
from itertools import chain

import numpy as np
import polars as pl
from tqdm import tqdm

import momentum
import utils.data_handler_polars
import volatility_trading_strategy
import extract_names

data_root = "/Users/gustavebesacier/Library/Mobile Documents/com~apple~CloudDocs/Documents/HEC/EPFL MA III/Financial big data/project/data/clean/APA/2004/02_bbo_trade.csv"

path = "./data/Raw/sp100_2004-8/bbo"
names = extract_names.extract_names(path)
print(names)

YEARS = "*"
MONTHS = "*"
TICKERS = names
data_root = "/Users/gustavebesacier/Library/Mobile Documents/com~apple~CloudDocs/Documents/HEC/EPFL MA III/Financial big data/project/data/clean/APA/2004/02_bbo_trade.csv"
TICKERS.pop(1)
TICKERS.pop(0)
TICKERS.pop(2)
TICKERS.pop(3)

load_data = True
mom = False

if load_data:
    print(f"Loading data for {", ".join(TICKERS)}")

    for ticker in TICKERS:
        files_bbo, files_trade = utils.data_handler_polars.handle_files(ticker=ticker, year=YEARS, month=MONTHS)
        concatenated_df = utils.data_handler_polars.read_data(files_bbo=files_bbo, files_trade=files_trade, ticker=ticker)
"""
dir = "/Users/gustavebesacier/Library/Mobile Documents/com~apple~CloudDocs/Documents/HEC/EPFL MA III/Financial big data/project/data/clean/APA"
list_files_test = os.listdir(dir)
ll = list()
for year in [i for i in list_files_test if not i.startswith('.')]:
    temp = os.path.join(dir, year)
    files = os.listdir(temp)
    temp_paths = list(map(lambda x: os.path.join(dir, year, x), files))
    ll.append(temp_paths)

list_files_test = list(chain(*ll))

if mom:
    grid_short = [10, 20, 50, 70, 100, 150, 200, 250, 300, 500, 1000, 2000]
    grid_long =  [20, 50, 70, 100, 150, 200, 250, 300, 500, 1000, 2000, 3000, 40000, 5000, 6000, 8000, 10000, 20000]

    mean_res, std_res, sr_res, comb_params = list(), list(), list(), list()

    for short in tqdm(grid_short):
        for long in grid_long:
            if long > short:
                parameters_mom = {"short_window": short, "long_window": long, "plot": False}

                def pipe(file_name):
                    df = pl.scan_csv(file_name)
                    return momentum.momentum_strat2(df, parameters=parameters_mom)

                mapped = map(pipe, list_files_test)
                con = pl.concat(mapped, parallel=True)

                d = con.collect()
                mean = d.select('return').mean().item()
                std  = d.select('return').std().item()
                pseudo_sharpie = mean/std
                mean_res.append(mean), std_res.append(std)
                sr_res.append(pseudo_sharpie)
                comb_params.append([short, long])

    best_sharpie = np.argmax(sr_res)
    
    print(f"Optimal parameters: \n"
      f" - Short {comb_params[best_sharpie][0]}\n"
      f" - Long  {comb_params[best_sharpie][1]}\n"
      f"Values: \n"
      f" - Return {mean_res[best_sharpie]*100:.2f}%\n"
      f" - Volati {std_res[best_sharpie]*100:.2f}%\n"
      f" - Sharpe {sr_res[best_sharpie]:.2f}\n")

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

"""