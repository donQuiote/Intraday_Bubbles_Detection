import matplotlib.pyplot as plt
import polars as pl

import utils.data_handler_polars
import utils.easy_plotter
from Strategies import momentum, excess_volume, volatility_trading_strategy
from strategy_runner import apply_strategy, build_strat_df

plt.rcParams.update({
    'text.usetex': True,
    'font.size': 14,         # Set default font size
    'axes.titlesize': 16,    # Title font size
    'axes.labelsize': 16,    # Axis labels font size
    'xtick.labelsize': 12,   # X-tick labels font size
    'ytick.labelsize': 12,   # Y-tick labels font size
    'legend.fontsize': 12,   # Legend font size
})

YEARS = "*"
MONTHS = "*"
TICKERS = ['EXC', 'DVN', 'IBM', 'GD', 'DIS', 'MON', 'BAC', 'CVS', 'BMY', 'PEP', 'MCD', 'HNZ', 'GE', 'DOW', 'APA', 'AA', 'COP', 'WFC', 'WMT', 'UNP', 'FCX', 'TWX', 'GS', 'T', 'MDT', 'KFT', 'CL', 'ALL', 'DD', 'FDX', 'VZ', 'JNJ', 'NOV', 'HPQ', 'ORCL', 'WMB', 'V', 'AEP', 'XRX', 'EMC', 'HON', 'ABT', 'MMM', 'MSFT', 'HD', 'MO', 'COF', 'USB', 'PG', 'MA', 'UPS', 'MS', 'JPM', 'LOW', 'RTN', 'CVX', 'TXN', 'ETR', 'UTX', 'BA', 'LMT', 'WY', 'AVP', 'MRK', 'AXP', 'PM', 'SLB', 'PFE', 'WAG', 'SO', 'BK', 'F', 'UNH', 'EMR', 'XOM', 'BHI', 'OXY', 'TGT', 'NSC', 'KO', 'CAT', 'C', 'HAL', 'BAX', 'MET', 'NKE', 'S']
data_root = "/Users/gustavebesacier/Library/Mobile Documents/com~apple~CloudDocs/Documents/HEC/EPFL MA III/Financial big data/project/data/clean/APA/2004/02_bbo_trade.csv"

load_data = False
mom = True
get_data = False
vol_strat = False
plot_data = False
find_error = False

#################
load_data = False
#################
apply_strat = True
#################
strategize = True
#strategy parameters
STLT = [(50,2000),(200,4000)]
for stlt in STLT:
    if mom:
        strategy = momentum.momentum_price
        parameters_mom = {
            "short_window": stlt[0],
            "long_window": stlt[1],
            "plot": False
        }
        s = parameters_mom["short_window"]
        l = parameters_mom["long_window"]
        param_names = f"_s{s}_l{l}"


    find_error = False
    excess_vol = False
    volatility = False

    if find_error:
        ticker = 'LOW'
        files_bbo, files_trade = utils.data_handler_polars.handle_files(ticker=ticker, year=[2007, 2005], month="*", force_return_list=True)
        utils.data_handler_polars.read_data(files_bbo, files_trade, ticker=ticker)

        # concatenated_df = utils.data_handler_polars.read_data(files_bbo=files_bbo, files_trade=files_trade, ticker=ticker)

    if plot_data:
        # utils.easy_plotter.plot_tickers_dates(bbo=True)
        ticker = 'RTN'
        df_average = utils.easy_plotter.daily_average_volume(ticker)
        utils.easy_plotter.plot_daily_average_volume_single_stock(df_average, ticker=ticker)


    #################
    # Loads the data and merges the bbo and trade files -> creation of cleaned data
    #################
    if load_data:
        #print(f"Loading data for {", ".join(TICKERS)}")

        for idx, ticker in enumerate(TICKERS[::-1]):
            print(ticker)
            print("+"*214)
            print(f"Handling file {ticker} ({idx+1}/{len(TICKERS)}).")

            files_bbo, files_trade = utils.data_handler_polars.handle_files(ticker=ticker, year=YEARS, month=MONTHS, force_return_list=True)
            concatenated_df = utils.data_handler_polars.read_data(files_bbo=files_bbo, files_trade=files_trade, ticker=ticker)

    #################
    # Apply a certain strategy on cleaned data -> creation of strategies daily returns
    #################
    if apply_strat:
        apply_strategy(strategy=strategy, param_names=param_names)

    #################
    # Creates a dataframe of daily returns for all tickers and dates available
    #################
    if strategize:
        build_strat_df(strategy=strategy, param_names=param_names)

#if mom:
    #parameters_mom = {"short_window": 100, "long_window": 1000, "plot": True}
    #df = pl.scan_csv(data_root)
    #daily_returns = momentum.momentum_price(df, parameters=parameters_mom)
    #print(daily_returns.collect())

if excess_vol:
    parameters_mom = \
        {"short_window_price": 10, "long_window_price": 200,
         "short_window_volume": 10, "long_window_volume": 200,
         "plot": True
         }
    df = pl.scan_csv(data_root)
    daily_returns = excess_volume.momentum_excess_vol(df, parameters=parameters_mom)

if volatility:
    parameters_mom = \
        {"short_window": 5, "long_window": 20,
         "plot": True
         }
    df = pl.scan_csv(data_root)
    daily_returns = volatility_trading_strategy.volatility_trading_strategy(df, parameters=parameters_mom)
