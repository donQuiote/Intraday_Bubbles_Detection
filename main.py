import matplotlib.pyplot as plt

import test
import utils.data_handler_polars
import utils.easy_plotter
from Strategies import momentum, excess_volume, volatility_trading_strategy
from strategy_runner import apply_strategy, build_strat_df, best_strat_finder, strat_of_strats, best_return_per_day
from strategy_runner import apply_strategy, build_strat_df, best_strat_finder, strat_of_strats, best_of_best

plt.rcParams.update({
    'text.usetex': True,
    'font.size': 14,               # Set default font size
    'axes.titlesize': 16,          # Title font size
    'axes.labelsize': 16,          # Axis labels font size
    'xtick.labelsize': 12,         # X-tick labels font size
    'ytick.labelsize': 12,         # Y-tick labels font size
    'legend.fontsize': 12,         # Legend font size
    'grid.alpha': 0.5,             # Set grid opacity (0: transparent, 1: opaque)
    'grid.linestyle': '--',        # Set grid line style (e.g., dashed)
    'grid.color': 'gray',          # Set grid line color
    'axes.grid': True,             # Enable grid for all axes
    'axes.spines.top': False,      # Remove the top spine
    'axes.spines.right': False,    # Remove the right spine
    'axes.spines.left': False,     # Optionally remove the left spine
    'axes.spines.bottom': False,   # Optionally remove the bottom spine
})

YEARS = "*"
MONTHS = "*"
TICKERS = ['EXC', 'DVN', 'IBM', 'GD', 'DIS', 'MON', 'BAC', 'CVS', 'BMY', 'PEP', 'MCD', 'HNZ', 'GE', 'DOW', 'APA', 'AA', 'COP', 'WFC', 'WMT', 'UNP', 'FCX', 'TWX', 'GS', 'T', 'MDT', 'KFT', 'CL', 'ALL', 'DD', 'FDX', 'VZ', 'JNJ', 'NOV', 'HPQ', 'ORCL', 'WMB', 'V', 'AEP', 'XRX', 'EMC', 'HON', 'ABT', 'MMM', 'MSFT', 'HD', 'MO', 'COF', 'USB', 'PG', 'MA', 'UPS', 'MS', 'JPM', 'LOW', 'RTN', 'CVX', 'TXN', 'ETR', 'UTX', 'BA', 'LMT', 'WY', 'AVP', 'MRK', 'AXP', 'PM', 'SLB', 'PFE', 'WAG', 'SO', 'BK', 'F', 'UNH', 'EMR', 'XOM', 'BHI', 'OXY', 'TGT', 'NSC', 'KO', 'CAT', 'C', 'HAL', 'BAX', 'MET', 'NKE', 'S']

#################
demo_project = False
#################
plot_data = True
find_error = False
#################
plot_eda = True
plot_stratOstrat = True
rapide = False
#################
load_data = False
#################
gen_strategies = False
#################
apply_strat = False #Only chose one of the following otherwise the last will be chosen
mom = False
excess_vol = False
volatility = False
strategize = False
#################
strats = False

if demo_project:
    test.test()

#################
# Loads the data and merges the bbo and trade files -> creation of cleaned data
#################
if load_data:
    # print(f"Loading data for {", ".join(TICKERS)}")

    for idx, ticker in enumerate(TICKERS[::-1]):
        print(ticker)
        print("+" * 214)
        print(f"Handling file {ticker} ({idx + 1}/{len(TICKERS)}).")

        files_bbo, files_trade = utils.data_handler_polars.handle_files(ticker=ticker, year=YEARS, month=MONTHS,
                                                                        force_return_list=True)
        concatenated_df = utils.data_handler_polars.read_data(files_bbo=files_bbo, files_trade=files_trade,
                                                              ticker=ticker)

if gen_strategies:
    #strategy hyperparameters
    STLT = [(5,50,20,2000),(5,50,5,150),(5,10,5,100),(10,100,100,1500)]
    for stlt in STLT:
        #################
        # Momentum Setup
        #################
        if mom:
            strategy = momentum.momentum_price
            parameters = {
                "short_window": stlt[0],
                "long_window": stlt[1],
                "plot": False
            }
            s = parameters["short_window"]
            l = parameters["long_window"]
            param_names = f"_s{s}_l{l}"

        #################
        # Excess Volume Setup
        #################
        if excess_vol:
            strategy = excess_volume.momentum_excess_vol
            parameters = \
                {"short_window_price": stlt[2], "long_window_price": stlt[3],
                 "short_window_volume": stlt[0], "long_window_volume": stlt[1],
                 "plot": False
                 }
            s = parameters["short_window_price"]
            l = parameters["long_window_price"]
            k = parameters["short_window_volume"]
            r = parameters["long_window_volume"]

            param_names = f"_ps{s}_pl{l}_vs{k}_vl{r}"

        #################
        # Volatility strategy Setup
        #################
        if volatility:
            strategy = volatility_trading_strategy.volatility_trading_strategy
            parameters = \
                {"short_window": stlt[0], "long_window": stlt[1],
                 "plot": False
                 }
            s = parameters["short_window"]
            l = parameters["long_window"]
            param_names = f"_s{s}_l{l}"

        #################
        # Apply a certain strategy on cleaned data -> creation of strategies daily returns
        #################
        if apply_strat:
            apply_strategy(strategy=strategy, param_names=param_names, parameters = parameters)

        #################
        # Creates a dataframe of daily returns for all tickers and dates available
        #################
        if strategize:
            build_strat_df(strategy=strategy, param_names=param_names)

if strats:
    strategy = excess_volume.momentum_excess_vol
    strat_dict = best_strat_finder(intra_strat=True,strategy=strategy)
    print(strat_dict)
    #strat_of_strats()
    utils.easy_plotter.plot_best_strategy()
    df = best_of_best()
    utils.easy_plotter.plot_best_of_best(df)


if plot_data:
    utils.easy_plotter.plot_tickers_dates(bbo=True)
    ticker = 'RTN'
    df_average = utils.easy_plotter.daily_average_volume(ticker)
    utils.easy_plotter.plot_daily_average_volume_single_stock(df_average, ticker=ticker)
    df = best_of_best()

    file_name = "data/clean/APA/2004/02_bbo_trade.csv"
    import polars as pl

    df = pl.scan_csv(file_name)
    parameters_mom = {"short_window": 100, "long_window": 1000, "plot": True}
    parameters_volatility = {"short_window_price": 100, "long_window_price": 1000, "short_window_volume": 100, "long_window_volume": 1000, "plot": True}
    momentum.momentum_price(df, parameters=parameters_mom)
    volatility_trading_strategy.volatility_trading_strategy(df, parameters=parameters_mom)
    excess_volume.momentum_excess_vol(df, parameters=parameters_volatility)

if plot_eda:
    for t in ['EXC', 'DVN', 'IBM', 'GD', 'DIS', 'MON', 'BAC', 'CVS', 'BMY', 'PEP']:
        utils.easy_plotter.plot_mean_vs_median_traded_volume(t)
        utils.easy_plotter.plot_intraday_spread(t)

if plot_stratOstrat:
    data = "data/optimum_strategy_tracker.csv"
    dic = best_strat_finder(intra_strat=False)

    utils.easy_plotter.plot_tracker_best_strat_families(data, dict_trad=dic)
    utils.easy_plotter.plot_tracker_best_strat(data, dict_trad=dic)
    utils.easy_plotter.plot_best_returns()
    utils.easy_plotter.plot_returns()

    utils.easy_plotter.generate_latex_table(dic)

    utils.easy_plotter.plot_tracker_best_strat_families(data, dict_trad=dic)
    utils.easy_plotter.plot_tracker_best_strat(data, dict_trad=dic)
    utils.easy_plotter.plot_best_returns()
    utils.easy_plotter.plot_returns()

    utils.easy_plotter.plot_tracker_best_strat_periods(data, dict_trad=dic)


# if rapide:
#     data = "data/optimum_strategy_tracker.csv"
#     dic = {-1: 'momentum_excess_vol__ps400_pl5000_vs400_vl5000_df.csv', 0: 'momentum_price__s10_l400_df.csv',
#            1: 'momentum_excess_vol__ps200_pl4000_vs200_vl4000_df.csv', 2: 'momentum_price__s200_l4000_df.csv',
#            3: 'momentum_price__s30_l1200_df.csv', 4: 'momentum_excess_vol__ps5_pl200_vs5_vl200_df.csv',
#            5: 'momentum_price__s400_l5000_df.csv', 6: 'momentum_excess_vol__ps30_pl1200_vs30_vl1200_df.csv',
#            7: 'momentum_price__s100_l1500_df.csv', 8: 'momentum_price__s50_l2000_df.csv',
#            9: 'momentum_excess_vol__ps100_pl2000_vs100_vl2000_df.csv', 10: 'momentum_price__s5_l200_df.csv',
#            11: 'momentum_excess_vol__ps10_pl400_vs10_vl400_df.csv'}
#     utils.easy_plotter.plot_tracker_best_strat_periods(data, dict_trad=dic)