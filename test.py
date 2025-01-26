import matplotlib.pyplot as plt
import polars as pl

import utils.data_handler_polars
import utils.easy_plotter
from Strategies import momentum, excess_volume, volatility_trading_strategy
from strategy_runner import apply_strategy, build_strat_df, best_strat_finder, strat_of_strats, best_return_per_day

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


def test():
    print("Loading the data")
    test_tickers = ['AA', 'ABT']
    for idx, ticker in enumerate(test_tickers):
        print("+" * 214)
        print(f"Handling file {ticker} ({idx + 1}/{len(test_tickers)}).")
        files_bbo, files_trade = utils.data_handler_polars.handle_files(ticker=ticker, year=[2004, 2005], month="*", force_return_list=True)
        utils.data_handler_polars.read_data(files_bbo=files_bbo, files_trade=files_trade, ticker=ticker)

    STLT = [(5, 50), (10, 400)]
    for stlt in STLT:

        strategy = momentum.momentum_price
        parameters = {
            "short_window": stlt[0],
            "long_window": stlt[1],
            "plot": False
        }
        s = parameters["short_window"]
        l = parameters["long_window"]
        param_names = f"_s{s}_l{l}"
        apply_strategy(strategy=strategy, param_names=param_names, parameters = parameters)

        # build_strat_df(strategy=strategy, param_names=param_names)

        strategy = excess_volume.momentum_excess_vol
        parameters = \
            {"short_window_price": stlt[0], "long_window_price": stlt[1],
             "short_window_volume": stlt[0], "long_window_volume": stlt[1],
             "plot": False
             }
        s = parameters["short_window_price"]
        l = parameters["long_window_price"]
        k = parameters["short_window_volume"]
        r = parameters["long_window_volume"]

        param_names = f"_ps{s}_pl{l}_vs{k}_vl{r}"
        apply_strategy(strategy=strategy, param_names=param_names, parameters = parameters)
        build_strat_df(strategy=strategy, param_names=param_names)

        strategy = volatility_trading_strategy.volatility_trading_strategy
        parameters = \
            {"short_window": stlt[0], "long_window": stlt[1],
             "plot": False
             }
        s = parameters["short_window"]
        l = parameters["long_window"]
        param_names = f"_s{s}_l{l}"
        apply_strategy(strategy=strategy, param_names=param_names, parameters = parameters)
        build_strat_df(strategy=strategy, param_names=param_names)


    strat_dict = best_strat_finder()
    print("Dict strat", strat_dict)
    strat_of_strats()
    best_return_per_day()
    utils.easy_plotter.plot_best_strategy()


    for t in test_tickers:
        utils.easy_plotter.plot_mean_vs_median_traded_volume(t)
        utils.easy_plotter.plot_intraday_spread(t)


    data = "data/optimum_strategy_tracker.csv"

    # Plot the heatmaps for best strategy
    utils.easy_plotter.plot_tracker_best_strat_families(data, dict_trad=strat_dict)
    utils.easy_plotter.plot_tracker_best_strat(data, dict_trad=strat_dict)
    # Plot the translation table
    utils.easy_plotter.generate_latex_table(strat_dict)

    utils.easy_plotter.plot_best_strategy()

    utils.easy_plotter.plot_tracker_best_strat_families(data, dict_trad=strat_dict)
    utils.easy_plotter.plot_tracker_best_strat(data, dict_trad=strat_dict)

    # Plot the time series of the strategies
    utils.easy_plotter.plot_best_returns()
    utils.easy_plotter.plot_returns()