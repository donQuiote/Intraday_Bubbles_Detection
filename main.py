import polars as pl

import momentum
import utils.data_handler_polars
import utils.easy_plotter

YEARS = "*"
MONTHS = "*"
TICKERS = ['EXC', 'DVN', 'IBM', 'GD', 'DIS', 'MON', 'BAC', 'CVS', 'BMY', 'PEP', 'MCD', 'HNZ', 'GE', 'DOW', 'APA', 'AA', 'COP', 'WFC', 'WMT', 'UNP', 'FCX', 'TWX', 'GS', 'T', 'MDT', 'KFT', 'CL', 'ALL', 'DD', 'FDX', 'VZ', 'JNJ', 'NOV', 'HPQ', 'ORCL', 'WMB', 'V', 'AEP', 'XRX', 'EMC', 'HON', 'ABT', 'MMM', 'MSFT', 'HD', 'MO', 'COF', 'USB', 'PG', 'MA', 'UPS', 'MS', 'JPM', 'LOW', 'RTN', 'CVX', 'TXN', 'ETR', 'UTX', 'BA', 'LMT', 'WY', 'AVP', 'MRK', 'AXP', 'PM', 'SLB', 'PFE', 'WAG', 'SO', 'BK', 'F', 'UNH', 'EMR', 'XOM', 'BHI', 'OXY', 'TGT', 'NSC', 'KO', 'CAT', 'C', 'HAL', 'BAX', 'MET', 'NKE', 'S']
data_root = "/Users/gustavebesacier/Library/Mobile Documents/com~apple~CloudDocs/Documents/HEC/EPFL MA III/Financial big data/project/data/clean/APA/2004/02_bbo_trade.csv"

load_data = False
mom = False
plot_data = False
find_error = True

if find_error:
    ticker = 'LOW'
    files_bbo, files_trade = utils.data_handler_polars.handle_files(ticker=ticker, year=[2007, 2005], month="*", force_return_list=True)
    utils.data_handler_polars.read_data(files_bbo, files_trade, ticker=ticker)

    # concatenated_df = utils.data_handler_polars.read_data(files_bbo=files_bbo, files_trade=files_trade, ticker=ticker)

if plot_data:
    # utils.easy_plotter.plot_tickers_dates(bbo=True)
    ticker = 'APA'
    df_average = utils.easy_plotter.daily_average_volume(ticker)
    utils.easy_plotter.plot_daily_average_volume_single_stock(df_average, ticker=ticker)

if load_data:
    print(f"Loading data for {", ".join(TICKERS)}")

    for idx, ticker in enumerate(TICKERS):
        print()
        print("+"*214)
        print(f"Handling file {ticker} ({idx+1}/{len(TICKERS)}).")

        files_bbo, files_trade = utils.data_handler_polars.handle_files(ticker=ticker, year=YEARS, month=MONTHS, force_return_list=True)
        concatenated_df = utils.data_handler_polars.read_data(files_bbo=files_bbo, files_trade=files_trade, ticker=ticker)

if mom:
    parameters_mom = {"short_window": 100, "long_window": 1000, "plot": True}
    df = pl.scan_csv(data_root)
    daily_returns = momentum.momentum_strat2(df, parameters=parameters_mom)
    print(daily_returns.collect())

# dir = "/Users/gustavebesacier/Library/Mobile Documents/com~apple~CloudDocs/Documents/HEC/EPFL MA III/Financial big data/project/data/clean/APA"
# list_files_test = os.listdir(dir)
# ll = list()
# for year in [i for i in list_files_test if not i.startswith('.')]:
#     temp = os.path.join(dir, year)
#     files = os.listdir(temp)
#     temp_paths = list(map(lambda x: os.path.join(dir, year, x), files))
#     ll.append(temp_paths)
#
# list_files_test = list(chain(*ll))

# if mom:
#     grid_short = [0, 5, 10, 20, 50, 100, 150, 200, 250, 500, 1000, 2000]
#     grid_long =  [10, 20, 50, 100, 150, 200, 250, 300, 500, 1000, 2000, 4000, 6000, 8000, 10000, 20000]
#
#     mean_res, std_res, sr_res, comb_params = list(), list(), list(), list()
#
#     for short in tqdm(grid_short):
#         for long in grid_long:
#             if long > short:
#                 parameters_mom = {"short_window": short, "long_window": long, "plot": False}
#
#                 def pipe(file_name):
#                     df = pl.scan_csv(file_name)
#                     return momentum.momentum_strat2(df, parameters=parameters_mom)
#
#                 mapped = map(pipe, list_files_test)
#                 con = pl.concat(mapped, parallel=True)
#
#                 d = con.collect()
#                 mean = d.select('return').mean().item()
#                 std  = d.select('return').std().item()
#                 pseudo_sharpie = mean/std
#                 mean_res.append(mean), std_res.append(std)
#                 sr_res.append(pseudo_sharpie)
#                 comb_params.append([short, long])
#
#     best_sharpie = np.argmax(sr_res)
#
#     print(f"Optimal parameters: \n"
#           f" - Short {comb_params[best_sharpie][0]}\n"
#           f" - Long  {comb_params[best_sharpie][1]}\n"
#           f"Values: \n"
#           f" - Return {mean_res[best_sharpie]*100:.2f}%\n"
#           f" - Volati {std_res[best_sharpie]*100:.2f}%\n"
#           f" - Sharpe {sr_res[best_sharpie]:.2f}\n")