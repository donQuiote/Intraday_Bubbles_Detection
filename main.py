import utils.data_handler_polars

TICKER = "ABT"
YEARS = [2004, 2005]
MONTHS = [1, 2, 3, 4] # [4]
# filtered_files = utils.data_handler_polars.get_one_month_data(ticker="ABT", year=2007, month=6)


# files_bbo, files_trade = utils.data_handler_polars.handle_files(ticker="ABT", year=[2004, 2005], month=[4, 5])
files_bbo, files_trade = utils.data_handler_polars.handle_files(ticker=TICKER, year=YEARS, month=MONTHS)

concatenated_df = utils.data_handler_polars.read_data(files_bbo=files_bbo, files_trade=files_trade, ticker=TICKER)


# print(concatenated_df.collect())

# print(concatenated_df.collect())