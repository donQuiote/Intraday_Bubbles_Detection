import utils.data_handler_polars


# filtered_files = utils.data_handler_polars.get_one_month_data(ticker="ABT", year=2007, month=6)


# files_bbo, files_trade = utils.data_handler_polars.handle_files(ticker="ABT", year=[2004, 2005], month=[4, 5])
files_bbo, files_trade = utils.data_handler_polars.handle_files(ticker="ABT", year=[2004, 2005], month=[4, 5])
print(len(files_bbo), len(files_trade))
print(files_bbo, files_trade)

# concatenated_df = utils.data_handler_polars.read_data(files_bbo=files_bbo, files_trade=files_trade, ticker="ABT", year=2004, month=6)

# print(concatenated_df.collect())