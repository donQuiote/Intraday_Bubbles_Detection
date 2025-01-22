import utils.data_handler_polars

TICKER = "ABT"
YEARS = [2004]
MONTHS = "*"

# TODO: HANDLE MSFT
for ticker in ["MMM", "COP", "S", "TWX", "UPS"]:
    files_bbo, files_trade = utils.data_handler_polars.handle_files(ticker=ticker, year=YEARS, month=MONTHS)
    concatenated_df = utils.data_handler_polars.read_data(files_bbo=files_bbo, files_trade=files_trade, ticker=ticker)