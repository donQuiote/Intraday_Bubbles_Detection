import utils.data_handler_polars


YEARS = [2004]
MONTHS = "*"

for ticker in ["MSFT", "PEP", "APA"]:
    files_bbo, files_trade = utils.data_handler_polars.handle_files(ticker=ticker, year=YEARS, month=MONTHS)
    concatenated_df = utils.data_handler_polars.read_data(files_bbo=files_bbo, files_trade=files_trade, ticker=ticker)