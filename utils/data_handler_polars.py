import glob
import itertools
import os
import random
import sys
import tarfile

import polars as pl
import regex as re
from tqdm.contrib.itertools import product

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
root = os.getcwd()
root_data = os.path.join(root,"data")                                   # /Project/data
root_data_raw_bbo = os.path.join(root_data, "Raw", "bbo")               # /Project/data/Raw/bbo
root_data_raw_trade = os.path.join(root_data, "Raw", "trade")           # /Project/data/Raw/trade
root_data_raw_sp100 = os.path.join(root_data, "Raw", "sp100_2004-8")    # /Project/data/Raw/sp100_2004-8
root_data_raw_sp100_bbo = os.path.join(root_data_raw_sp100, "bbo")      # /Project/data/Raw/sp100_2004-8/bbo
root_data_raw_sp100_trade = os.path.join(root_data_raw_sp100, "trade")  # /Project/data/Raw/sp100_2004-8/trade

root_handler_folder = os.path.join(root_data, "handler")
os.makedirs(root_handler_folder, exist_ok=True)

root_clean_folder = os.path.join(root_data, "clean")
os.makedirs(root_clean_folder, exist_ok=True)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
column_time_name = 'xltime'
verbose = False
circle_symbols = ["| Month "+ "I" * x for x in range(1, 13)]
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def list_files_data(folder: str) -> list[str]:
    """Given a folder, it returns the list of all the tickers that are contained in it.
    A ticker is a series of 1 to 4 capital letters, followed by '.N' or '.O'"""

    file_names_raw = os.listdir(folder)

    rx = re.compile(r'^[A-Z]{1,4}.[NO]$')  # Ensure the folders are: TICKER.N or TICKER.O (MSFT corner case)

    file_names = list(filter(
        lambda x: bool(rx.match(x)) and os.path.isdir(os.path.join(folder, x)),  # and keep only folders
        file_names_raw
    ))

    return file_names


def extract_files(tar_file_path: str, destination_path: str, yyyy, mm) -> None:
    """Extract the content of all the .tar files in trade and bbo folders"""

    with tarfile.open(tar_file_path) as tar:
        members_to_extract = [member for member in tar.getmembers() if (str(yyyy) in member.name and str(mm) in member.name)]
        # print("MEMBERS", members_to_extract)

        tar.extractall(path=destination_path, members=members_to_extract)

        if "MSFT" in tar_file_path:
            print("MSFT ticker cannot be loaded as the .tar file is empty. Process continued without it.")


def filter_year_month(list_paths: list, year: int, month: int, day: int = None):
    """Filter the files to keep only specific date(s)"""

    if not (2003 < year < 2009):
        raise ValueError(f"The year must be correct (expected int, 2004 or more, got {year!r}).")

    if not (0 < month < 13):
        raise ValueError(f"The month must be correct (expected int between 1 and 12, got {month!r}).")

    if day:
        if not (31 >= day > 0):
            raise ValueError(f"The day must be correct (expected int between 1 and 31, got {day!r}).")
        day = f"{day:02}"

    month = f"{month:02}"  # Make sure it is 2-digit

    if day:
        rx = re.compile(rf'^.*/{str(year)}-{month}-{day}-.*\.csv\.gz$')
    else:
        rx = re.compile(rf'^.*/{str(year)}-{month}-\d{{2}}-.*\.csv\.gz$')

    filtered_files = list(filter(
        lambda x: rx.match(x),
        list_paths
    ))

    return filtered_files


def get_one_month_data(ticker: str, year: int, month: int, day: int = None, bbo: bool = True):
    bbo_trade = "bbo" if bbo else "trade"
    folder_root = root_data_raw_sp100_bbo if bbo else root_data_raw_sp100_trade

    folder_handler = os.path.join(root_handler_folder, ticker.upper(), str(bbo_trade))

    tickers = list_files_data(folder_root)

    # tickers = list(map(lambda x: x.split(".")[0], tickers))  # keeps only element before .N (ex: ABT.N -> ABT)
    tickers_and_extension = list(map(lambda x: [x.split(".")[0], x.split(".")[1]], tickers))
    tickers_only = [elem[0] for elem in tickers_and_extension]
    extension_only = [elem[1] for elem in tickers_and_extension]

    if not ticker in tickers_only:  # Make sure the ticker exists
        raise ValueError(f"The ticker {ticker!r} is not available at {folder_root!r}")

    extension_ticker = extension_only[tickers_only.index(ticker)]

    tar_file_path = os.path.join(folder_root, f"{ticker.upper()}.{extension_ticker.upper()}/{ticker.upper()}.{extension_ticker.upper()}_{bbo_trade}.tar")
    # print(tar_file_path)
    # Extract the .tar file and save it in data/handler
    extract_files(tar_file_path=tar_file_path, destination_path=folder_handler, yyyy=year, mm=month)

    items = glob.glob(folder_handler + "/*.gz")

    # Filter the files for the correct date
    filtered_files = filter_year_month(list_paths=items, year=year, month=month, day=day)  # list of file paths

    return filtered_files


def filter_correct_times(dataframe: pl.DataFrame, hhmmss_open: str = "09:30:00",
                         hhmmss_close: str = "16:00:00") -> pl.DataFrame:
    """Given a dataframe and an open/close time, return a filtered dataframe. Very closely inspired by the FIN-525 course material."""
    # Format the date to match
    open_hours, open_min, open_sec = [int(time) for time in hhmmss_open.split(":")]
    close_hours, close_min, close_sec = [int(time) for time in hhmmss_close.split(":")]

    sec_open = open_hours * 3600 + open_min * 60 + open_sec
    sec_close = close_hours * 3600 + close_min * 60 + close_sec

    dataframe = dataframe.filter(
        pl.col('date').dt.hour().cast(pl.Int32) * 3600
        + pl.col('date').dt.minute().cast(pl.Int32) * 60
        + pl.col('date').dt.second().cast(pl.Int32) >= sec_open,

        pl.col('date').dt.hour().cast(pl.Int32) * 3600
        + pl.col('date').dt.minute().cast(pl.Int32) * 60
        + pl.col('date').dt.second().cast(pl.Int32) <= sec_close,
    )

    return dataframe


def open_bbo_files(dataframe: pl.DataFrame, timezone: str = "America/New_York", only_regular_hours: bool = True,
                   merge_sub_trades: bool = True, hhmmss_open: str = "09:30:00", hhmmss_close: str = "16:00:00"):
    """Load dataframes for selected bbo files. Very closely inspired by the FIN-525 course material."""
    base_date = pl.datetime(1899, 12, 30)
    dataframe = (
        dataframe.with_columns(
            (pl.col(column_time_name) * pl.duration(days=1) + base_date)
            .alias('date')
            .dt.convert_time_zone(timezone))
        .drop(column_time_name)
    )

    # Keep only positive prices and ask>bid
    dataframe = dataframe.filter(
        pl.col("ask-price") > 0
    ).filter(
        pl.col("bid-price") > 0
    ).filter(
        pl.col("ask-price") > pl.col("bid-price")
    )

    if merge_sub_trades:  # Merge the lines at the same date
        dataframe = dataframe.group_by(
            'date', maintain_order=True
        ).last()

    if only_regular_hours:
        # Format the date to match
        dataframe = filter_correct_times(dataframe=dataframe, hhmmss_open=hhmmss_open, hhmmss_close=hhmmss_close)

    return dataframe


def open_trade_files(dataframe: pl.DataFrame, timezone: str = "America/New_York", only_regular_hours: bool = True,
                     only_regular_trades: bool = True, merge_sub_trades: bool = True, hhmmss_open: str = "09:30:00",
                     hhmmss_close: str = "16:00:00"):
    """Load dataframes for selected trade files. Very closely inspired by the FIN-525 course material."""
    base_date = pl.datetime(1899, 12, 30)
    dataframe = (
        dataframe.with_columns(
            (pl.col(column_time_name) * pl.duration(days=1) + base_date)
            .alias('date')
            .dt.convert_time_zone(timezone))
        .drop(column_time_name)
    )

    if only_regular_trades:
        dataframe = dataframe.filter(
            pl.col('trade-stringflag') == 'uncategorized'
        )

    if only_regular_hours:
        dataframe = filter_correct_times(dataframe=dataframe, hhmmss_open=hhmmss_open, hhmmss_close=hhmmss_close)

    dataframe = dataframe.drop(["trade-rawflag", "trade-stringflag"])  # No need this anymore

    if merge_sub_trades:
        dataframe = dataframe.group_by(
            'date', maintain_order=True
        ).agg(
            [
                (pl.col('trade-price') * pl.col('trade-volume')).sum() / (pl.col('trade-volume').sum()).alias(
                    'trade-price'),
                pl.sum('trade-volume')
            ]
        )

    return dataframe


def merge_bbo_trade(df_bbo: pl.DataFrame, df_trade: pl.DataFrame) -> pl.DataFrame:
    """Given a bbo and a trade dataframe, returns a merged dataframe, with second time increments."""

    # Merge the bbo and trade dataframes
    merged = df_bbo.join(df_trade, on='date', how='full', coalesce=True).sort('date')

    # Round to the nearest second
    merged = merged.with_columns(
        (pl.col('date').dt.round('1s'))
    )

    # Aggregate the dataframe to keep 1 row per second
    merged = merged.group_by(
        'date', maintain_order=True
    ).agg(
        [
            (pl.col('bid-price').mean()),
            (pl.col('bid-volume').sum()),
            (pl.col('ask-price').mean()),
            (pl.col('ask-volume').sum()),
            (pl.col('trade-price').mean()),
            (pl.col('trade-volume').sum()),
        ]
    )

    # Filling the missing values using interpolation
    merged = merged.with_columns(
        [
            pl.col('bid-price').interpolate(),  # .fill_null(strategy="forward"),
            pl.col('bid-volume').interpolate(),  # .fill_null(strategy="forward"),
            pl.col('ask-price').interpolate(),  # .fill_null(strategy="forward"),
            pl.col('ask-volume').interpolate(),  # .fill_null(strategy="forward"),
            pl.col('trade-price').interpolate(),  # .fill_null(strategy="forward"),
            pl.col('trade-volume').interpolate(),  # .fill_null(strategy="forward"),
        ]
    )
    merged = merged.drop_nulls()

    return merged


def extract_ticker_yyyymmdd(file_names: list) -> (list, list, list, list):
    """Given a list of file names, it returns the unique years, months, days, tickers and code (TICKERYYYYMMDD) it covers."""

    unique = {
        'years': [],
        'months': [],
        'days': [],
        'tickers': [],
        'codes': []
    }

    pattern = r"(\d{4})-(\d{2})-(\d{2})-([A-Z]+)\."

    # Extract information
    for file in file_names:
        match = re.search(pattern, file)
        if match:
            year, month, day, ticker = match.groups()
            code = f"{ticker}{year}{month}{day}"
            unique['years'].append(year), unique['months'].append(month), unique['days'].append(day), unique[
                'tickers'].append(ticker), unique['codes'].append(code)

    unique_years = list(set(unique['years']))
    unique_months = list(set(unique['months']))
    unique_days = list(set(unique['days']))
    unique_tickers = list(set(unique['tickers']))
    unique_codes = list(set(unique['codes']))

    return unique_years, unique_months, unique_days, unique_tickers, unique_codes


def get_random_tickers(nb:int=5):
    """Select randomly some tickers from the avalaible tickers."""
    files_names = list_files_data(folder=root_data_raw_sp100_bbo)
    _, _, _, unique_tickers, _ = extract_ticker_yyyymmdd(files_names)

    print(root_data_raw_sp100_bbo)

    print(os.listdir(root_data_raw_sp100_bbo))

    print(unique_tickers)

    return random.sample(unique_tickers, nb)


def handle_files(ticker: str, year: int|list|str, month: int|str|list, day: int|str|list|bool = None, force_return_list=False):

    if ticker == "MSFT":
        # MSFT contains empty tar file
        print("MSFT ticker cannot be loaded as the .tar file is empty. Process continued without it.")
        return None, None

    if year == "*":
        year = [2004, 2005, 2006, 2007, 2008]
    elif isinstance(year, int):
        year = [year]
    elif (isinstance(year, list) and all(isinstance(item, int) for item in year)):
        year = year
    else:
        raise TypeError(f"The argument 'year' expected to be an int, list of int or '*' (got {year})")

    if month == "*":
        month = [i for i in range(1, 13)]
    elif isinstance(month, int):
        month = [month]
    elif (isinstance(month, list) and all(isinstance(item, int) for item in month)):
        month = month
    else:
        raise TypeError(f"The argument 'month' expected to be an int, list of int or '*' (got {month})")

    if day == "*":
        day = [i for i in range(1, 32)]
    elif isinstance(day, int):
        day = [day]
    elif (isinstance(day, list) and all(isinstance(item, int) for item in day)):
        day = day
    elif not day:
        day = [None]
    else:
        raise TypeError(f"The argument 'day' expected to be an None, int, list of int or '*' (got {day})")

    year.sort()
    month.sort()
    day.sort()

    # Check if the files were already loaded:
    # Check if the folder exists
    path_target_bbo = os.path.join(root_handler_folder, ticker.upper(), 'bbo')
    path_target_trade = path_target_bbo.replace('bbo', 'trade')

    # if os.path.exists(path_target_bbo):
    #     list_existing_files_bbo = os.listdir(path_target_bbo)
    #     list_existing_files_trade = os.listdir(path_target_trade)
    #     str_year = [str(y) for y in year]
    #     str_month = [str(m) for m in month]
    #
    #     kept_only_bbo = [item for item in list_existing_files_bbo if
    #                      (item.split("-")[0] in str_year and item.split("-")[1] in str_month)]
    #     kept_only_trade = [item for item in list_existing_files_trade if
    #                      (item.split("-")[0] in str_year and item.split("-")[1] in str_month)]
    #
    #
    #     kept_only_bbo = [os.path.join(root_handler_folder, 'bbo', ticker, x) for x in kept_only_bbo]
    #     kept_only_trade = [os.path.join(root_handler_folder, 'trade', ticker, x) for x in kept_only_trade]
    #
    #     kept_only_bbo.sort(), kept_only_trade.sort()
    #
    #     if not force_return_list:
    #         return None, None
    #     else:
    #         return kept_only_bbo, kept_only_trade


    files_bbo = []
    files_trade = []

    for yyyy, mm, dd in product(year, month, day, desc=f"Extracting the files for {ticker}", dynamic_ncols=True):
        files_bbo.append(get_one_month_data(ticker=ticker, year=yyyy, month=mm, day=dd, bbo=True))
        files_trade.append(get_one_month_data(ticker=ticker, year=yyyy, month=mm, day=dd, bbo=False))

    files_bbo = list(itertools.chain(*files_bbo))
    files_trade = list(itertools.chain(*files_trade))

    return files_bbo, files_trade


def full_pipeline_merge(file_source_bbo) -> pl.DataFrame|None:
    """Given the file name of the bbo and the trade file, returns a merged dataframe."""
    file_source_trade = file_source_bbo.replace("bbo", "trade")

    bbo_dtypes = {
        "bid-price": pl.Float64,
        "bid-volume": pl.Int64,
        "ask-price": pl.Float64,
        "ask-volume": pl.Int64,
    }

    trade_dtypes = {
        "trade-price": pl.Float64,
        "trade-volume": pl.Int64,  # Adjust here if necessary
    }


    try:
        # df_bbo = pl.read_csv(file_source_bbo, dtypes=bbo_dtypes, ignore_errors=True)
        # df_trade = pl.read_csv(file_source_trade, dtypes=trade_dtypes, ignore_errors=True)
        df_bbo = pl.scan_csv(file_source_bbo, dtypes=bbo_dtypes, ignore_errors=True)
        df_trade = pl.scan_csv(file_source_trade, dtypes=trade_dtypes, ignore_errors=True)
        df_bbo_clean = open_bbo_files(df_bbo)
        df_trade_clean = open_trade_files(df_trade)
        df_merged = merge_bbo_trade(df_bbo=df_bbo_clean, df_trade=df_trade_clean)
        df_merged = df_merged.unique()

        return df_merged

    except Exception as e:
        if verbose:
            print(f"File {file_source_bbo} could not be loaded: {e}")
        return None


def read_data(files_bbo, files_trade, ticker: str, disable=True):

    if not files_bbo and not files_trade:
        return None

    files_bbo.sort(), files_trade.sort()

    yyyy_bbo, mm_bbo, dd_bbo, tickers_bbo, codes_bbo =  extract_ticker_yyyymmdd(files_bbo)
    yyyy_trade, mm_trade, dd_trade, tickers_trade, codes_trade = extract_ticker_yyyymmdd(files_trade)

    # Reduce to the files that exists in both bbo and trade
    union = list(filter(
        lambda x: x in codes_bbo,
        codes_trade
    ))  # ['ABT20040630', 'ABT20040603']
    # To make sure we have complete pairs of bbo/trade files
    union = [[elem[-2:], elem[-4:-2], elem[-8:-4], elem[:-8]][::-1] for elem in
             union]  # [['ABT', '2004', '06', '30'], ['ABT', '2004', '06', '03']]

    union_alt =  list(map(list, zip(*union)))
    union_tickers, union_year, union_month, union_day = list(set(union_alt[0])), list(set(union_alt[1])), list(set(union_alt[2])), list(set(union_alt[3]))
    union_month.sort()

    # for year in tqdm.tqdm(union_year, total=len(union_year), desc=f"Concatenation of the files", disable=disable):
    for year in union_year:

        # Filter on the months that are not already in the clean file
        destination_path = os.path.join(root_clean_folder, ticker, year)
        os.makedirs(destination_path, exist_ok=True)
        # Check the content:
        list_f = os.listdir(destination_path)
        if len(list_f) > 0:
            m = [elem.split("_")[0] for elem in list_f]
            m.sort()
            month_iter = [item for item in union_month if not item in m] # keep only the months that are missing
        else:
            month_iter = union_month
        symbol_index = 0

        if len(month_iter) > 0:
            for month in month_iter:
                # Keeps track of the progress
                sys.stdout.write(f'\r\r{ticker} | Loading {year} {circle_symbols[symbol_index]}')
                sys.stdout.flush()
                symbol_index = (symbol_index + 1) % len(circle_symbols)  # Cycle through symbols

                file_name_union_bbo = [
                    os.path.join(root_handler_folder, code[0], "bbo", f"{year}-{month}-{code[3]}-{code[0]}.N-bbo.csv.gz")
                    for code in union
                ]
                # destination_path = os.path.join(root_clean_folder, ticker, year)
                # os.makedirs(destination_path, exist_ok=True)

                file_name_union_bbo = [x for x in file_name_union_bbo if os.path.exists(x)]
                mapped = map(full_pipeline_merge, file_name_union_bbo)
                filtered = [result for result in mapped if result is not None]

                if len(filtered) > 0:
                    concatenated_df = pl.concat(filtered, parallel=True)
                    concatenated_df = concatenated_df.unique()
                    concatenated_df = concatenated_df.sort(pl.col('date'))
                    concatenated_df.collect().write_csv(destination_path+f"/{month}_bbo_trade.csv")

    sys.stdout.write(f'\r{ticker} | {year} complete         \n')