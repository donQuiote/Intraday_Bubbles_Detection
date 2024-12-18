import os
import re
import tarfile

import numpy as np
import pandas as pd
import xlrd
from tqdm import tqdm

root = "/Users/gustavebesacier/Library/Mobile Documents/com~apple~CloudDocs/Documents/HEC/EPFL MA III/Financial big data/Fin_Big_Data-FIN-525-/Data/Raw/SP500_2010"
bbo = os.path.join(root, 'bbo')
trade = os.path.join(root, 'trade')

saving_root = "/Users/gustavebesacier/Library/Mobile Documents/com~apple~CloudDocs/Documents/HEC/EPFL MA III/Financial big data/Project/data"

BBO_FILE = os.path.join(bbo, 'AA_05.tar')
TRA_FILE = os.path.join(trade, 'AA_05.tar')

def load_data(path:str, date_column:str="xltime", timezone:str="America/New_York"):

    data = pd.read_parquet(path)

    # Handle the date
    data[date_column] = data[date_column].apply(lambda x: xlrd.xldate_as_datetime(x, 0))

    #  Add timezone
    data[date_column] = data[date_column].dt.tz_localize(timezone)

    return data

def refactor_date_df(data:pd.DataFrame, date_column:str= "xltime", timezone:str= "America/New_York"):
    # Handle the date
    data[date_column] = data[date_column].apply(lambda x: xlrd.xldate_as_datetime(x, 0))

    #  Add timezone
    data[date_column] = data[date_column].dt.tz_localize(timezone)

    return data

def clean_up_data(data:pd.DataFrame, by_col:list=None, agg_dict:dict=None):

    if not by_col:
        by_col = ["xltime"]  # ["xltime", "trade-rawflag"]
    if not agg_dict:
        agg_dict = {
        "trade-price": "mean",
        "trade-volume": "sum"
    }
    data_grouped = data.groupby(by_col).agg(agg_dict)

    return data_grouped

def check_folder_exists(path:str):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder {path!r} did not existed, and has been created.")

def prepare_data_bbo_trade_date(trade_path:str, bbo_path:str, save_file:str, col_time:str='xltime', agg_dict_trade:dict=None, agg_dict_bbo:dict=None):

    try:
        tradeSPX = load_data(trade_path)
        bboSPX   = load_data(bbo_path)
    except:
        print(f"Couldn't load data on file(s) {trade_path!r} and/or {bbo_path!r}.")
        return None

    if not agg_dict_trade:
        agg_dict_trade = {"trade-price": "mean", "trade-volume": "sum"}

    if not agg_dict_bbo:
        agg_dict_bbo = {"bid-price": "mean", "bid-volume": "sum", "ask-price": "mean", "ask-volume": "sum"}


    tradeSPX = clean_up_data(tradeSPX, by_col=[col_time], agg_dict=agg_dict_trade)
    bboSPX = clean_up_data(bboSPX, by_col=[col_time], agg_dict=agg_dict_bbo)

    data_combined = tradeSPX.merge(bboSPX, on=col_time)

    check_folder_exists("Data/Clean/intraday")

    data_combined.to_csv("Data/Clean/intraday/dataSPX.csv")

    return data_combined

def prepare_data_bbo_trade_date_already_dataframe(trade_df:pd.DataFrame, bbo_df:pd.DataFrame, save_file:str, file_name:str, col_time:str='xltime', agg_dict_trade:dict=None, agg_dict_bbo:dict=None):

    trade_df, bbo_df = refactor_date_df(trade_df), refactor_date_df(bbo_df)

    if not agg_dict_trade:
        agg_dict_trade = {"trade-price": "mean", "trade-volume": "sum"}

    if not agg_dict_bbo:
        agg_dict_bbo = {"bid-price": "mean", "bid-volume": "sum", "ask-price": "mean", "ask-volume": "sum"}


    tradeSPX = clean_up_data(trade_df, by_col=[col_time], agg_dict=agg_dict_trade)
    bboSPX = clean_up_data(bbo_df, by_col=[col_time], agg_dict=agg_dict_bbo)

    data_combined = tradeSPX.merge(bboSPX, on=col_time)

    if save_file:
        check_folder_exists("Data/Clean/intraday")
        data_combined.to_csv(os.path.join(saving_root, f"Clean/intraday/{file_name}.csv"))

    return data_combined

def get_tickers(folder:str) -> dict:
    """Given a folder, returns a dictionary {month: [list_tickers_month]}."""
    master = dict()
    for file in os.listdir(folder):
        reg = re.match(r"(?P<tick>[a-zA-Z]+)_(?P<month>[0-9]+)\.tar", file)
        if reg:
            tick = reg.group('tick')
            month= reg.group('month')
            if month in master:
                master[month].append(tick)
            else:
                master[month] = [tick]
    return master

def select_random_ticekrs(n=20, folder=bbo) -> list:
    """Randomly selects a list of n tickers from all the tickers in a folder."""
    dict_tickers = get_tickers(folder)
    month = list(dict_tickers.keys())[0]
    random_selection = np.random.choice(dict_tickers[month], size=n, replace=False)

    return random_selection


def extract_specific_files(tar_file_path, extract_to, file_list):
    with tarfile.open(tar_file_path, 'r') as tar:
        for file_name in file_list:
            try:
                tar.extract(file_name, extract_to)
            except KeyError:
                print(f"Warning: File '{file_name}' not found in the tar archive.")

def extract_date_tar_filename(filename:str="data/extraction/TRTH/raw/equities/US/bbo/AA/2010-05-03-AA-bbo.parquet"):
    year, month, day, ticker = filename.split("/")[-1].split(".")[0].split("-")[:-1]
    return year, month, day, ticker

def only_day(filename):
    _, _, day, _ = extract_date_tar_filename(filename)
    return day

def only_ticker(filename):
    _, _, _, ticker = extract_date_tar_filename(filename)
    return ticker

def extract_combine_data_tar(tar_file_bbo=BBO_FILE, tar_file_trade=TRA_FILE):

    with tarfile.open(tar_file_bbo, 'r') as file_bbo:

        with tarfile.open(tar_file_trade, 'r') as file_trade:
            bbo_files = [bbo.name for bbo in file_bbo]
            tra_files = [trade.name for trade in file_trade]

            for bbo, trade in tqdm(zip(bbo_files, tra_files), total = len(bbo_files), disable=True):
                bbo_year, bbo_month, bbo_day, bbo_ticker = extract_date_tar_filename(bbo)
                trade_year, trade_month, trade_day, trade_ticker = extract_date_tar_filename(trade)
                if (bbo_year == trade_year) and (bbo_month == trade_month) and (bbo_day == trade_day) and (bbo_ticker == trade_ticker):
                    bbo_object = file_bbo.extractfile(bbo)
                    trade_object = file_trade.extractfile(trade)
                    bbo_df = pd.read_parquet(bbo_object)
                    trade_df = pd.read_parquet(trade_object)

                    _ = prepare_data_bbo_trade_date_already_dataframe(
                        trade_df=trade_df,
                        bbo_df=bbo_df,
                        save_file=True,
                        file_name=f"{trade_ticker}_{trade_year}_{trade_month}_{trade_day}")
                else:
                    print(f"No match between dates: bbo: {bbo_year, bbo_month, bbo_day, bbo_ticker}, trade {trade_year, trade_month, trade_day, trade_ticker}")

def concat_days_to_months():
    pass

if __name__ == "__main__":

    selection_tickers = select_random_ticekrs()

    for titicker in tqdm(selection_tickers):
        ticker_bbo_filename = os.path.join(bbo, f'{titicker}_05.tar')
        ticker_tra_filename = os.path.join(trade, f'{titicker}_05.tar')

        extract_combine_data_tar(tar_file_bbo=ticker_bbo_filename, tar_file_trade=ticker_tra_filename)


    # extract_combine_data_tar()
    individual_data = os.path.join(saving_root, "Clean/intraday")
    tickers = list()

    list_files = os.scandir(individual_data)
    all_tickers = [f.name.split("_")[0] for f in list_files if not f.name.startswith(".")]
    tickers = list(set(all_tickers))
    for tick in tickers:
        all_files = [file for file in os.listdir(individual_data) if file.endswith(".csv") and file.startswith(tick)]
        all_files.sort()
        print(all_files)
        concatenated_df = pd.DataFrame()
        for csv_file in all_files:
            file_path = os.path.join(individual_data, csv_file)
            temp_df = pd.read_csv(file_path)  # Read each CSV file
            concatenated_df = pd.concat([concatenated_df, temp_df], ignore_index=True)

        year, month = all_files[0].split('_')[1], all_files[0].split('_')[2]
        print(year, month)
        # Save the full dataset

        file_dir = os.path.join(saving_root, "concat")
        check_folder_exists(file_dir)
        concatenated_df.to_csv(os.path.join(file_dir, f"{tick}_{year}_{month}.csv"))


    # for file in os.path.join(saving_root)

    # print(select_random_ticekrs())



    #
    # with tarfile.open(BBO_FILE, 'r') as file_bbo:
    #
    #     with tarfile.open(TRA_FILE, 'r') as file_trade:
    #         bbo_files = [bbo.name for bbo in file_bbo]
    #         tra_files = [trade.name for trade in file_trade]
    #         bbo_days = [only_day(file) for file in bbo_files]
    #         tra_days = [only_day(file) for file in tra_files]
    #
    #         for bbo, trade in zip(bbo_files, tra_files):
    #             bbo_year, bbo_month, bbo_day, bbo_ticker = extract_date_tar_filename(bbo)
    #             trade_year, trade_month, trade_day, trade_ticker = extract_date_tar_filename(trade)
    #             if (bbo_year == trade_year) and (bbo_month == trade_month) and (bbo_day == trade_day) and (bbo_ticker == trade_ticker):
    #                 bbo_object = file_bbo.extractfile(bbo)
    #                 trade_object = file_trade.extractfile(trade)
    #                 bbo_df = pd.read_parquet(bbo_object)
    #                 trade_df = pd.read_parquet(trade_object)
    #
    #                 dataset = prepare_data_bbo_trade_date_already_dataframe(
    #                     trade_df=trade_df,
    #                     bbo_df=bbo_df,
    #                     save_file=True,
    #                     file_name=f"{trade_ticker}_{trade_year}_{trade_month}_{trade_day}")
    #             else:
    #                 print("No match between dates.")



            # for idx, day in enumerate(bbo_days):
            #     if not day in tra_days:
            #         print(f"Couldn't load the data for day {day}.")
            #     bbo_file_name_day = bbo_files[idx]
            #     year_from_file, month_from_file, day_from_file, ticker_from_file = extract_date_tar_filename(bbo_file_name_day)
            #     if day_from_file == day:
            #         # trade_file = os.path.join(trade, f"data/extraction/TRTH/raw/equities/US/trade/{ticker_from_file}_{month_from_file}.tar/data/extraction/TRTH/raw/equities/US/trade/{ticker_from_file}/{year_from_file}-{month_from_file}-{day_from_file}-{ticker_from_file}-trade.parquet")
            #         # bbo_file = os.path.join(bbo, f"data/extraction/TRTH/raw/equities/US/trade/{ticker_from_file}_{month_from_file}.tar/data/extraction/TRTH/raw/equities/US/trade/{ticker_from_file}/{year_from_file}-{month_from_file}-{day_from_file}-{ticker_from_file}-bbo.parquet")
            #         trade_file = file_trade.extractfile(f"data/extraction/TRTH/raw/equities/US/trade/{ticker_from_file}/{year_from_file}-{month_from_file}-{day_from_file}-{ticker_from_file}-trade.parquet")
            #         bbo_file = file_trade.extractfile(f"data/extraction/TRTH/raw/equities/US/trade/{ticker_from_file}/{year_from_file}-{month_from_file}-{day_from_file}-{ticker_from_file}-bbo.parquet")
            #
            #         # Create the file:
            #         trade_df = pd.read_parquet(trade_file)
            #         bbo_df = pd.read_parquet(bbo_file)
            #
            #         dataset = prepare_data_bbo_trade_date_already_dataframe(trade_df=trade_df, bbo_df=bbo_df, save_file=True, name=f"{ticker_from_file}_{year_from_file}_{month_from_file}_{day_from_file}")
            #         print(dataset)
            #     else:
            #         print("C'est la d")