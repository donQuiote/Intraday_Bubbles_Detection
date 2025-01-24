import os
import re
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

import utils.data_handler_polars
from utils.data_handler_polars import root_handler_folder

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cwd = os.getcwd()
root_data = os.path.join(cwd, 'data', 'Raw','sp100_2004-8')
root_data_bbo = os.path.join(root_data, 'bbo')
root_data_trade = root_data_bbo.replace('bbo', 'trade')
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def file_names_from_ticker(ticker, path):
    """Given a ticker and a path (path to the bbo or the trade directory), returns the list of the files in that folder for that ticker."""
    ticker_path = os.path.join(path, ticker)
    tar_file = os.listdir(ticker_path)
    tar_file_path = os.path.join(ticker_path, tar_file[0])

    with tarfile.open(tar_file_path, 'r') as tar:
        file_names = tar.getnames()

    return file_names

def extract(ticker, path_bbo) -> list:
    """Given a ticker and a path, it returns a list of all the dates for which we have data, of the form of a code 'YYYYMMDD' """

    file_names = file_names_from_ticker(ticker=ticker, path=path_bbo)
    pattern = r"(\d{4})-(\d{2})-(\d{2})-([A-Z]+)\."

    code_list = list()
    # Extract information
    for file in file_names:
        match = re.search(pattern, file)
        if match:
            year, month, day, ticker = match.groups()
            code = f"{year}{month}{day}"
            code_list.append(code)

    return code_list
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_pdf_stats(df,x_col_name):
    mean_val = df.select(pl.col(x_col_name).mean()).to_numpy()[0][0]
    median_val = df.select(pl.col(x_col_name).median()).to_numpy()[0][0]
    std_val = df.select(pl.col(x_col_name).std()).to_numpy()[0][0]

    # Print stats
    print(f"Mean: {mean_val}")
    print(f"Median: {median_val}")
    print(f"Standard Deviation: {std_val}")

    # Plot the Probability Density Function (PDF)
    plt.figure(figsize=(8, 6))
    plt.hist(df[x_col_name].to_numpy(), bins=1000, density=True, alpha=0.6, color='g')

    # Plotting the PDF using a Kernel Density Estimate (KDE)
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(df[x_col_name].to_numpy())
    x = np.linspace(min(df[x_col_name].to_numpy()), max(df[x_col_name].to_numpy()), 1000)
    plt.plot(x, kde(x), label="PDF (KDE)", color='red')

    plt.title("PDF of calc_value")
    plt.xlabel("calc_value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def plot_trades(stock_price, time, positions):
    plt.figure(figsize=(10, 6))
    plt.plot(time, stock_price, label="Stock Price", color="blue")

    for i in range(1, len(positions)):
        if positions[i - 1] == 1:
            plt.axvspan(i - 1, i, color='green', alpha=0.3)

def plot_tickers_dates(bbo=True):
    """Plots the presence of data for each ticker for the bbo or trade files."""
    root_data_path = root_data_bbo if bbo else root_data_trade

    file_names_raw = os.listdir(root_data_path)
    rx = re.compile(r'^[A-Z]{1,4}.[NO]$')  # Ensure the folders are: TICKER.N or TICKER.O (MSFT corner case)
    ticker_names = list(filter(
        lambda x: bool(rx.match(x)) and os.path.isdir(os.path.join(root_data_path, x)),  # and keep only folders
        file_names_raw
    ))

    dict_data = dict()
    for ticker in tqdm(ticker_names):
        dict_data[ticker] = extract(ticker, root_data_path)

    # +++++ Plot part
    # Extract all unique dates and sort them
    all_dates = sorted(set(date for dates in dict_data.values() for date in dates))

    # Create a binary matrix to represent the presence of dates for each ticker

    tickers = list(dict_data.keys())  # random.sample(list(dict_data_bbo.keys()), 20)
    matrix = np.zeros((len(tickers), len(all_dates)))

    # Populate the matrix with 1s and 0s
    for i, ticker in enumerate(tickers):
        for j, date in enumerate(all_dates):
            if date in dict_data[ticker]:
                matrix[i, j] = 1

    fig, ax = plt.subplots(figsize=(20, 15))
    cax = ax.matshow(matrix, cmap="binary", aspect='auto')

    # Set labels
    step = 50
    ax.set_xticks(np.arange(0, len(all_dates), step=step))  # Display every 10th date
    ax.set_xticklabels(all_dates[::step], rotation=90)  # Show corresponding date labels
    ax.set_yticks(np.arange(len(tickers)))
    ax.set_yticklabels(tickers)

    # Label the axes
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Ticker')
    os.makedirs("Graphs", exist_ok=True)
    plt.savefig(f"Graphs/Data_presence_{'bbo' if bbo else 'trade'}.pdf", dpi=1000)

    plt.tight_layout()
    plt.show()

def plot_daily_average_volume_single_stock(average_vol:pl.LazyFrame, ticker:str):

    df = average_vol.collect()
    df = df.with_columns(pl.col('date').cast(pl.Utf8))

    plt.figure()

    # Plot the data
    plt.plot(df['date'], df['trade-volume'])

    # Select a subset of the time stamps for the x-axis
    num_ticks = 6  # Adjust this number to control how many ticks are displayed
    x_ticks = np.linspace(0, len(df['date']) - 1, num_ticks, dtype=int)  # Indices for the ticks
    x_labels = df['date'][x_ticks]  # Corresponding time stamps

    plt.xticks(x_ticks, x_labels, rotation=45)  # Set the ticks and rotate for better readability

    plt.title(f"Average daily traded volume ({ticker})")
    plt.show()

def daily_average_volume(ticker):
    _, files_trade = utils.data_handler_polars.handle_files(ticker=ticker, year="*", month="*", force_return_list=True)

    files_trade = [os.path.join(root_handler_folder, ticker, "trade", x) for x in files_trade]

    mapped = map(trade_file_pipeline, files_trade)
    concat = pl.concat(mapped, parallel=True)

    concat = concat.sort('date')

    average = concat.group_by('date').agg(
        pl.col('trade-volume').mean()
    )

    return average

def trade_file_pipeline(path_file):

    df = utils.data_handler_polars.open_trade_files(dataframe=pl.scan_csv(path_file))

    df = df.group_by_dynamic("date", every="1m").agg(
        pl.col('trade-volume').sum()
    )
    df = df.with_columns(
        pl.col('date').dt.time().alias('date')
    )

    return df