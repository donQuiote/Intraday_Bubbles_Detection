import os
import re
import tarfile

from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
import glob

import utils.data_handler_polars
from utils.data_handler_polars import root_handler_folder

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cwd = os.getcwd()
root_data = os.path.join(cwd, 'data', 'Raw','sp100_2004-8')
root_data_bbo = os.path.join(root_data, 'bbo')
root_data_trade = root_data_bbo.replace('bbo', 'trade')
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
size_graphs_eda = (18, 5)
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

    fig, ax = plt.subplots(figsize=(25, 15))
    cax = ax.matshow(matrix, cmap="binary", aspect='auto')

    # Set labels
    step = 50
    ax.set_xticks(np.arange(0, len(all_dates), step=step))  # Display every 10th date
    ax.set_xticklabels(all_dates[::step], rotation=90)  # Show corresponding date labels
    ax.set_yticks(np.arange(len(tickers)))
    ax.set_yticklabels(tickers)

    # Label the axes
    ax.set_xlabel('Date')#,font_size=18)
    ax.set_ylabel('Ticker')

    plt.tight_layout()
    plt.grid(False)
    os.makedirs("Graphs", exist_ok=True)
    plt.savefig(f"Graphs/Data_presence_{'bbo' if bbo else 'trade'}.pdf", dpi=1000)

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


# Base data path
BASE_PATH = "./data/clean/"

def get_all_tickers():
    """Retrieves all available tickers from the directory structure."""
    return [t for t in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, t))]

def plot_tracker_best_strat(file_path, dict_trad=None):

    data = pd.read_csv(file_path)

    # Transform the data for heatmap plotting
    data_melted = data.melt(id_vars="day", var_name="Ticker", value_name="Value")

    # Pivot the data to create a matrix for the heatmap
    heatmap_data = data_melted.pivot(index="Ticker", columns="day", values="Value")
    heatmap_data = heatmap_data.T

    std_devs = heatmap_data.std()
    sorted_columns = std_devs.sort_values(ascending=True).index
    heatmap_data = heatmap_data[sorted_columns].T
    # Set the figure size
    plt.figure(figsize=(20, 10))

    sns.heatmap(heatmap_data, cmap="viridis", cbar_kws={'label': 'Value'}, annot=False, cbar=True,)

    # Customize the plot
    plt.title("Heatmap of Ticker Values Over Time", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Ticker", fontsize=12)
    # plt.figtext(0.5, -0.05, 'Caption: This plot shows the values of different strategies for each ticker over time.',
    #             ha='center', fontsize=12, color='black')

    # Show the plot
    plt.tight_layout()


    os.makedirs("Graphs", exist_ok=True)
    plt.savefig(f"Graphs/Ticker_strat_overtime.pdf", dpi=1000)
    plt.grid(False)
    plt.show()

def plot_tracker_best_strat_families(file_path, dict_trad=None):

    print(dict_trad)

    # Extract the family names from the dictionary
    family_mapping = {k: v.split('__')[0] for k, v in dict_trad.items()}

    # Create a mapping of families to unique integers
    unique_families = {family: idx for idx, family in enumerate(set(family_mapping.values()))}

    # Create the final mapping from original values to new categories
    value_to_category = {k: unique_families[family] for k, family in family_mapping.items()}
    print(value_to_category)
    data = pd.read_csv(file_path)
    data = data.replace(value_to_category)
    # print(data)

    # Transform the data for heatmap plotting
    data_melted = data.melt(id_vars="day", var_name="Ticker", value_name="Value")

    # Pivot the data to create a matrix for the heatmap
    heatmap_data = data_melted.pivot(index="Ticker", columns="day", values="Value")
    heatmap_data = heatmap_data.T

    std_devs = heatmap_data.std()
    sorted_columns = std_devs.sort_values(ascending=True).index
    heatmap_data = heatmap_data[sorted_columns].T
    # Set the figure size
    plt.figure(figsize=(20, 10))

    sns.heatmap(heatmap_data, cmap="viridis", cbar_kws={'label': 'Value'}, annot=False, cbar=True,)

    # Customize the plot
    plt.title("Heatmap of Ticker Values Over Time", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Ticker", fontsize=12)
    # plt.figtext(0.5, -0.05, 'Caption: This plot shows the values of different strategies for each ticker over time.',
    #             ha='center', fontsize=12, color='black')

    # Show the plot
    plt.tight_layout()
    plt.grid(False)

    os.makedirs("Graphs", exist_ok=True)
    plt.savefig(f"Graphs/Ticker_strat_overtime_families.pdf", dpi=1000)
    plt.show()


def compute_5min_traded_volume_distribution(ticker: str, use_median: bool = False) -> pl.DataFrame:
    """
    Computes the distribution of traded volume per 5-minute interval across all available days.

    Args:
        ticker (str): Stock ticker symbol.
        use_median (bool): If True, computes the median instead of the mean.

    Returns:
        pl.DataFrame: DataFrame with trade volume statistics per 5-minute interval.
    """

    path_to_clean_data = os.path.join(BASE_PATH, ticker)

    if not os.path.exists(path_to_clean_data):
        raise ValueError(f"No data found for {ticker} in {BASE_PATH}")

    # Find all available years (folders)
    years = [y for y in os.listdir(path_to_clean_data) if os.path.isdir(os.path.join(path_to_clean_data, y))]

    lazy_frames = []

    for year in years:
        path_to_clean_data_year = os.path.join(path_to_clean_data, year)
        csv_files = glob.glob(os.path.join(path_to_clean_data_year, "*.csv"))

        for csv_file in csv_files:
            # print(f"Processing: {csv_file}")

            # Use scan_csv() for lazy loading
            df = pl.scan_csv(csv_file, try_parse_dates=True)

            # Ensure 'date' is properly formatted
            df = df.with_columns(pl.col("date").cast(pl.Datetime))

            # Convert to market timezone (EST)
            df = df.with_columns(
                pl.col("date").dt.convert_time_zone("America/New_York")
            )

            # Round timestamp to 5-minute bins
            df = df.with_columns(
                (pl.col("date").dt.truncate("5m")).alias("5min_bar")
            )

            # Extract only time (HH:MM) for grouping by intraday periods
            df = df.with_columns(
                pl.col("5min_bar").dt.strftime("%H:%M").alias("intraday_time")
            )

            # Filter out NULL or zero trade-volume before computing the median
            df = df.filter(pl.col("trade-volume") > 0)

            # Aggregate trade volume per 5-minute interval
            agg_func = pl.median if use_median else pl.mean
            df = df.group_by("intraday_time").agg(
                agg_func("trade-volume").alias("traded_volume")
            )

            # Append LazyFrame to list
            lazy_frames.append(df)

    if not lazy_frames:
        raise ValueError(f"No CSV data found for {ticker}.")

    # Concatenate all LazyFrames
    aggregated_df = pl.concat(lazy_frames)

    # Compute the final statistic across all days
    final_summary = aggregated_df.group_by("intraday_time").agg(
        pl.median("traded_volume").alias("median_traded_volume") if use_median else
        pl.mean("traded_volume").alias("mean_traded_volume")
    ).collect()

    # ---- Ensure All Market Hours Are Included ----
    market_hours = [f"{h:02d}:{m:02d}" for h in range(9, 16) for m in range(0, 60, 5)][1:]  # 9:30 to 16:00

    # Filter only valid trading hours
    final_summary = final_summary.filter(pl.col("intraday_time").is_in(market_hours))

    # ---- Ensure Correct Sorting ----
    final_summary = final_summary.sort("intraday_time")

    return final_summary

def plot_mean_vs_median_traded_volume(ticker: str):
    """
    Plots mean and median traded volume per 5-minute interval in the same figure.

    Args:
        ticker (str): Stock ticker symbol.
    """

    # Compute mean and median traded volume
    mean_df = compute_5min_traded_volume_distribution(ticker, use_median=False)
    median_df = compute_5min_traded_volume_distribution(ticker, use_median=True)

    # ---- Ensure Both DataFrames Are on the Same Time Axis ----
    merged_df = mean_df.join(median_df, on="intraday_time", how="inner")

    # ---- Compute Y-Axis Limit ----
    max_value = max(merged_df["mean_traded_volume"].max(), merged_df["median_traded_volume"].max())

    # ---- Plot the results ----
    plt.figure(figsize=size_graphs_eda)

    # Plot mean
    plt.plot(merged_df["intraday_time"], merged_df["mean_traded_volume"], marker="o", linestyle="-",
             label="Mean Volume", color="blue")

    # Plot median
    plt.plot(merged_df["intraday_time"], merged_df["median_traded_volume"], marker="o", linestyle="--",
             label="Median Volume", color="red")

    # Format the x-axis to show only every hour (9:30, 10:30, ..., 16:00)
    hourly_labels = [t for t in merged_df["intraday_time"] if t.endswith(":30") or t == "16:00"]
    plt.xticks(hourly_labels, rotation=45)

    plt.xlabel("Time (HH:MM)")
    plt.ylabel("Traded Volume")

    plt.title(f"Mean vs. Median Traded Volume per 5-Min Interval - {ticker}")


    plt.legend()
    plt.ylim(0, max_value * 1.05)  # Add a 5% margin on top
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("Graphs", exist_ok=True)
    plt.savefig(f'Graphs/{ticker}volume_intraday.png', dpi=1000)
    plt.show()

def compute_intraday_spread(ticker: str) -> pl.DataFrame:
    """
    Computes average and median bid-ask spread per 5-minute interval, ensuring valid market hours.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        pl.DataFrame: Intraday spread statistics.
    """
    path_to_clean_data = os.path.join(BASE_PATH, ticker)

    if not os.path.exists(path_to_clean_data):
        raise ValueError(f"No data found for {ticker} in {BASE_PATH}")

    years = [y for y in os.listdir(path_to_clean_data) if os.path.isdir(os.path.join(path_to_clean_data, y))]
    lazy_frames = []

    for year in years:
        path_to_clean_data_year = os.path.join(path_to_clean_data, year)
        csv_files = glob.glob(os.path.join(path_to_clean_data_year, "*.csv"))

        for csv_file in csv_files:
            #  print(f"Processing: {csv_file}")

            df = pl.scan_csv(csv_file, try_parse_dates=True)
            df = df.with_columns(pl.col("date").cast(pl.Datetime))

            # Extract the time component (HH:MM) and filter market hours (9:30 - 16:00 EST)
            df = df.with_columns(pl.col("date").dt.strftime("%H:%M").alias("time_only"))
            market_hours = [f"{h:02d}:{m:02d}" for h in range(9, 16) for m in range(0, 60, 5)][1:]  # 9:30 to 16:00
            df = df.filter(pl.col("time_only").is_in(market_hours))

            # Round timestamp to 5-minute bins
            df = df.with_columns((pl.col("date").dt.truncate("5m")).alias("5min_bar"))

            # Compute bid-ask spread
            df = df.with_columns((pl.col("ask-price") - pl.col("bid-price")).alias("spread"))

            # Compute average and median spread per 5-minute bar
            df_summary = df.group_by("5min_bar").agg([
                pl.mean("spread").alias("avg_spread"),
                pl.median("spread").alias("median_spread")
            ])

            lazy_frames.append(df_summary)

    if not lazy_frames:
        raise ValueError(f"No CSV data found for {ticker}.")

    final_summary = pl.concat(lazy_frames).collect()

    return final_summary

def compute_intraday_spread(ticker: str) -> pl.DataFrame:
    """
    Computes the average bid-ask spread per 5-minute interval, ensuring valid market hours (9:30 - 16:00).

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        pl.DataFrame: Intraday spread statistics averaged over all days.
    """
    path_to_clean_data = os.path.join(BASE_PATH, ticker)

    if not os.path.exists(path_to_clean_data):
        raise ValueError(f"No data found for {ticker} in {BASE_PATH}")

    years = [y for y in os.listdir(path_to_clean_data) if os.path.isdir(os.path.join(path_to_clean_data, y))]
    lazy_frames = []

    for year in years:
        path_to_clean_data_year = os.path.join(path_to_clean_data, year)
        csv_files = glob.glob(os.path.join(path_to_clean_data_year, "*.csv"))

        for csv_file in csv_files:
            print(f"Processing: {csv_file}")

            df = pl.scan_csv(csv_file, try_parse_dates=True)
            df = df.with_columns(pl.col("date").cast(pl.Datetime))

            # Convert date to EST timezone
            df = df.with_columns(pl.col("date").dt.convert_time_zone("America/New_York"))

            # Round timestamp to 5-minute bins and extract only the time (HH:MM)
            df = df.with_columns(
                pl.col("date").dt.truncate("5m").dt.strftime("%H:%M").alias("intraday_time")
            )

            # Filter data within market hours (9:30 - 16:00)
            market_hours = [f"{h:02d}:{m:02d}" for h in range(9, 16) for m in range(0, 60, 5)][1:]
            df = df.filter(pl.col("intraday_time").is_in(market_hours))

            # Compute bid-ask spread
            df = df.with_columns((pl.col("ask-price") - pl.col("bid-price")).alias("spread"))

            # Compute average spread per 5-minute bar
            df_summary = df.group_by("intraday_time").agg(
                pl.mean("spread").alias("avg_spread")
            )

            lazy_frames.append(df_summary)

    if not lazy_frames:
        raise ValueError(f"No CSV data found for {ticker}.")

    # Concatenate and compute final averages over all days
    spread_summary = pl.concat(lazy_frames).group_by("intraday_time").agg(
        pl.mean("avg_spread").alias("avg_spread")
    ).sort("intraday_time").collect()

    return spread_summary

def plot_intraday_spread(ticker: str):
    """
    Plots the average bid-ask spread over the trading day (9:30 - 16:00).

    Args:
        ticker (str): Stock ticker symbol.
    """
    spread_df = compute_intraday_spread(ticker)

    # Ensure correct sorting before plotting
    spread_df = spread_df.sort("intraday_time")

    # Define proper market hours from 9:30 to 16:00
    market_hours = [f"{h:02d}:{m:02d}" for h in range(9, 16) for m in range(0, 60, 5)][1:]  # 9:30 to 16:00
    spread_df = spread_df.filter(pl.col("intraday_time").is_in(market_hours))

    # ---- Plot ----
    plt.figure(figsize=size_graphs_eda)
    plt.plot(spread_df["intraday_time"], spread_df["avg_spread"], marker="o", linestyle="-", label="Mean Spread",
             color="blue")

    # Format the x-axis to show only every hour (9:30, 10:30, ..., 16:00)
    hourly_labels = [t for t in spread_df["intraday_time"] if t.endswith(":30") or t == "16:00"]
    plt.xticks(hourly_labels, rotation=45)

    plt.xlabel("Time (HH:MM)")
    plt.ylabel("Bid-Ask Spread")
    plt.title(f"Intraday Bid-Ask Spread - {ticker}", fontsize=14)

    plt.legend()
    plt.ylim(0, spread_df["avg_spread"].max() * 1.05)  # Add 5% margin
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("Graphs", exist_ok=True)
    plt.savefig(f'Graphs/{ticker}spread_intraday.png', dpi=1000)
    plt.show()

def plot_returns():
    # Read the CSV file
    cwd = os.getcwd()
    result_file = os.path.join(cwd, "data", "strat_of_strats.csv")
    df = pl.read_csv(result_file)

    # Convert 'day' column to datetime format
    df = df.with_columns(pl.col("day").str.strptime(pl.Date, "%Y-%m-%d"))

    # Define cutoff date
    cutoff_date = datetime(2007, 1, 1)

    # Split the data into two parts: before 2007 and from 2007 onwards
    df_before_2007 = df.filter(pl.col("day") < cutoff_date)
    df_from_2007 = df.filter(pl.col("day") >= cutoff_date)

    # Create subplots with independent y-axes
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    color_map = cm.get_cmap("tab20", len(df.columns) - 1)  # Unique colors for lines

    # Plot data before 2007
    for i, col in enumerate(df_before_2007.columns):
        if col != "day":
            axes[0].plot(
                df_before_2007["day"].to_list(),
                df_before_2007[col].to_list(),
                # label=col,
                color=color_map(i),
            )
    axes[0].set_title("Returns Before 2007")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Mean Return")
    axes[0].grid(True)

    # Plot data from 2007 onwards
    for i, col in enumerate(df_from_2007.columns):
        if col != "day":
            axes[1].plot(
                df_from_2007["day"].to_list(),
                df_from_2007[col].to_list(),
                label=col,
                color=color_map(i),
            )
    axes[1].set_title("Returns From 2007")
    axes[1].set_xlabel("Date")
    axes[1].grid(True)

    # Add legend below plots
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),  # Adjusted for better placement
        ncol=3,
        fontsize=10,
    )

    for ax in axes:
        plt.sca(ax)
        plt.xticks(rotation=25, ha="right")

    # Adjust layout for space
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.38)

    # Save and show the plot
    os.makedirs("Graphs", exist_ok=True)
    plt.savefig("Graphs/Returns_per_strategy.pdf", dpi=1000)
    plt.show()


def plot_best_returns():

    cwd = os.getcwd()
    result_file = os.path.join(cwd, "data", "strat_of_strats.csv")

    # Read the consolidated mean return DataFrame
    df = pl.read_csv(result_file)

    # Convert 'day' column to a datetime format for proper plotting
    df = df.with_columns(pl.col("day").str.strptime(pl.Date, "%Y-%m-%d"))

    columns = df.columns
    columns = [c for c in columns if c != "day"]

    df = df.with_columns(max=pl.max_horizontal(pl.exclude("day")))
    for col in columns:
        df = df.with_columns(
            pl.when(pl.col(col) == pl.col('max')).then(pl.col(col)).otherwise(None).alias(f"new_{col}")
        )

    columns_new = [c for c in df.columns if c.startswith("new_")]
    columns_new.append('day')
    df = df.select(columns_new)

    for col in df.columns[:-1]:
        df = df.with_columns(pl.col(col).alias(col[4:]))

    columns.append('day')
    df = df[columns]

    # Define the cutoff date
    cutoff_date = datetime(2007, 1, 1)

    # Split the data into two parts: before 2007 and from 2007 onwards
    df_before_2007 = df.filter(pl.col("day") < cutoff_date)
    df_from_2007 = df.filter(pl.col("day") >= cutoff_date)

    # Create subplots with independent y-axes
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Plot data before 2007
    for col in df_before_2007.columns:
        if col != "day":  # Skip the 'day' column
            axes[0].plot(df_before_2007["day"].to_list(), df_before_2007[col].to_list(), label=col)
    axes[0].set_title("Returns Before 2007")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Mean Return")
    axes[0].grid(True)

    # Plot data from 2007 onwards
    for col in df_from_2007.columns:
        if col != "day":  # Skip the 'day' column
            axes[1].plot(df_from_2007["day"].to_list(), df_from_2007[col].to_list())
    axes[1].set_title("Returns From 2007")
    axes[1].set_xlabel("Date")
    axes[1].grid(True)

    # Add legend below plots
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),  # Adjusted for better placement
        ncol=3,
        fontsize=10,
    )

    for ax in axes:
        plt.sca(ax)  # Switch to current axis
        plt.xticks(rotation=25, ha='right')  # Rotate labels 45 degrees, and right-align them

    # Adjust layout to make space for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.38)

    # Show the plot
    os.makedirs("Graphs", exist_ok=True)
    plt.savefig("Graphs/Returns_per_strategy_best_returns.pdf", dpi=1000)
    plt.show()


def generate_latex_table(data):

    table = "\\begin{tabular}{c|ll}\n\n"
    table += "Key & Category & Parameters \\\\\n\\hline\n"

    for key, filename in data.items():
        filename_no_ext = filename.replace('.csv', '').replace("__", "&").replace("_", "\\_").replace("_df", "")

        table += f"{key} & {filename_no_ext} \\\\\n"

    table += "\n"
    table += "\\end{tabular}"

    os.makedirs(f"Graphs", exist_ok=True)

    file_path = f"Graphs/Ticker_strat_overtime.tex"
    with open(file_path, 'w') as file:
        file.write(table)

    print(f"Dictionary of mapping saved at {file_path}")

def plot_best_strategy():
    cwd = os.getcwd()

    # Load the data
    strat_of_strats_file = os.path.join(cwd, 'data', "strat_of_strats.csv")
    best_returns_file = os.path.join(cwd, 'data', "best_returns_per_day.csv")
    strat_df = pl.read_csv(strat_of_strats_file)
    best_df = pl.read_csv(best_returns_file)

    # Extract the days and strategies
    days = best_df["day"]
    best_strategies = best_df["strategy"]

    # Get unique strategies to assign them y-coordinates
    all_strategies = [col for col in strat_df.columns if col != "day"]
    all_strategies_formatted = [extract_correct_name(col) for col in strat_df.columns if col != "day"]
    strategy_to_y = {strategy: i for i, strategy in enumerate(all_strategies)}

    # Map the best strategy to its y-coordinate
    y_coords = [strategy_to_y[strategy] for strategy in best_strategies]

    # Plot the data
    plt.figure(figsize=(15, 8))
    plt.scatter(days, y_coords, color='blue', label="Best Strategy", alpha=0.7, facecolors='None', edgecolors='blue', s=10)

    # Customize the plot
    plt.yticks(range(len(all_strategies)), all_strategies_formatted, fontsize=15)  # Reduced font size for y-labels
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Strategies", fontsize=12)
    plt.title("Best Strategy Over Time", fontsize=14)

    # Control x-axis tick frequency
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8))  # Display a maximum of 8 ticks on the x-axis

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save and show the plot
    os.makedirs(f"Graphs", exist_ok=True)
    plt.savefig(f"Graphs/Best_strategy.pdf", dpi=1000)
    plt.show()


def extract_correct_name(file_name) -> str:
    return file_name.replace('.csv', '').replace("__", " ").replace("_", " ").replace("df", "")

def plot_best_of_best(df):

    df_pandas = df.to_pandas()

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(df_pandas["day"], df_pandas["max_mean_ret"], label="Max Mean Return", color="blue", marker='o', markerfacecolor='none',linewidth=1, markersize=3)
    plt.xlabel("Day")
    plt.ylabel("Max Mean Return")
    plt.title("Max Mean Return Over Time")
    plt.legend()
    plt.xticks(rotation=30)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))

    # plt.grid(False)
    # Show the plot
    os.makedirs("Graphs", exist_ok=True)
    plt.savefig("Graphs/best_strat_cheat.png", dpi=1000)

    plt.show()


def plot_tracker_best_strat_periods(file_path, dict_trad=None):

    # data = pd.read_csv(file_path)

    df = pl.read_csv(file_path)

    # Assuming your original dataframe is 'df'
    df = df.with_columns(pl.col("day").str.strptime(pl.Date, "%Y-%m-%d"))

    # Create the dataframe for the period before June 2007
    df_before_june_2007 = df.filter(pl.col("day") < pl.date(year=2007, month=6,day=1)) #"2007-06-01"))

    # Create the dataframe for the period after June 2007
    df_after_june_2007 = df.filter(pl.col("day") >= pl.date(year=2007, month=6,day=1))
    name = ['before2007', 'after2007']

    for i, df in enumerate([df_before_june_2007, df_after_june_2007]):
        data = df.to_pandas()
        # Transform the data for heatmap plotting
        data_melted = data.melt(id_vars="day", var_name="Ticker", value_name="Value")

        # Pivot the data to create a matrix for the heatmap
        heatmap_data = data_melted.pivot(index="Ticker", columns="day", values="Value")
        heatmap_data = heatmap_data.T

        std_devs = heatmap_data.std()
        sorted_columns = std_devs.sort_values(ascending=True).index
        heatmap_data = heatmap_data[sorted_columns].T
        # Set the figure size
        plt.figure(figsize=(20, 10))

        sns.heatmap(heatmap_data, cmap="viridis", cbar_kws={'label': 'Value'}, annot=False, cbar=True,)

        # Customize the plot
        plt.title(f"Heatmap of Ticker Values Over Time ({name[i]})", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Ticker", fontsize=12)
        # plt.figtext(0.5, -0.05, 'Caption: This plot shows the values of different strategies for each ticker over time.',
        #             ha='center', fontsize=12, color='black')

        # Show the plot
        plt.tight_layout()


        os.makedirs("Graphs", exist_ok=True)
        plt.savefig(f"Graphs/Ticker_strat_overtime_{name[i]}.pdf", dpi=1000)
        plt.grid(False)
        plt.show()