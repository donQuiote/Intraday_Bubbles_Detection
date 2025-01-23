import pandas as pd
import numpy as np

def volatility_trading_strategy(tick, df, params=None):
    """
    Implements a volatility-based trading strategy.

    Parameters:
    - tick (str): The stock ticker symbol (used for reference).
    - df (pd.DataFrame): The dataset containing market data with a 'date' column.
    - params (dict, optional): Dictionary of strategy parameters.
        - short_window (int): Rolling window for short-term volatility (default=30).
        - long_window (int): Rolling window for long-term volatility (default=240).
        - volatility_threshold (float): Multiplier for detecting abnormal volatility (default=2.0).

    Returns:
    - pd.DataFrame: A DataFrame containing 'date' and 'daily_return'.
    """

    # Default parameters if none provided
    if params is None:
        params = {
            "short_window": 30,
            "long_window": 240,
            "volatility_threshold": 2.0
        }

    short_window = params.get("short_window", 30)
    long_window = params.get("long_window", 240)
    volatility_threshold = params.get("volatility_threshold", 2.0)

    # Ensure 'date' is a datetime object
    df["date"] = pd.to_datetime(df["date"])

    # Sort by date
    df = df.sort_values(by="date")

    # Calculate log returns for trade-price
    df["log_return"] = np.log(df["trade-price"] / df["trade-price"].shift(1))

    # Compute volatility using rolling standard deviation
    df["short_volatility"] = df["log_return"].rolling(short_window).std()
    df["long_volatility"] = df["log_return"].rolling(long_window).std()

    # Define entry condition: When short-term volatility exceeds threshold * long-term volatility
    df["signal"] = np.where(df["short_volatility"] > volatility_threshold * df["long_volatility"], 1, 0)  # Enter position

    # Define exit condition: When short-term volatility returns below long-term volatility
    df["exit"] = np.where(df["short_volatility"] < df["long_volatility"], 1, 0)  # Exit position

    # Track positions
    df["position"] = 0  # 1 for long, 0 for out
    in_trade = False  # Track whether we are in a trade

    for i in range(1, len(df)):
        if df.iloc[i]["signal"] == 1 and not in_trade:
            df.at[df.index[i], "position"] = 1  # Enter trade
            in_trade = True
        elif df.iloc[i]["exit"] == 1 and in_trade:
            df.at[df.index[i], "position"] = 0  # Exit trade
            in_trade = False
        else:
            df.at[df.index[i], "position"] = df.at[df.index[i - 1], "position"]  # Carry position forward

    # Calculate daily returns when in position
    df["daily_return"] = df["log_return"] * df["position"]

    # Compute daily return per day
    df_daily_returns = df.groupby(df["date"].dt.date)["daily_return"].sum().reset_index()
    df_daily_returns.columns = ["date", "return"]
    

    # Generate full date range for the given month
    start_date = df_daily_returns["date"].min()
    end_date = df_daily_returns["date"].max()
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Merge with full date range and fill missing values with 0
    df_full = pd.DataFrame({"date": full_date_range})
    df_daily_returns["date"] = pd.to_datetime(df_daily_returns["date"])
    df_full = df_full.merge(df_daily_returns, on="date", how="left").fillna(0)

    return df_full
    

