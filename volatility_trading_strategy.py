
import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def volatility_trading_strategy(df, parameters):
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

    # Clean up the dates
    df = df.with_columns(
        pl.col("date")
        .str.slice(0, 19)  # Extract only the first 19 characters (YYYY-mm-ddTHH:MM:SS)
        .str.replace("T", " ")  # Replace 'T' with a space
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")  # Parse as datetime
        .alias("date")  # Optional: Rename the column if needed
    )

    volatility_threshold = parameters.get("volatility_threshold", 2.0)

    df = df.sort(by='date')

    df = df.with_columns(
        (pl.col('trade-price')/pl.col('trade-price').shift(1)).log().alias('log_return')
    )

    df = df.drop_nulls()

    df = df.with_columns(
        pl.col('log_return').rolling_std_by("date", window_size=f"{parameters['short_window']}s").alias('short_vol'),
        pl.col('log_return').rolling_std_by("date", window_size=f"{parameters['long_window']}s").alias('long_vol')
    )

    df = df.with_columns(
        (pl.col('short_vol') > (volatility_threshold * pl.col('long_vol'))).alias('short_above')
    )

    df = df.with_columns(
        pl.when((~ pl.col('short_above').shift(-1)) & pl.col('short_above')).then(1)
        .when(pl.col('short_above').shift(-1) & (~pl.col('short_above'))).then(-1)
        .otherwise(0)
        .alias('trigger')
    )

    df = df.with_columns(
        pl.when(pl.col('trigger') == 1).then(pl.col('trade-price')).otherwise(None).alias('buy'),
        pl.when(pl.col('trigger') == -1).then(pl.col('trade-price')).otherwise(None).alias('sell')
    )

    df = df.with_columns(pl.col("date").dt.date().alias("day"))

    daily_returns = df.group_by("day").map_groups(
        compute_strategy_return,
        schema={"day": pl.Date, "return": pl.Float64}
    )


    if parameters['plot']:
        df = df.collect().tail(10000)
        df_pandas = df.to_pandas()
        plt.figure(figsize=(15, 10))

        plt.plot(df_pandas['trade-price'], label='Trade', color='black')
        # Add dots for buy/sell signals
        plt.scatter(np.arange(df_pandas.shape[0]), df_pandas['buy'], color='red', label='Buy', s=50, zorder=5,
                    facecolors='none', edgecolors='r')
        plt.scatter(np.arange(df_pandas.shape[0]), df_pandas['sell'], color='red', label='Sell', s=50, zorder=5,
                    marker='x')
        # Add legend for primary axis
        plt.legend(loc="upper left")

        # Save and show the plot
        os.makedirs("Graphs", exist_ok=True)
        plt.savefig(
            f'Graphs/example_signal_volatility_short{parameters["short_window"]}_long{parameters['long_window']}.pdf',
            dpi=1000)
        plt.show()

    return daily_returns


def compute_strategy_return(group: pl.DataFrame) -> pl.DataFrame:
    # Extract buy and sell signals
    buy_signals = group.filter(pl.col("trigger") == 1)
    sell_signals = group.filter(pl.col("trigger") == -1)

    # Ensure there are equal numbers of buy and sell signals
    min_trades = min(len(buy_signals), len(sell_signals))
    buy_signals = buy_signals.head(min_trades)
    sell_signals = sell_signals.head(min_trades)

    # Calculate returns
    if min_trades > 0:
        returns = (sell_signals["trade-price"] - buy_signals["trade-price"]) / buy_signals["trade-price"]
        return_sum = returns.sum()
    else:
        return_sum = 0.0  # No trades executed

    # Return as a DataFrame with the computed value
    return pl.DataFrame({"day": [group["day"][0]], "return": [return_sum]})