import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

parameters_mom = {
    "short_window": 100,
    "long_window": 500,
    "plot":False
}


def momentum_price(df:pl.DataFrame, parameters:dict=parameters_mom) -> pl.DataFrame:
    """Strategy: start with position of 0 and: go short when S_MA crosses L_MA from above, go long when S_MA crosses L_MA from below"""
    # Format the date column
    df = df.with_columns(
        pl.col("date")
        .str.slice(0, 19)  # Extract only the first 19 characters (YYYY-mm-ddTHH:MM:SS)
        .str.replace("T", " ")  # Replace 'T' with a space
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")  # Parse as datetime
        .alias("date")  # Optional: Rename the column if needed
    )

    df = df.sort(by='date')

    df = df.with_columns(
        pl.col('trade-price').rolling_mean(window_size=parameters['long_window']).alias('L_MA-price'),
        pl.col('trade-price').rolling_mean(window_size=parameters['short_window']).alias('S_MA-price')
    )

    df = df.with_columns(
        (pl.col("S_MA-price") < pl.col("L_MA-price")).alias("S_MA_leq_L_MA")
    )

    df = df.with_columns(
        pl.when(pl.col('S_MA_leq_L_MA').shift(-1) & (~ pl.col('S_MA_leq_L_MA')))
        .then(1)
        .when((~ pl.col('S_MA_leq_L_MA').shift(-1)) & pl.col('S_MA_leq_L_MA'))
        .then(-1)
        .otherwise(0)
        .alias('trigger')
    )

    df = df.with_columns(
        pl.when(pl.col('trigger') == 1).then(pl.col('trade-price')).otherwise(None).alias('buy'),
        pl.when(pl.col('trigger') == -1).then(pl.col('trade-price')).otherwise(None).alias('sell')
    )

    df = df.with_columns(pl.col("date").dt.date().alias("day"))

    # Group by day and calculate the daily return
    daily_returns = df.group_by("day").map_groups(
        compute_strategy_return,
        schema={"day": pl.Date, "return": pl.Float64}
    )

    if parameters['plot']:
        # Example: Convert your Polars DataFrame to Pandas for plotting
        df = df.collect().tail(10000)
        df_pandas = df.to_pandas()

        plt.figure(figsize=(15, 10))


        # Plot the primary axis data
        plt.plot(df_pandas['trade-price'], label='Trade', color='black')
        # plt.plot(df_pandas['S_M-price'], label='Short', color='green', linewidth=0.5, alpha=0.5)
        # plt.plot(df_pandas['L_M-price'], label='Long', color='blue', linewidth=0.5, alpha=0.5)
        plt.plot(df_pandas['L_MA-price'], label=f"Long MA ({parameters['long_window']})", color='blue', linestyle='--')
        plt.plot(df_pandas['S_MA-price'], label=f"Short MA ({parameters['short_window']})", color='blue')

        # Add dots for buy/sell signals
        plt.scatter(np.arange(df_pandas.shape[0]), df_pandas['buy'], color='red', label='Buy', s=50, zorder=5,
                    facecolors='none', edgecolors='r')
        plt.scatter(np.arange(df_pandas.shape[0]), df_pandas['sell'], color='red', label='Sell', s=50, zorder=5,
                    marker='x')

        plt.xlabel('Time')
        plt.ylabel('Trade price')

        # Add legend for primary axis
        plt.legend(loc="upper left")

        # Save and show the plot
        os.makedirs("../Graphs", exist_ok=True)
        plt.savefig(f'Graphs/example_signal_mom_sma{parameters_mom["short_window"]}_lma{parameters_mom["long_window"]}.pdf', dpi=1000)
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