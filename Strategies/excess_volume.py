import numpy as np
import polars as pl


def short_excess_vol_strategy(df):
    enter_scheme = df.select(["trading_scheme_enter"]).to_numpy()
    leave_scheme = df.select(["trading_scheme_leave"]).to_numpy()
    stock_price = df.select(["trade-price"]).to_numpy()

    positions = np.zeros(len(df))

    in_position = False
    position_count = 0
    value = 0

    for i in range(len(df)):
        if enter_scheme[i] == 1 and not in_position:

            value -= stock_price[i]
            in_position = True
            position_count += 1
            positions[i] = 1

        elif leave_scheme[i] == 1 and in_position:
            value += stock_price[i]
            in_position = False

        elif in_position:
            positions[i] = 1

    print(f"Number of times in position: {position_count}")

    time = np.arange(len(df))

    #plot_trades(stock_price, time, positions)

    return df, value

def momentum_excess_vol_strategy(df, thresh, plot_graph=True):
    df = df.collect()

    df = df.with_columns(
        pl.col("date")
        .str.slice(0, 19)  # Extract only the first 19 characters (YYYY-mm-ddTHH:MM:SS)
        .str.replace("T", " ")  # Replace 'T' with a space
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")  # Parse as datetime
        .alias("date")  # Optional: Rename the column if needed
    )

    df = df.sort(by='date')

    df = df.with_columns(
        (pl.col("trade-volume").diff()).alias("vol_diff"),
        (pl.col("trade-price").pct_change()).alias("returns")
    )

    df = df.with_columns(
        pl.col("date").dt.strftime("%Y-%m-%d").alias("day")  # Create a "day" column for grouping
    )

    # Compute daily mean and std for "vol_diff/dt_ms"
    daily_stats = df.group_by("day").agg([
        pl.col("vol_diff").mean().alias("daily_vol_mean"),
        pl.col("vol_diff").std().alias("daily_vol_std"),
    ])

    # Shift the daily stats to use the previous day's values
    daily_stats = daily_stats.with_columns(
        pl.col("daily_vol_mean").shift(1).fill_null(pl.col("daily_vol_mean")),
        pl.col("daily_vol_std").shift(1).fill_null(pl.col("daily_vol_std"))
    )

    # Join daily statistics back to the main DataFrame
    df = df.join(daily_stats, on="day", how="left")

    # Generate trading signals
    df = df.with_columns(
        (pl.when(pl.col("vol_diff") > pl.col("daily_vol_mean") + thresh * pl.col("daily_vol_std"))
        .then(1)
        .otherwise(
            pl.when(pl.col("vol_diff") < pl.col("daily_vol_mean") - thresh * pl.col("daily_vol_std"))
            .then(-1)
            .otherwise(0)
        )).alias("trading_scheme_enter").shift(),
        pl.col("returns").shift(),
        (pl.when(pl.col("vol_diff") < pl.col("daily_vol_mean")).then(1)
        .otherwise(
            pl.when(pl.col("vol_diff") > pl.col("daily_vol_mean")).then(-1).otherwise(0)
        )).alias("trading_scheme_leave").shift()
    )

    # Initialize variables for simulation
    enter_scheme = df.select(["trading_scheme_enter"]).to_numpy()
    leave_scheme = df.select(["trading_scheme_leave"]).to_numpy()
    stock_price = df.select(["trade-price"]).to_numpy()
    stock_return = df.select(["returns"]).to_numpy()

    ask_price = df.select(["ask-price"]).to_numpy()
    bid_price = df.select(["bid-price"]).to_numpy()

    positions = np.zeros(len(df))
    daily_returns = []  # List to store tuples of (day, return)
    daily_value_change = 0  # Array to track daily returns for each day

    in_position = False
    position = 0
    position_count = 0
    value = 0
    daily_value_change = 0  # Track P&L for the current day

    # Add a column for tracking days
    days = df.select(["day"]).to_numpy().flatten()
    prev_day = None

    for i in range(len(df)):
        current_day = days[i]

        # If it's a new day, finalize the previous day's return
        if prev_day is not None and current_day != prev_day:
            daily_returns.append((prev_day, daily_value_change))  # Append the previous day's return
            daily_value_change = 0  # Reset for the new day

        # Enter or exit trades
        if enter_scheme[i] == 1 and not in_position:
            if stock_return[i] > 0:
                value += ask_price[i]
                position = +1
            elif stock_return[i] < 0:
                value -= bid_price[i]
                position = -1
            if position != 0:
                in_position = True
                position_count += 1
                positions[i] = 1

        elif leave_scheme[i] == 1 and in_position:
            if position == +1:
                value -= bid_price[i]
            else:
                value += ask_price[i]
            daily_value_change += (value - stock_price[i]) / stock_price[i]  # Update daily return
            in_position = False

        elif in_position:
            positions[i] = 1

        prev_day = current_day

    # Capture returns for the final day
    daily_returns.append((prev_day, daily_value_change))

    # Print statistics
    print(f"Number of times in position: {position_count}")
    print(f"Daily returns: {daily_returns}")

    # Plot trades
    time = np.arange(len(df))
    #if plot_graph:
        #plot_trades(stock_price, time, positions)

    daily_returns_df = pl.DataFrame(daily_returns, schema=["day", "return"])

    return df, daily_returns

def extreme_excess_vol_reversion(df):
    return None