import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from Strategies.momentum import compute_strategy_return


def momentum_excess_vol(df, parameters):

    # TODO: remove this
    df = df.with_columns(
        pl.col("date")
        .str.slice(0, 19)  # Extract only the first 19 characters (YYYY-mm-ddTHH:MM:SS)
        .str.replace("T", " ")  # Replace 'T' with a space
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")  # Parse as datetime
        .alias("date")  # Optional: Rename the column if needed
    )

    df = df.sort(by='date')

    df = df.with_columns(
        pl.col('trade-price').rolling_mean_by('date', window_size=f"{parameters['long_window_price']}s").alias('L_MA-price'),
        pl.col('trade-price').rolling_mean_by('date', window_size=f"{parameters['short_window_price']}s").alias('S_MA-price'),
        pl.col('trade-volume').rolling_mean_by('date', window_size=f"{parameters['long_window_volume']}s").alias('L_MA-volume'),
        pl.col('trade-volume').rolling_mean_by('date', window_size=f"{parameters['short_window_volume']}s").alias('S_MA-volume')
    )

    # Create a bool for the price and the volume, is True is ST moving average smaller than LT moving average
    df = df.with_columns(
        (pl.col("S_MA-price") < pl.col("L_MA-price")).alias("S_MA_leq_L_MA-price"),
        (pl.col("S_MA-volume") < pl.col("L_MA-volume")).alias("S_MA_leq_L_MA-volume")
    )

    # Create the trigger of the trading strategy: if volume of STMA > LTMA at t-1, and STMA < LTMA at t
    df = df.with_columns(
        pl.when((~pl.col('S_MA_leq_L_MA-volume').shift(-1)) & pl.col('S_MA_leq_L_MA-volume'))
        .then(1)
        .otherwise(0)
        .alias('trigger-event')
    )

    # At the trigger events, buy if the price STMA > LTMA, sell otherwise
    df = df.with_columns(
        pl.when((pl.col('trigger-event') == 1) & (~ pl.col('S_MA_leq_L_MA-price')))
        .then(1).otherwise(0).alias('trigger-buy'),
        pl.when((pl.col('trigger-event') == 1) & (pl.col('S_MA_leq_L_MA-price')))
        .then(1).otherwise(0).alias('trigger-sell')
    )

    df = df.with_columns((pl.col('trigger-buy')-pl.col('trigger-sell')).alias('position'))

    len_df = df.select(pl.count()).collect().item() # Get the LazyFrame length

    action = []
    cum = 0

    df_ = df.collect()

    for row in range(len_df):
        a = bool_condition(dataframe=df_, idx=row, cum_1=cum) #1 if ((df_.select('trigger-buy')[row] == 1) and (cum[-1]==0 or cum[-1]==-1)) elif 0 else 1
        action.append(a)
        cum += a

    df = df.with_columns(
        pl.Series(name = 'trigger', values=action)
    )

    df = df.with_columns(pl.col("date").dt.date().alias("day"))

    # Group by day and calculate the daily return
    daily_returns = df.group_by("day").map_groups(
        compute_strategy_return,
        schema={"day": pl.Date, "return": pl.Float64}
    )

    df = df.with_columns(
        pl.when(pl.col('trigger')==1).then(pl.col('trade-price')).otherwise(None).alias('buy'),
        pl.when(pl.col('trigger') == -1).then(pl.col('trade-price')).otherwise(None).alias('sell')
    )

    if parameters['plot']:
        df = df.collect().tail(10000)
        df_pandas = df.to_pandas()

        plt.figure(figsize=(15, 10))

        # Plot the primary axis data
        plt.plot(df_pandas['trade-price'], label='Trade', color='black')
        plt.plot(df_pandas['S_MA-price'], label=f"ST MA price ({parameters['short_window_price']}s)", color='red',
                 linewidth=0.5, alpha=0.5)
        plt.plot(df_pandas['L_MA-price'], label=f"LT MA price ({parameters['long_window_price']}s)", color='red',
                 linewidth=0.5, alpha=0.5, linestyle='--')

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
        plt.savefig(
            f'Graphs/example_signal_mom_volume_p_sma{parameters["short_window_price"]}_p_lma{parameters["long_window_price"]}_v_sma{parameters["short_window_volume"]}_v_lma{parameters["long_window_volume"]}.pdf',
            dpi=1000)
        plt.show()

    return daily_returns

def bool_condition(dataframe, idx, cum_1):
    if (dataframe['trigger-buy'][idx] == 1) and (cum_1 == 0 or cum_1 == -1):
        return 1
    elif (dataframe['trigger-sell'][idx] == 1) and (cum_1 == 1):
        return -1
    else:
        return 0

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
    df = df.with_columns(
        pl.col("date")
        .str.slice(0, 19)  # Extract only the first 19 characters (YYYY-mm-ddTHH:MM:SS)
        .str.replace("T", " ")  # Replace 'T' with a space
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")  # Parse as datetime
        .alias("date")  # Optional: Rename the column if needed
    )

    df = df.unique(subset=["date"])

    df = df.sort(by='date')
    print(df.collect())

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
    df = df.join(daily_stats, on="day", how="left")

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
    df = df.collect()
    date = df.select(["date"]).to_numpy()
    enter_scheme = df.select(["trading_scheme_enter"]).to_numpy()
    leave_scheme = df.select(["trading_scheme_leave"]).to_numpy()
    stock_price = df.select(["trade-price"]).to_numpy()
    stock_return = df.select(["returns"]).to_numpy()

    ask_price = df.select(["ask-price"]).to_numpy()
    bid_price = df.select(["bid-price"]).to_numpy()

    positions = np.zeros(len(date))
    returns = np.zeros(len(date))

    in_position = False
    position = 0
    value = 0

    # Add a column for tracking days
    days = df.select(["day"]).to_numpy().flatten()

    for i in range(len(date)):
        if enter_scheme[i] == 1 and not in_position:
            if stock_return[i] > 0:
                value = ask_price[i]
                position = +1
                positions[i] = 1

            else:
                value = bid_price[i]
                position = -1
                positions[i] = -1
            in_position = True

        elif leave_scheme[i] == 1 and in_position:
            if position == 1:
                ret = bid_price[i] / value - 1
                positions[i] = 2
                returns[i] = ret
            else:
                ret = -1 * (ask_price[i] / value - 1)
                positions[i] = -2
                returns[i] = ret

    dates_ret = np.column_stack((days, returns))
    dates_ret_cleaned = [
        {"day": row[0], "return": row[1]} for row in dates_ret
    ]

    # Create a Polars DataFrame with the correct schema
    dates_ret_df = pl.DataFrame(
        dates_ret_cleaned,
        schema={"day": pl.Utf8, "return": pl.Float64}  # Temporarily use Utf8 for 'day'
    ).with_columns(
        pl.col("day").str.strptime(pl.Date, "%Y-%m-%d").alias("day")  # Convert 'day' to Date
    )
    daily_returns = dates_ret_df.group_by("day").sum().sort(by='day')

    return daily_returns

def extreme_excess_vol_reversion(df):
    return None