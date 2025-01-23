import numpy as np
import polars as pl

from Utils import plot_trades

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

    plot_trades(stock_price, time, positions)

    return df, value

def momentum_excess_vol_strategy(df, plot_graph=True):
    enter_scheme = df.select(["trading_scheme_enter"]).to_numpy()
    leave_scheme = df.select(["trading_scheme_leave"]).to_numpy()
    stock_price = df.select(["trade-price"]).to_numpy()
    stock_return = df.select(["returns/dt_ms*avg_dt"]).to_numpy()

    #"bid-price","ask-price"
    ask_price = df.select(["ask-price"]).to_numpy()
    bid_price = df.select(["bid-price"]).to_numpy()

    positions = np.zeros(len(df))

    in_position = False
    position = 0
    position_count = 0
    value = 0

    for i in range(len(df)):
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
            else :
                value += ask_price[i]
            in_position = False

        elif in_position:
            positions[i] = 1

    print(f"Number of times in position: {position_count}")

    time = np.arange(len(df))

    if plot_graph:
        plot_trades(stock_price, time, positions)

    return df, value

def extreme_excess_vol_reversion(df):
    return None