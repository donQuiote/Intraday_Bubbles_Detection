import polars as pl
import os


from Strategies import momentum_excess_vol_strategy
parameters_mom = {
    "short_window": 100,
    "long_window": 500,
    "plot":True
}

def momentum_strat2(df:pl.DataFrame, parameters:dict=parameters_mom) -> pl.DataFrame:
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
        pl.col('trade-price').shift(parameters['short_window']).alias('S_M-price'),
        pl.col('trade-price').shift(parameters['long_window']).alias('L_M-price'),
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

    return  daily_returns

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

def run_strategy(ticker: str, month: int, year: int, strategy: callable, yearly: bool = False, **kwargs)-> pl.DataFrame:
    """
    Executes a trading strategy by fetching and loading the appropriate dataset.

    This function retrieves the correct CSV file based on the provided `ticker`, `month`,
    and `year`, and applies the specified `strategy` to compute daily returns. If `yearly`
    is set to True, the function will load all files for the given year.

    Parameters
    ----------
    ticker : str
        The ticker symbol of the asset (e.g., "APA").
    month : int
        The month as an integer (1-12).
    year : int
        The year as an integer (e.g., 2004).
    strategy : callable
        A function that computes the daily returns of the strategy.
    yearly : bool, optional
        If True, loads all files for the specified year. Defaults to False.
    **kwargs : dict
        Additional arguments passed to the `strategy` function. Expected keys:
        - For `momentum_excess_vol_strategy`: `est` (e.g., estimation window).
        - For other strategies: Additional parameters specific to the strategy.

    Returns
    -------
    pl.LazyFrame
        A Polars LazyFrame containing the loaded data with computed strategy results.
    """
    # Ensure month is zero-padded
    month_str = f"{month:02d}"

    #Path to ticker and year
    root = os.getcwd()
    root_data = os.path.join(root, "data")
    base_path = os.path.join(root_data, "clean", ticker, str(year))

    #Checker
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Directory {base_path} does not exist, generate usign the main. Check for layout.")

    # Construct the file name
    file_name = f"{month_str}_bbo_trade.csv" #Hardcoded watchout
    file_path = os.path.join(base_path, file_name)

    #Checker
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist, generate usign the main. Check for layout.")

    # Load the CSV file
    df_scanner = pl.scan_csv(file_path)
    daily_ret = strategy(df_scanner, **kwargs)

    # Path to save daily returns
    daily_returns_dir = os.path.join(root_data, "daily_returns")
    strategy_dir = os.path.join(daily_returns_dir, strategy.__name__)
    ticker_dir = os.path.join(strategy_dir, ticker, str(year))

    # Create directories if they do not exist
    os.makedirs(ticker_dir, exist_ok=True)

    # Save the daily returns
    output_file_name = f"{month_str}_daily_returns.csv"
    output_file_path = os.path.join(ticker_dir, output_file_name)

    # daily_ret is a Polars DF
    daily_ret.collect().write_csv(output_file_path)

    print(f"Daily returns saved to {output_file_path}")

    return None

run_strategy(ticker= "APA", month = 9 , year = 2004, strategy=momentum_strat2, parameters=parameters_mom)