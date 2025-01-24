import os

import polars as pl


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

import momentum
parameters_mom = {"short_window": 100, "long_window": 1000, "plot": False}

run_strategy(ticker='ABT', month=2, year=2004, strategy=momentum.momentum_strat2, yearly=False, parameters = parameters_mom)
