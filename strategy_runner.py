#%%
import polars as pl
import os


def run_strategy(ticker: str, month: int, year: int, strategy: callable, **kwargs)-> pl.DataFrame:
    """
    Executes a trading strategy by fetching the relevant dataset and applying the strategy to compute daily returns.

    This function retrieves the appropriate CSV file based on the provided `ticker`, `month`, and `year`, and applies
    the specified `strategy` function to calculate daily returns. The resulting daily returns are then saved to a file
    in the specified directory structure. If the file already exists, the function will skip processing for that month.

    :param ticker: str
        The ticker symbol of the asset (e.g., "APA").
    :param month: int
        The month as an integer (1-12).
    :param year: int
        The year as an integer (e.g., 2004).
    :param strategy: callable
    :param strategy: callable
        A function that computes the daily returns for the given dataset.
    :param kwargs: dict
        Additional arguments passed to the `strategy` function. These may include:
        - For `momentum_excess_vol_strategy`: `est` (e.g., estimation window).
        - For "momentum_strat2": `rolling_window_size`.

    :return: None
        This function does not return any value. It processes the data and saves the results to a file.

    :note:
        - If the corresponding daily returns file already exists for the given month, the function will skip processing.
        - The function will create directories if they do not exist and will save the computed daily returns as a CSV file.
        - The path where the daily returns are saved follows the structure: `data/daily_returns/{strategy_name}/{ticker}/{year}/{month}_daily_returns.csv`.

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

    return daily_ret

def apply_strategy(strategy: callable, param_names : str, verbose : bool=False, **kwargs) -> None:
    """
    Applies the specified trading strategy to the available dataset for each ticker, year, and month.

    This function processes the clean data by applying the given strategy across all the data in the directory structure
    for each ticker, year, and month. If the daily returns for a specific month do not exist, the strategy will be applied
    and the results will be saved.

    :param strategy: callable
        The strategy function that computes daily returns for each ticker.
    :param verbose: bool, optional, default=False
        If set to `True`, the function will print the progress and the path of the saved results.
    :param kwargs: dict
        Additional arguments passed to the `strategy` function. These will be used when calling the `run_strategy` function
        for each ticker, year, and month.

    :return: None
        This function does not return any value. It applies the strategy to each dataset and saves the results.

    :note:
        - The function will only apply the strategy if the corresponding daily returns file does not already exist for a given month.
        - The strategy is applied to each ticker's dataset found in the `data/clean` directory, and the results are saved in the `daily_returns` directory.
        - This is a batch processing function for applying the strategy to multiple tickers over time.

    """

    cwd = os.getcwd()
    root_data_clean = os.path.join(cwd, 'data', 'clean')
    root_data_ret = os.path.join(cwd, 'data', "daily_returns", f"{strategy.__name__}_{param_names}")

    tickers = [ticker for ticker in os.listdir(root_data_clean) if os.path.isdir(os.path.join(root_data_clean, ticker))]

    for ticker in tickers:
        path_to_clean_data = os.path.join(root_data_clean, ticker)
        years = [y for y in os.listdir(path_to_clean_data) if os.path.isdir(os.path.join(path_to_clean_data, y))]
        for year in years:
            path_to_clean_data_year = os.path.join(path_to_clean_data, year)
            months = [m for m in os.listdir(path_to_clean_data_year) if os.path.isfile(os.path.join(path_to_clean_data_year, m))]
            for month in months:
                path_to_clean_data_month = os.path.join(path_to_clean_data_year, month)
                path_to_daily_ret_month = os.path.join(root_data_ret, ticker , year, f"{month[:2]}_daily_returns.csv")
                # Check if the output exist
                if not os.path.exists(path_to_daily_ret_month):
                    name = ticker +"_"+ year+"_"+month
                    print(f"Processing: {name} with strat: {strategy.__name__}")

                    # Apply the strategy and save the result
                    daily_ret = run_strategy(ticker=ticker, month=int(month[:2]), year=int(year), strategy=strategy, verbose=False, ** kwargs)

                    # Path to save daily returns
                    ticker_dir = os.path.join(root_data_ret, ticker, str(year))

                    # Create directories if they do not exist
                    os.makedirs(ticker_dir, exist_ok=True)

                    # Save the daily returns
                    output_file_name = f"{month[:2]}_daily_returns.csv"
                    output_file_path = os.path.join(ticker_dir, output_file_name)

                    # daily_ret is a Polars DF
                    daily_ret.collect().write_csv(output_file_path)

                    if verbose:
                        print(f"Daily returns saved to {output_file_path}")

                else:
                    print(f"Skipping: {path_to_clean_data_month} (already exists)")


def build_strat_df(strategy: callable, param_names : str) -> None:
    """
    Builds and updates a strategy-specific DataFrame with daily returns data for each ticker.

    This function creates a new DataFrame or updates an existing one with the daily returns of multiple tickers for a
    given strategy. It iterates through the `daily_returns` directory and aggregates the data for each ticker across years
    and months. The results are stored in a strategy-specific directory.

    :param strategy: callable
        The strategy function whose results are being aggregated into the DataFrame.

    :return: None
        This function does not return any value. It creates or updates a CSV file with the daily returns data for the strategy.

    :note:
        - A new DataFrame is created with "day" as the index, and each ticker's returns are added as columns.
        - The function ensures that only new daily returns data (those not already present) is added to the DataFrame.
        - The resulting DataFrame is saved in a strategy-specific directory within the `data/strategies` folder.
    """

    # General setup
    cwd = os.getcwd()
    root_data_strategies = os.path.join(cwd, 'data', 'strategies')
    os.makedirs(root_data_strategies, exist_ok=True)

    # Strategy-specific setup
    root_data_ret = os.path.join(cwd, 'data', "daily_returns", f"{strategy.__name__}_{param_names}")

    # Path to the strategy's DataFrame file
    df_path = os.path.join(root_data_strategies, f"{strategy.__name__}_{param_names}_df.csv")

    # Initialize the DataFrame with the correct structure, including "day"
    #if True:
    # Create a dummy row to ensure the schema is set correctly
    dummy_row = {"day": [None]}  # Initialize with "day" as None
    existing_df = pl.DataFrame(dummy_row)

    # Create other columns based on the known tickers (empty columns)
    existing_columns = ["day"]
    for ticker in os.listdir(os.path.join(cwd, 'data', "handler")):
        ticker_path = os.path.join(root_data_ret, ticker)
        if os.path.isdir(ticker_path):
            existing_columns.append(ticker)

    # Create the dataframe with all columns initialized to None
    for col in existing_columns[1:]:  # Skip "day" column for now
        existing_df = existing_df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    # Retard move to counter error
    existing_df = existing_df.with_columns(
        pl.col("day").cast(pl.Date).alias("day")
    )

    # Discard dummy
    existing_df = existing_df.filter(pl.col("day").is_not_null())

    #else os.path.exists(df_path):
    #    # Load the existing DataFrame
    #    existing_df = pl.read_csv(df_path)

    schema = {"day": pl.Date, "return": pl.Float64}

    # Traverse the daily returns directory
    for ticker in os.listdir(root_data_ret):
        ticker_path = os.path.join(root_data_ret, ticker)
        if not os.path.isdir(ticker_path):
            continue

        for year in os.listdir(ticker_path):
            year_path = os.path.join(ticker_path, year)
            if not os.path.isdir(year_path):
                continue

            for file in os.listdir(year_path):
                if not file.endswith("_daily_returns.csv"):
                    continue

                month = file[:2]  # Extract the month from the file name
                file_path = os.path.join(year_path, file)

                # Read daily returns with correspionding shema
                daily_returns = pl.read_csv(file_path, schema=schema)

                #Checks
                if "day" not in daily_returns.columns:
                    raise ValueError(f"File {file_path} is missing the 'day' column.")
                if "return" not in daily_returns.columns:
                    raise ValueError(f"File {file_path} is missing the 'return' column.")

                # Convert "day" to list for comparison
                existing_days = existing_df["day"].to_list()

                # Filter rows that are already in the existing DataFrame
                ticker_col = ticker  # Column name for the ticker

                # Debug: Print daily returns for ticker
                print(f"Daily returns for {ticker}:")

                # Update the DataFrame with the new data
                for row in daily_returns.iter_rows(named=True):
                    day, value = row["day"], row["return"]  # Assuming "day" and "return" values
                    if day not in existing_days:
                        # Add a new row with appropriate column types
                        new_row = {col: None if col != "day" else day for col in existing_df.columns}
                        new_row[ticker_col] = value if value is not None else None  # Ensure correct type for ticker_col

                        # Ensure all columns are initialized with the correct types
                        new_row = {k: v if v is not None else None for k, v in new_row.items()}

                        # Debug: Print new row
                        #print(f"New row to append: {new_row}")

                        # Convert new_row to a DataFrame and append
                        new_row_df = pl.DataFrame([new_row])

                        # Stack them together
                        existing_df = existing_df.vstack(new_row_df)
                    else:
                        #add the returns in the correct row and column
                        existing_df = existing_df.with_columns(
                            pl.when(pl.col("day") == day)
                            .then(pl.lit(value).cast(pl.Float64))
                            .otherwise(pl.col(ticker_col))
                            .alias(ticker_col)
                        )
                # Ensure "day" is ordered chronologically
                existing_df = existing_df.sort("day")

    # Save the updated DataFrame
    existing_df.write_csv(df_path)
    print(f"Updated strategy DataFrame saved to {df_path}")
    return None

def best_strat_finder():
    cwd = os.getcwd()
    root_data_strategies = os.path.join(cwd, 'data', 'strategies')
    strategies = [
        strat for strat in os.listdir(root_data_strategies)
        if os.path.isfile(os.path.join(root_data_strategies, strat)) and strat.endswith('.csv')
    ]
    # Initialize a DataFrame for the "best strategy" with the same schema as the first strategy file
    first_file_path = os.path.join(root_data_strategies, strategies[0])
    best_df = pl.read_csv(first_file_path)

    # Iterate through strategy files and update the best_df with the maximum daily returns
    for strat_file in strategies[1:]:
        strat_file_path = os.path.join(root_data_strategies, strat_file)
        current_df = pl.read_csv(strat_file_path)

        # Update best_df with the maximum values for each column except 'day'
        best_df = best_df.with_columns([
            pl.when(current_df[col] > best_df[col]).then(current_df[col]).otherwise(best_df[col]).alias(col)
            for col in best_df.columns if col != "day"
        ])
    best_df.write_csv(os.path.join(cwd, 'data', "optimum.csv"))

