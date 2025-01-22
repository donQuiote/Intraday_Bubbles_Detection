import matplotlib.pyplot as plt
import numpy as np
import polars as pl

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