import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "us_equities_logreturns_cut.parquet"
df = pd.read_parquet(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Define a function to calculate sign ratios for combined stocks
def calculate_sign_ratios_combined(data, block_size):
    """
    Calculate the ratio of positive (+) and negative (-) signs across all stocks combined in fixed blocks.

    Parameters:
        data (pd.DataFrame): DataFrame containing the trade data for all stocks.
        block_size (int): Number of trades per block.

    Returns:
        pd.DataFrame: A DataFrame summarizing the ratio of + and - signs for each block.
    """
    results = []  # Store results for each block

    for start in range(0, len(data) - block_size + 1, block_size):
        # Extract the block
        block = data.iloc[start:start + block_size]
        
        # Calculate positive and negative ratios across all columns in the block
        positive_ratio = (block > 0).mean().mean() * 100  # Mean of positive signs across all stocks
        negative_ratio = (block < 0).mean().mean() * 100  # Mean of negative signs across all stocks
        
        # Append results
        results.append({
            'start_index': start,
            'end_index': start + block_size - 1,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio
        })
    
    # Convert results to a DataFrame
    return pd.DataFrame(results)

# Set the block size
block_size = 50

# Calculate the sign ratios for all stocks combined
ratios_df = calculate_sign_ratios_combined(df, block_size)

# Display the first few rows of the results
print("Combined Positive and Negative Ratios by Block:")
print(ratios_df.head())

# Plot histogram for the distribution of positive ratios
plt.figure(figsize=(10, 6))
sns.histplot(ratios_df['positive_ratio'], bins=30, kde=True, color='blue', label='Positive Ratio (All Stocks)')

# Add labels and title
plt.title('Distribution of Positive Ratios Across All Stocks Combined')
plt.xlabel('Positive Ratio (%)')
plt.ylabel('Density')
plt.legend()
plt.grid()

# Show the plot
plt.show()
