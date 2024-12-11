import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset
file_path = "us_equities_logreturns_cut.parquet"
df = pd.read_parquet(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Define a function to calculate sign ratios for combined stocks
def calculate_sign_ratios_combined(data, block_size):
    results = []  # Store results for each block

    for start in range(0, len(data) - block_size + 1, block_size):
        # Extract the block
        block = data.iloc[start:start + block_size]

        # Calculate positive and negative ratios
        positive_ratio = (block > 0).mean().mean() * 100
        negative_ratio = (block < 0).mean().mean() * 100

        # Append results
        results.append({
            'start_index': start,
            'end_index': start + block_size - 1,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio
        })

    # Convert results to a DataFrame
    return pd.DataFrame(results)

# Function to calculate magnitude-weighted sign ratios
def calculate_magnitude_weighted_sign_ratios(data, block_size):
    results = []

    for start in range(0, len(data) - block_size + 1, block_size):
        block = data.iloc[start:start + block_size]
        weights = block.abs()

        weighted_positive = ((block > 0) * weights).sum().sum() / weights.sum().sum() * 100
        weighted_negative = ((block < 0) * weights).sum().sum() / weights.sum().sum() * 100

        results.append({
            'start_index': start,
            'end_index': start + block_size - 1,
            'magnitude_weighted_positive_ratio': weighted_positive,
            'magnitude_weighted_negative_ratio': weighted_negative
        })

    return pd.DataFrame(results)

# Function to calculate volatility-weighted sign ratios
def calculate_volatility_weighted_sign_ratios(data, block_size, rolling_window=5):
    rolling_volatility = data.rolling(window=rolling_window, min_periods=1).std()
    results = []

    for start in range(0, len(data) - block_size + 1, block_size):
        block = data.iloc[start:start + block_size]
        block_volatility = rolling_volatility.iloc[start:start + block_size]

        weighted_positive = ((block > 0) * block_volatility).sum().sum() / block_volatility.sum().sum() * 100
        weighted_negative = ((block < 0) * block_volatility).sum().sum() / block_volatility.sum().sum() * 100

        results.append({
            'start_index': start,
            'end_index': start + block_size - 1,
            'volatility_weighted_positive_ratio': weighted_positive,
            'volatility_weighted_negative_ratio': weighted_negative
        })

    return pd.DataFrame(results)

# Set the block size
block_size = 20

# Calculate ratios
ratios_df = calculate_sign_ratios_combined(df, block_size)
magnitude_weighted_ratios_df = calculate_magnitude_weighted_sign_ratios(df, block_size)
volatility_weighted_ratios_df = calculate_volatility_weighted_sign_ratios(df, block_size, rolling_window=5)

# Interactive plots using Plotly
# Positive Ratios Histogram
fig = px.histogram(ratios_df, x='positive_ratio', nbins=30, title='Distribution of Positive Ratios',
                   labels={'positive_ratio': 'Positive Ratio (%)'}, color_discrete_sequence=['blue'])
fig.update_layout(xaxis_title='Positive Ratio (%)', yaxis_title='Count')
fig.show()

# Magnitude-Weighted Positive Ratios Histogram
fig = px.histogram(magnitude_weighted_ratios_df, x='magnitude_weighted_positive_ratio', nbins=30,
                   title='Distribution of Magnitude-Weighted Positive Ratios',
                   labels={'magnitude_weighted_positive_ratio': 'Magnitude-Weighted Positive Ratio (%)'},
                   color_discrete_sequence=['green'])
fig.update_layout(xaxis_title='Magnitude-Weighted Positive Ratio (%)', yaxis_title='Count')
fig.show()

#Volatility-Weighted Positive Ratios Histogram
fig = px.histogram(volatility_weighted_ratios_df, x='volatility_weighted_positive_ratio', nbins=30,
                   title='Distribution of Volatility-Weighted Positive Ratios',
                   labels={'volatility_weighted_positive_ratio': 'Volatility-Weighted Positive Ratio (%)'},
                   color_discrete_sequence=['orange'])
fig.update_layout(xaxis_title='Volatility-Weighted Positive Ratio (%)', yaxis_title='Count')
fig.show()

"""
#Line Plot for Positive Ratios Over Time
fig = go.Figure()
fig.add_trace(go.Scatter(x=ratios_df.index, y=ratios_df['positive_ratio'], mode='lines', name='Positive Ratio'))
fig.update_layout(title='Positive Ratios Over Time',
                  xaxis_title='Block Index', yaxis_title='Positive Ratio (%)')
fig.show()

#Compare Ratios Across Methods
fig = go.Figure()
fig.add_trace(go.Scatter(x=ratios_df.index, y=ratios_df['positive_ratio'], mode='lines', name='Unweighted'))
fig.add_trace(go.Scatter(x=magnitude_weighted_ratios_df.index, y=magnitude_weighted_ratios_df['magnitude_weighted_positive_ratio'],
                         mode='lines', name='Magnitude-Weighted'))
fig.add_trace(go.Scatter(x=volatility_weighted_ratios_df.index, y=volatility_weighted_ratios_df['volatility_weighted_positive_ratio'],
                         mode='lines', name='Volatility-Weighted'))
fig.update_layout(title='Comparison of Positive Ratios (Unweighted vs Weighted)',
                  xaxis_title='Block Index', yaxis_title='Positive Ratio (%)')
fig.show()
"""
