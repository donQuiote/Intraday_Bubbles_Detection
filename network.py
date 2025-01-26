import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def stock_strat_network(threshold = 0.6):
    # Load the CSV file
    file_path = 'data/optimum_strategy_tracker.csv'
    df = pd.read_csv(file_path)

    # Convert 'day' column to datetime
    df['day'] = pd.to_datetime(df['day'])
    df['year'] = df['day'].dt.year
    df['month'] = df['day'].dt.month

    stocks = df.columns[1:-2]  #delete dates

    #Separate for each year
    for year in df['year'].unique():
        # Get all months for the current year
        months = sorted(df[df['year'] == year]['month'].unique())

        # Determine grid size for the subplots
        n_months = len(months)
        n_rows = int(np.ceil(n_months / 3))  # 3 columns per row
        n_cols = min(3, n_months)           # At most 3 columns

        # Create a large figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5), dpi=300)
        axes = axes.flatten() if n_months > 1 else [axes]  # Flatten if multiple subplots

        for i, month in enumerate(months):
            # Filter data for the current month and year
            monthly_data = df[(df['year'] == year) & (df['month'] == month)]

            # Compute the correlation matrix
            correlation_matrix = monthly_data[stocks].corr()

            # Create the network graph
            G = nx.Graph()

            # Add nodes (stocks)
            for stock in stocks:
                G.add_node(stock)

            # Add edges based on correlation threshold
            for j, stock1 in enumerate(stocks):
                for k, stock2 in enumerate(stocks):
                    if j < k and correlation_matrix.loc[stock1, stock2] > threshold:
                        G.add_edge(stock1, stock2, weight=correlation_matrix.loc[stock1, stock2])

            # Plot the graph for this month
            ax = axes[i]
            ax.set_title(f"{year}-{month:02d}", fontsize=14)
            pos = nx.spring_layout(G, seed=42)  # Consistent layout
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, node_color='lightblue')
            nx.draw_networkx_edges(G, pos, ax=ax, width=2, alpha=0.5, edge_color='gray')
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
            ax.axis("off")

        # Turn off unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        # Adjust layout and save the figure
        fig.suptitle(f"Network of Stocks Based on Strategy Correlation - Year: {year}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space for the title
        plt.savefig(f"correlation_networks_{year}.png", dpi=300)
        plt.show()