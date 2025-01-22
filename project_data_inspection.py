import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset
file_path = "data_SPX.csv"

# Charger les données
df = pd.read_csv(file_path, parse_dates=['xltime'])
df.set_index('xltime', inplace=True)

print(df.head(90000))
