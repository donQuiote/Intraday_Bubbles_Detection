#%%
import pandas as pd

#%%
df = pd.read_parquet("2010-05-03-ETN-trade.parquet copy")
print(df.head())
