import pandas as pd


# Load data
df = pd.read_csv(r'data\SOLUSD_1.csv', header=None)
df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']

# Convert timestamp to datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df.set_index('datetime', inplace=True)

# Resampling function
def resample_candles(df, rule):
    resampled_df = df.loc[pd.to_datetime(1627776000, unit='s'):].resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'trades': 'sum'
    }).dropna()
    resampled_df['timestamp'] = (resampled_df.index.astype('int64') // 10**9)
    return resampled_df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']]

# Generate resampled candles
df_30min = resample_candles(df, '30min')

# Save to CSV
df_30min.to_csv('30min_candles.csv', index=False, header=False)