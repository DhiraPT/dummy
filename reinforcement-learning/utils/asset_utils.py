import pandas as pd
import talib as ta

from custom_types.asset_type import Asset


def preprocess_data(file_path: str, timeframe: int) -> pd.DataFrame:
    df = pd.read_csv(file_path, header=None)
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('timestamp')

    all_timestamps = pd.date_range(start=df.index.min(), end=df.index.max(), freq=f'{timeframe}min')
    df = df.reindex(all_timestamps)

    # # Fill missing values for OHLC with the previous close value
    df['close'] = df['close'].ffill()
    df['open'] = df['open'].fillna(df['close'])
    df['high'] = df['high'].fillna(df['close'])
    df['low'] = df['low'].fillna(df['close'])

    # Fill missing values for volume and trades with 0
    df['volume'] = df['volume'].fillna(0)
    df['trades'] = df['trades'].fillna(0)

    df['rsi'] = ta.RSI(df['close'].values, timeperiod=14)
    df['ema'] = ta.EMA(df['close'].values, timeperiod=20)
    df['macd'], df['macdsignal'], df['macdhist'] = ta.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    df['hammer'] = ta.CDLHAMMER(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
    df['doji'] = ta.CDLDOJI(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
    df['upside_gap_two_crows'] = ta.CDLUPSIDEGAP2CROWS(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
    df['breakaway'] = ta.CDLBREAKAWAY(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
    df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = ta.BBANDS(df['close'].values, timeperiod=10, nbdevup=2, nbdevdn=2, matype=0)
    df['mom'] = ta.MOM(df['close'].values, timeperiod=10)
    df['willr'] = ta.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)

    df = df.ffill()

    return df


# def create_asset(name: str, min1: pd.DataFrame, min5: pd.DataFrame, min15: pd.DataFrame,
#                  min30: pd.DataFrame, min60: pd.DataFrame) -> Asset:
#     return {
#         'name': name,
#         'data': {
#             '1min': min1,
#             '5min': min5,
#             '15min': min15,
#             '30min': min30,
#             '60min': min60
#         }
#     }