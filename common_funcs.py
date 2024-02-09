import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import talib

def ColorPrint(input, color):
    """
    The function ColorPrint takes an input string and a color as arguments, and returns the input string
    printed in the specified color.
    
    :param input: The input parameter is the text that you want to print in color
    :param color: The `color` parameter is a string that represents the color you want to use for
    printing the `input` text
    :return: the input string with the specified color applied to it.
    """
    ColorsDict = {
    'Red' : '\033[91m',
    'Green' : '\033[92m',
    'Yellow' : '\033[93m',
    'Blue' : '\033[94m',
    'Magenta' : '\033[95m',
    'Cyan' : '\033[96m',
    'Green' : "\033[92m",
    'reset' : "\033[0m"
    }
    return f'{ColorsDict[color]}{input}{ColorsDict["reset"]}'

def sliding_window(df, window_size):

    labels = pd.DataFrame(df.pop("Response"))
    
    X, y, open_prices = [], [], []
    l, r = window_size, window_size * 2
    with tqdm(total=len(df) - window_size * 2, desc= ColorPrint("Applying Sliding Window", "Blue")) as pbar:
        while r < len(df):
            X.append(df.iloc[l:r])
            y.append(labels['Response'].iloc[r - 1])
            open_prices.append(df['open'].iloc[r - 1])
            l += 1
            r += 1
            
            # Update the loading bar
            pbar.update(1)
    return np.array(X), np.array(y), np.array(open_prices)

def load_csv(file_path):
    # Get the total number of lines in the CSV file
    with open(file_path, 'rb') as file:
        total_lines = sum(1 for _ in file)

    # Initialize the tqdm loading bar
    with tqdm(total=total_lines, desc= ColorPrint(f"Loading {file_path.split('/')[-1]}", "Blue")) as pbar:
        # Use pandas to read the CSV file
        df = pd.read_csv(file_path).set_index('timestamp')
        pbar.update(total_lines)  # Update the loading bar to completion

    return df

def collect_new_data(start_date =  datetime(2023,9,1), save_file = False, symbol = 'BTC'):
    ### ESTABLISH CLIENT ###
    client = CryptoHistoricalDataClient()
    ### END OF SECTION ###

    bars_params = CryptoBarsRequest(
                        symbol_or_symbols=[f"{symbol}/USD"],
                        timeframe=TimeFrame.Minute,
                        start = start_date
                    )
    symbol = bars_params.symbol_or_symbols[0]
        
    print(f"----COLLECTING DATA FOR {symbol}----")

    bars = client.get_crypto_bars(bars_params)
    bars = bars.df.reset_index().drop(['symbol', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap'], axis = 1)

    df = pd.DataFrame(columns = ['timestamp'])
    timestamp_list = pd.date_range(start=bars['timestamp'].iloc[0], end=bars['timestamp'].iloc[-1], freq='1T')
    df['timestamp'] = timestamp_list
    df = df.merge(bars, on = 'timestamp', how = 'left')
    df['open'] = df['open'].fillna(method = 'ffill')
    for i in tqdm(range(2,61,2)):
        df[f"SMA_{i}"] = talib.SMA(df['open'], timeperiod = i)
        df[f"EMA_{i}"] = talib.EMA(df['open'], timeperiod = i)
        df[f"RSI_{i}"] = talib.RSI(df["open"], timeperiod = i)
    MACD, Signal, Histo = talib.MACD(df['open'])
    df["MACD"] = MACD
    df["Signal"] = Signal
    df["Histo"] = Histo
    df = df.set_index("timestamp")
    response = [1 if ((df['SMA_8'].iloc[x]) < df['SMA_8'].iloc[x+1]) else 0 for x in tqdm(range(len(df)-1))]
    df = df[:-1]
    df['Response'] = response
    df = df.dropna()
    if save_file:
        file = df
        file.to_csv(f"trainingDS/{symbol}/alldata.csv", index = True)
    return df

def collect_new_open_data(start_date =  datetime(2023,9,1), save_file = False, symbol="BTC"):
    ### ESTABLISH CLIENT ###
    client = CryptoHistoricalDataClient()
    ### END OF SECTION ###

    bars_params = CryptoBarsRequest(
                        symbol_or_symbols=[f"{symbol}/USD"],
                        timeframe=TimeFrame.Minute,
                        start = start_date
                    )
    symbol_name = bars_params.symbol_or_symbols[0]
        
    print(f"----COLLECTING DATA FOR {symbol_name}----")

    bars = client.get_crypto_bars(bars_params)
    bars = bars.df.reset_index().drop(['symbol', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap'], axis = 1)

    df = pd.DataFrame(columns = ['timestamp'])
    timestamp_list = pd.date_range(start=bars['timestamp'].iloc[0], end=bars['timestamp'].iloc[-1], freq='1T')
    df['timestamp'] = timestamp_list
    df = df.merge(bars, on = 'timestamp', how = 'left')
    df['open'] = df['open'].fillna(method = 'ffill')
    if save_file:
        file = df
        file.to_csv(f"trainingDS/{symbol}/alldata.csv", index = True)
    
    return df

def add_features(df):
    timestamp_list = pd.date_range(start=df['timestamp'].iloc[0], end=df['timestamp'].iloc[-1], freq='1T')
    new_df = pd.DataFrame(columns = ['timestamp'])
    new_df['timestamp'] = timestamp_list
    new_df = new_df.merge(df, on = 'timestamp', how = 'left')
    new_df['open'] = new_df['open'].fillna(method = 'ffill')
    for i in tqdm(range(2,61,2)):
        new_df[f"SMA_{i}"] = talib.SMA(new_df['open'], timeperiod = i)
        new_df[f"EMA_{i}"] = talib.EMA(new_df['open'], timeperiod = i)
        new_df[f"RSI_{i}"] = talib.RSI(new_df["open"], timeperiod = i)
    MACD, Signal, Histo = talib.MACD(new_df['open'])
    new_df["MACD"] = MACD
    new_df["Signal"] = Signal
    new_df["Histo"] = Histo
    new_df = new_df.set_index("timestamp")
    response = [1 if ((new_df['SMA_8'].iloc[x]) < new_df['SMA_8'].iloc[x+1]) else 0 for x in tqdm(range(len(new_df)-1))]
    new_df = new_df[:-1]
    new_df['Response'] = response
    df = new_df.dropna()

    return df