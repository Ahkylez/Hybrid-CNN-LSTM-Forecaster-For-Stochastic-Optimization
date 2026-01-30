from typing import Union
import pandas as pd
import yfinance as yf
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

def create_dataloaders(df, config):
    df_train, df_val, df_test = split_data(
        df, 
        config['data']['train_split'], 
        config['data']['val_split']
    )
    
    scaler = MinMaxScaler()
    scaler.fit(df_train) # Important: Fit ONLY on training data
    
    train_s = scaler.transform(df_train)
    val_s = scaler.transform(df_val)
    test_s = scaler.transform(df_test) # We just transform test, never fit

    window_size = config['model']['lookback_window']
    batch_size = config['model']['batch_size'] # Get batch size from config too
    
    X_train, y_train = create_sliding_windows(train_s, window_size)
    X_train_t = torch.FloatTensor(X_train).transpose(1, 2)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    X_val, y_val = create_sliding_windows(val_s, window_size)
    X_val_t = torch.FloatTensor(X_val).transpose(1, 2)
    y_val_t = torch.FloatTensor(y_val).view(-1, 1)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    X_test, y_test = create_sliding_windows(test_s, window_size)
    
    X_test_t = torch.FloatTensor(X_test).transpose(1, 2)
    y_test_t = torch.FloatTensor(y_test).view(-1, 1)
    
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler


def _add_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Access the 'features' section of your loaded yaml
    indicators_cfg = config['features'] 
    
    # --- 1. RSI (Relative Strength Index) ---
    delta = df['Close'].diff()
    # Note: yaml keys are lowercase (rsi_period vs RSI_PERIOD)
    gain = (delta.where(delta > 0, 0)).ewm(span=indicators_cfg['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=indicators_cfg['rsi_period']).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- 2. MACD ---
    exp1 = df['Close'].ewm(span=indicators_cfg['macd_fast']).mean()
    exp2 = df['Close'].ewm(span=indicators_cfg['macd_slow']).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=indicators_cfg['macd_signal']).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # --- 3. KDJ Indicator ---
    low_n = df['Low'].rolling(window=indicators_cfg['kdj_n']).min()
    high_n = df['High'].rolling(window=indicators_cfg['kdj_n']).max()
    
    rsv = 100 * ((df['Close'] - low_n) / (high_n - low_n))
    
    df['KDJ_K'] = rsv.ewm(com=indicators_cfg['kdj_m1']-1).mean()
    df['KDJ_D'] = df['KDJ_K'].ewm(com=indicators_cfg['kdj_m2']-1).mean()
    df['KDJ_J'] = 3 * df['KDJ_K'] - 2 * df['KDJ_D']

    # --- 4. Bollinger Bands ---
    sma_bb = df['Close'].rolling(window=indicators_cfg['bb_period']).mean()
    std_bb = df['Close'].rolling(window=indicators_cfg['bb_period']).std()
    df['BB_Upper'] = sma_bb + (indicators_cfg['bb_std'] * std_bb)
    df['BB_Lower'] = sma_bb - (indicators_cfg['bb_std'] * std_bb)

    # --- 5. Simple Moving Averages ---
    for window in indicators_cfg['sma_windows']:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()

    # --- 6. Log Returns ---
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    return df.dropna()



# Make the data of each stock a clean timeseries for the model
def _clean_ticker_data(df: pd.DataFrame) -> pd.DataFrame:
    # Name code is not working because I swithced from download to history
    # will have to move name through functions but i really dont feel like fixing that rn 
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
    existing_cols = [c for c in ohlcv if c in df.columns]
    print(f'updated name: {df.columns.name}')
    df.index = pd.to_datetime(df.index)
    return df[existing_cols]

def download_ticker_data(tickers: Union[list, str], config):
    # Make it work if user inputs string
    if isinstance(tickers, str):
        tickers = [tickers]
    # ---- Improve data pipline later
    for ticker in tickers:
        t = yf.Ticker(ticker)
        
        df = t.history(interval=config['data']['interval'], period=config['data']['period'])
        if df.empty:
            continue

        info = t.info
        fundamentals = {
            'Market_Cap': info.get('marketCap'),
            'PB_Ratio': info.get('priceToBook'),
            'PS_Ratio': info.get('priceToSalesTrailing12Months')
        }
        df.columns.name = ticker

        df = _clean_ticker_data(df)

        for key, value in fundamentals.items():
            df[key] = value # This broadcasts the single value to all rows


        df = _add_features(df, config)

        root = Path(__file__).parent.parent
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        df.to_parquet(f'{data_dir}/{ticker}.parquet')



def get_ticker_data(ticker: str, config):
    ticker = ticker.upper()

    root = Path(__file__).parent.parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / f"{ticker}.parquet"
    try:
        # Try to check if we already have the data
        df = pd.read_parquet(file_path)
        return df
    except (FileNotFoundError, Exception):
        # If file doesn't exist download it
        print(f"Data for {ticker} not found. Downloading...")
        download_ticker_data(ticker, config)
        
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            print(f"Failed to retrieve data for {ticker}: {e}")
            return None
        
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15 # Used to fine tune hyper parameters
TICKER = 'amzn'

def split_data(df: pd.DataFrame, train_split: float = 0.70, val_split: float = 0.15):
    """
    Splits the data into a training list, validation list, and a testing list. Given the train split percentage, validation split percentage, and producing a test split for the rest.

    :param df: The data
    :type df: pd.DataFrame
    :param train_split: The percent of data to be training split
    :type train_split: float
    :param val_split: The percent of data ot be the valdidation split
    :type val_split: float
    """
    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test



def create_sliding_windows(data, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        # Grab 5 days of data
        window = data[i : i + window_size] 
        # Target is the closing price (usually index 3 or 4) of the next day
        target = data[i + window_size, 3] # Adjust '3' to match your 'Close' column index
        
        X.append(window)
        y.append(target)
        
    return np.array(X), np.array(y)