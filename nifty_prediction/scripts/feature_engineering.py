# nifty_prediction/scripts/feature_engineering.py
import pandas as pd

def calculate_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_features():
    # Load the data
    nifty_data = pd.read_csv("C:/Users/Shadow/Desktop/nifty_prediction/data/nifty_data.csv", index_col="Date", parse_dates=True)

    # Calculate moving averages
    nifty_data['SMA_20'] = nifty_data['Close'].rolling(window=20).mean()
    nifty_data['SMA_50'] = nifty_data['Close'].rolling(window=50).mean()

    # Calculate RSI
    nifty_data['RSI_14'] = calculate_rsi(nifty_data['Close'], 14)

    # Calculate MACD
    nifty_data['EMA_12'] = nifty_data['Close'].ewm(span=12, adjust=False).mean()
    nifty_data['EMA_26'] = nifty_data['Close'].ewm(span=26, adjust=False).mean()
    nifty_data['MACD'] = nifty_data['EMA_12'] - nifty_data['EMA_26']
    nifty_data['Signal_Line'] = nifty_data['MACD'].ewm(span=9, adjust=False).mean()

    # Drop rows with NaN values
    nifty_data = nifty_data.dropna()

    # Save the features
    nifty_data.to_csv("C:/Users/Shadow/Desktop/nifty_prediction/data/nifty_data_features.csv")

if __name__ == "__main__":
    create_features()