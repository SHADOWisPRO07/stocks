# nifty_prediction/scripts/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data():
    # Load the data
    nifty_data = pd.read_csv("C:/Users/Shadow/Desktop/nifty_prediction/data/nifty_data.csv", index_col="Date", parse_dates=True)

    # Handle missing values
    nifty_data = nifty_data.fillna(method='ffill')

    # Normalize the data
    scaler = MinMaxScaler()
    nifty_data_scaled = scaler.fit_transform(nifty_data)

    # Convert back to DataFrame
    nifty_data_scaled = pd.DataFrame(nifty_data_scaled, columns=nifty_data.columns, index=nifty_data.index)
    nifty_data_scaled.to_csv("C:/Users/Shadow/Desktop/nifty_prediction/data/nifty_data_scaled.csv")

if __name__ == "__main__":
    preprocess_data()