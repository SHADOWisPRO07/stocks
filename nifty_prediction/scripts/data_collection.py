# nifty_prediction/scripts/data_collection.py
import yfinance as yf
import pandas as pd

def collect_data():
    # Fetch historical data for Nifty 50
    nifty_data = yf.download("^NSEI", start="2010-01-01", end="2023-01-01")
    nifty_data.to_csv("C:/Users/Shadow/Desktop/nifty_prediction/data/nifty_data.csv")

if __name__ == "__main__":
    collect_data()