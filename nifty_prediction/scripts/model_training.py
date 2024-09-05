# nifty_prediction/scripts/model_training.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), :]
        X.append(a)
        Y.append(data[i + time_step, 3])  # Predicting the 'Close' price
    return np.array(X), np.array(Y)

def train_model():
    # Load the data
    nifty_data = pd.read_csv("C:/Users/Shadow/Desktop/nifty_prediction/data/nifty_data_features.csv", index_col="Date", parse_dates=True)
    data = nifty_data.values

    # Define the time step
    time_step = 60

    # Create the dataset
    X, Y = create_dataset(data, time_step)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, Y_train, batch_size=32, epochs=50, validation_data=(X_test, Y_test))

    # Save the model
    model.save("C:/Users/Shadow/Desktop/nifty_prediction/models/nifty_lstm_model.h5")

if __name__ == "__main__":
    train_model()