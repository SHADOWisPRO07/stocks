# nifty_prediction/scripts/prediction_visualization.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), :]
        X.append(a)
        Y.append(data[i + time_step, 3])  # Predicting the 'Close' price
    return np.array(X), np.array(Y)

def visualize_predictions():
    # Load the data
    nifty_data = pd.read_csv("C:/Users/Shadow/Desktop/nifty_prediction/data/nifty_data_features.csv", index_col="Date", parse_dates=True)
    data = nifty_data.values

    # Define the time step
    time_step = 60

    # Create the dataset
    X, Y = create_dataset(data, time_step)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Load the model
    model = tf.keras.models.load_model("C:/Users/Shadow/Desktop/nifty_prediction/models/nifty_lstm_model.h5")

    # Make predictions
    predictions = model.predict(X_test)

    # Inverse transform the predictions to get actual values
    scaler = MinMaxScaler()
    scaler.fit(nifty_data)
    predictions = scaler.inverse_transform(predictions)

    # Plot the results
    plt.figure(figsize=(14, 5))
    plt.plot(nifty_data.index[-len(Y_test):], scaler.inverse_transform(Y_test.reshape(-1, 1)), color='blue', label='Actual')
    plt.plot(nifty_data.index[-len(predictions):], predictions, color='red', label='Predicted')
    plt.title('Nifty 50 Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    a=input('Enter')

if __name__ == "__main__":
    visualize_predictions()