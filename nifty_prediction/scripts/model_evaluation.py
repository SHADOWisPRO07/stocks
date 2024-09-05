# nifty_prediction/scripts/model_evaluation.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), :]
        X.append(a)
        Y.append(data[i + time_step, 3])  # Predicting the 'Close' price
    return np.array(X), np.array(Y)

def evaluate_model():
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

    # Evaluate the model
    loss = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {loss}")

if __name__ == "__main__":
    evaluate_model()