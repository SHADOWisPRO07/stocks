# nifty_prediction/main.py
import os

def main():
    # Check if data collection is needed
    if not os.path.exists("C:/Users/Shadow/Desktop/nifty_prediction/data/nifty_data.csv"):
        os.system("python C:/Users/Shadow/Desktop/nifty_prediction/scripts/data_collection.py")
    else:
        print("Data collection skipped. Data already exists.")

    # Check if data preprocessing is needed
    if not os.path.exists("C:/Users/Shadow/Desktop/nifty_prediction/data/nifty_data_scaled.csv"):
        os.system("python C:/Users/Shadow/Desktop/nifty_prediction/scripts/data_preprocessing.py")
    else:
        print("Data preprocessing skipped. Scaled data already exists.")

    # Check if feature engineering is needed
    if not os.path.exists("C:/Users/Shadow/Desktop/nifty_prediction/data/nifty_data_features.csv"):
        os.system("python C:/Users/Shadow/Desktop/nifty_prediction/scripts/feature_engineering.py")
    else:
        print("Feature engineering skipped. Features already exist.")

    # Check if model training is needed
    if not os.path.exists("C:/Users/Shadow/Desktop/nifty_prediction/models/nifty_lstm_model.h5"):
        os.system("python C:/Users/Shadow/Desktop/nifty_prediction/scripts/model_training.py")
    else:
        print("Model training skipped. Model already exists.")

    # Always run model evaluation and prediction visualization
    os.system("python C:/Users/Shadow/Desktop/nifty_prediction/scripts/model_evaluation.py")
    os.system("python C:/Users/Shadow/Desktop/nifty_prediction/scripts/prediction_visualization.py")

if __name__ == "__main__":
    main()