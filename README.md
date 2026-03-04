# Credit Card Fraud Detector 

An anomaly detection model built with Keras and TensorFlow to identify fraudulent credit card transactions. This project utilizes an Autoencoder neural network architecture to learn the patterns of normal transactions and flags anomalies based on reconstruction errors.

## Technologies Used
* **Deep Learning:** TensorFlow / Keras (Sequential API)
* **Data Processing & Scaling:** Pandas, NumPy, Scikit-Learn (`StandardScaler`)
* **Architecture:** Autoencoder (Encoder-Decoder Neural Network)

## Highlights
* Custom thresholding for anomaly detection using Mean Squared Error (MSE).
* Model trained exclusively on non-fraudulent data to establish a baseline for normal behavior, isolating anomalies during the reconstruction phase.
* Includes validation splitting and comprehensive performance metrics like Mean Absolute Error (MAE).

## ⚠️ Important Note on the Dataset
The `creditcard.csv` dataset used for training this model is not included in this repository due to GitHub's file size limitations. 

**To run this project locally:**
1. Download the dataset from Kaggle (search for "Credit Card Fraud Detection").
2. Extract the `creditcard.csv` file.
3. Place it directly in the root directory of this project alongside the Python script.
4. Run the code.