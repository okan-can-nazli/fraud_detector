# Credit Card Fraud Detector 🕵️‍♂️

An anomaly detection model built with Keras and TensorFlow to identify fraudulent credit card transactions. This project utilizes an Autoencoder neural network architecture to learn the patterns of normal transactions and flags anomalies based on reconstruction errors.

## Technologies Used
* **Deep Learning:** TensorFlow / Keras (Sequential API)
* **Data Processing & Scaling:** Pandas, NumPy, Scikit-Learn (`StandardScaler`)
* **Architecture:** Autoencoder (Encoder-Decoder Neural Network)

## Highlights
* Custom thresholding for anomaly detection using Mean Squared Error (MSE).
* Model trained exclusively on non-fraudulent data to establish a baseline for normal behavior, isolating anomalies during the reconstruction phase.
* Includes validation splitting and comprehensive performance metrics like Mean Absolute Error (MAE).