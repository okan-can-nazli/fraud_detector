# Credit Card Fraud Anomaly Detector

## Project Overview
An unsupervised anomaly detection system built with Keras and TensorFlow. This project identifies fraudulent credit card transactions by learning the baseline behavior of normal transactions and flagging statistical deviations.

Instead of traditional binary classification, this model uses a Deep Autoencoder architecture. By training exclusively on valid transactions, the network learns to compress and reconstruct normal behavior. Fraudulent transactions fail to reconstruct correctly, resulting in a high Mean Squared Error (MSE) which acts as the trigger for anomaly detection.

## Tech Stack
* **Deep Learning:** TensorFlow / Keras
* **Data Processing:** Pandas, NumPy, Scikit-Learn (`StandardScaler`)
* **Architecture:** 7-Layer Deep Autoencoder (30 → 16 → 8 → 3 → 8 → 16 → 30)

## Key Features
* **Latent Space Compression:** Forces 30-dimensional transaction data through a 3-dimensional bottleneck to capture core behavioral patterns.
* **Semi-Supervised Approach:** Trained strictly on non-fraudulent data with a 20% validation split to prevent data leakage and overfitting.
* **Custom Thresholding:** Evaluates reconstruction errors mathematically to flag fraud while minimizing false positives.

## Note on Dataset
The model is designed around the standard `creditcard.csv` dataset (~284,000 transactions). Due to GitHub size constraints, the CSV is not included. To run locally, place the Kaggle dataset in the root directory before execution.
