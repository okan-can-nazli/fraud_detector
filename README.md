# Gold Price Predictor (PyTorch) 📈

A time-series forecasting model built with PyTorch to predict future gold prices. This project automates the retrieval of real-world financial data, processes it using a sliding window approach, and utilizes a Long Short-Term Memory (LSTM) neural network to forecast future values based on historical trends.

## Features
* **Automated Data Pipeline:** Fetches up-to-date real historical gold data directly from Yahoo Finance.
* **Custom PyTorch Architecture:** Implements a custom `nn.Module` with an LSTM layer and a linear output layer for sequence prediction.
* **Sliding Window Processing:** Dynamically breaks down continuous time-series data into sliding sequences to effectively train the model on past trends.
* **Interactive Forecasting:** Includes a CLI tool that allows users to input any future date and calculates the predicted gold price step-by-step up to that specific day.

## Technologies Used
* **Deep Learning:** Python, PyTorch (`nn.LSTM`, `optim.Adam`, `MSELoss`)
* **Data Processing & Scaling:** Scikit-Learn (`MinMaxScaler`), NumPy, Pandas
* **Data Extraction:** `yfinance`

## How to Run

1. Make sure you have the required libraries installed (`torch`, `yfinance`, `scikit-learn`, `numpy`, `pandas`).
2. Run the main script to fetch the latest data, train the model, and start the prediction interface:
   ```bash
   python main.py