import pandas as pd
from sklearn.preprocessing import StandardScaler

import keras

data = pd.read_csv("creditcard.csv")

fraud_data = data[data["Class"] == 1].drop(columns=["Class"])
clean_data = data[data["Class"] == 0].drop(columns=["Class"])



scaler = StandardScaler() # u have to create a new scaler to use on diffrent sets (obj not a method)


# fit on clean_data = you build the ruler based on normal transactions (mean=X, std=Y)
# transform on clean_data = measure clean data with that ruler
# transform on fraud_data = measure fraud data with the same ruler
clean_data_scaled = scaler.fit_transform(clean_data)  
fraud_data_scaled = scaler.transform(fraud_data)  

# model layer sequence
model = keras.Sequential([
    keras.layers.Input(shape=(30,)),
    keras.layers.Dense(16, activation="relu", name="encoder1"),
    keras.layers.Dense(8, activation="relu", name="encoder2"),
    keras.layers.Dense(3, activation="relu", name="encoder3"),
    keras.layers.Dense(8, activation="relu", name="decoder1"),
    keras.layers.Dense(16, activation="relu", name="decoder2"),
    keras.layers.Dense(30, name="decoder3")
])
