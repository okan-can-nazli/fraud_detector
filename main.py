import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error

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


# model compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Adam is better verison (contains:momentum etc.) of Stochastic gradient descent.It is a standart (Using random data set from current data set to reduce derivative calculation stuff on loss function)
    loss=keras.losses.MeanSquaredError(), # we need usefull loss compute method for reconstructed number set
    metrics=[
        keras.metrics.MeanAbsoluteError() # we use metrics to calculate "real world error rate"
    ]
)

# model training
model.fit(
    x=clean_data_scaled,
    y=clean_data_scaled,
    batch_size=512, # non overlapping input window size
    epochs=100,
    verbose="auto",
    callbacks=None, # during training operations
    
    #use one of them for 
    validation_split=0.2, # do not use %20 of data on training instead of use them as validators (steal)
    validation_data=None, # enter a SEPERATED VALİDATİON DATA SET (no steal)
    
    #defaults 
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
)


# TESTİNG

clean_reconstructed = model.predict(clean_data_scaled)
fraud_reconstructed = model.predict(fraud_data_scaled)


clean_errors = np.mean((clean_data_scaled - clean_reconstructed)**2, axis=1)
fraud_errors = np.mean((fraud_data_scaled - fraud_reconstructed)**2, axis=1)

print(f"Clean error mean: {clean_errors.mean():.4f}")
print(f"Fraud error mean: {fraud_errors.mean():.4f}")
print(f"Clean error max: {clean_errors.max():.4f}")
print(f"Fraud error min: {fraud_errors.min():.4f}")


threshold = 5  

# count fraud catches
frauds_caught = (fraud_errors > threshold).sum()
total_frauds = len(fraud_errors)

print(f"Caught {frauds_caught}/{total_frauds} frauds")




