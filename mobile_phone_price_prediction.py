import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping


# ==============================
# 1. LOAD DATASET
# ==============================

data = pd.read_csv("mobile_phone_price_prediction_dataset.csv")

print("Dataset Shape:", data.shape)
print("\nFirst 5 Rows:")
print(data.head())


# ==============================
# 2. INPUT AND TARGET
# ==============================

X = data.drop("price_INR", axis=1)
y = data["price_INR"]

print("\nInput Features:")
print(X.columns)

print("\nTarget:")
print("price_INR")


# ==============================
# 3. TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=123
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==============================
# 4. SCALE INPUT FEATURES
# ==============================

X_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)


# ==============================
# 5. SCALE TARGET PRICE
# ==============================

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
)

y_test_scaled = y_scaler.transform(
    y_test.to_numpy().reshape(-1, 1)
)


# ==============================
# 6. BUILD ANN MODEL
# ==============================

model = Sequential()

model.add(Input(shape=(11,)))

model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))

model.add(Dense(1, activation="linear"))


# ==============================
# 7. COMPILE MODEL
# ==============================

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)


# ==============================
# 8. MODEL SUMMARY
# ==============================

print("\nANN MODEL SUMMARY:")
model.summary()


# ==============================
# 9. EARLY STOPPING
# ==============================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=30,
    restore_best_weights=True
)


# ==============================
# 10. TRAIN MODEL
# ==============================

history = model.fit(
    X_train,
    y_train_scaled,
    epochs=300,
    batch_size=32,
    validation_split=0.20,
    callbacks=[early_stop],
    verbose=1
)


# ==============================
# 11. PREDICTION
# ==============================

y_pred_scaled = model.predict(X_test)

y_pred = y_scaler.inverse_transform(
    y_pred_scaled
).flatten()


# ==============================
# 12. MODEL PERFORMANCE
# ==============================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n====================================")
print("MODEL PERFORMANCE")
print("====================================")

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)


# ==============================
# 13. ACTUAL VS PREDICTED
# ==============================

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Price:")
print(comparison.head(10))


# ==============================
# 14. TRAINING GRAPH
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("ANN Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()


# ==============================
# 15. ACTUAL VS PREDICTED GRAPH
# ==============================

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted Mobile Phone Price")

plt.grid(True)

plt.show()


# ==============================
# 16. SAVE MODEL
# ==============================

model.save("mobile_phone_price_model.keras")

print("\nModel saved successfully!")
# ==============================
# 17. MOBILE PRICE PREDICTION
# ==============================

print("\nEnter Mobile Phone Specifications:")

battery = float(input("Battery Capacity (mAh): "))
ram = float(input("RAM (GB): "))
storage = float(input("Storage (GB): "))
screen = float(input("Screen Size (inches): "))
camera = float(input("Camera (MP): "))
front_camera = float(input("Front Camera (MP): "))
cores = float(input("Processor Cores: "))
refresh_rate = float(input("Refresh Rate (Hz): "))
weight = float(input("Weight (g): "))
five_g = float(input("5G (1=Yes, 0=No): "))
fast_charging = float(input("Fast Charging (W): "))

mobile_data = np.array([[
    battery,
    ram,
    storage,
    screen,
    camera,
    front_camera,
    cores,
    refresh_rate,
    weight,
    five_g,
    fast_charging
]])

# Scale input
mobile_data_scaled = X_scaler.transform(mobile_data)

# Predict
predicted_price_scaled = model.predict(mobile_data_scaled)

# Convert back to original price
predicted_price = y_scaler.inverse_transform(
    predicted_price_scaled
)[0][0]

print("\n====================================")
print("PREDICTED MOBILE PHONE PRICE")
print("====================================")
print(f"Predicted Price: ₹{predicted_price:,.2f}")