import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


# ==============================
# LOAD TRAINED MODEL
# ==============================

model = tf.keras.models.load_model("mobile_phone_price_model.keras")


# ==============================
# LOAD DATASET FOR SCALER
# ==============================

import pandas as pd

data = pd.read_csv("mobile_phone_price_prediction_dataset.csv")

X = data.drop("price_INR", axis=1)
y = data["price_INR"]


# Create and fit input scaler
X_scaler = StandardScaler()
X_scaler.fit(X)


# Create and fit target scaler
y_scaler = StandardScaler()
y_scaler.fit(y.to_numpy().reshape(-1, 1))


# ==============================
# PREDICT FUNCTION
# ==============================

def predict_price():

    try:
        battery = float(battery_entry.get())
        ram = float(ram_entry.get())
        storage = float(storage_entry.get())
        screen = float(screen_entry.get())
        camera = float(camera_entry.get())
        front_camera = float(front_camera_entry.get())
        cores = float(cores_entry.get())
        refresh_rate = float(refresh_entry.get())
        weight = float(weight_entry.get())
        five_g = float(fiveg_entry.get())
        fast_charging = float(charging_entry.get())

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

        # Prediction
        predicted_scaled = model.predict(
            mobile_data_scaled,
            verbose=0
        )

        predicted_price = y_scaler.inverse_transform(
            predicted_scaled
        )[0][0]

        result_label.config(
            text=f"Predicted Price: ₹{predicted_price:,.2f}"
        )

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numbers in all fields."
        )


# ==============================
# GUI WINDOW
# ==============================

root = tk.Tk()

root.title("Mobile Phone Price Prediction")
root.geometry("600x700")
root.resizable(False, False)


# ==============================
# TITLE
# ==============================

title_label = tk.Label(
    root,
    text="Mobile Phone Price Prediction",
    font=("Arial", 22, "bold")
)

title_label.pack(pady=20)


# ==============================
# INPUT FRAME
# ==============================

frame = tk.Frame(root)
frame.pack(pady=10)


def create_input(label_text, row):

    label = tk.Label(
        frame,
        text=label_text,
        font=("Arial", 12)
    )

    label.grid(
        row=row,
        column=0,
        padx=10,
        pady=7,
        sticky="w"
    )

    entry = tk.Entry(
        frame,
        width=25,
        font=("Arial", 12)
    )

    entry.grid(
        row=row,
        column=1,
        padx=10,
        pady=7
    )

    return entry


battery_entry = create_input(
    "Battery Capacity (mAh)", 0
)

ram_entry = create_input(
    "RAM (GB)", 1
)

storage_entry = create_input(
    "Storage (GB)", 2
)

screen_entry = create_input(
    "Screen Size (inches)", 3
)

camera_entry = create_input(
    "Camera (MP)", 4
)

front_camera_entry = create_input(
    "Front Camera (MP)", 5
)

cores_entry = create_input(
    "Processor Cores", 6
)

refresh_entry = create_input(
    "Refresh Rate (Hz)", 7
)

weight_entry = create_input(
    "Weight (g)", 8
)

fiveg_entry = create_input(
    "5G (1=Yes, 0=No)", 9
)

charging_entry = create_input(
    "Fast Charging (W)", 10
)


# ==============================
# PREDICT BUTTON
# ==============================

predict_button = tk.Button(
    root,
    text="PREDICT PRICE",
    font=("Arial", 14, "bold"),
    command=predict_price,
    padx=20,
    pady=10
)

predict_button.pack(pady=20)


# ==============================
# RESULT
# ==============================

result_label = tk.Label(
    root,
    text="Predicted Price: ₹0.00",
    font=("Arial", 18, "bold")
)

result_label.pack(pady=10)


# ==============================
# RUN GUI
# ==============================

root.mainloop()