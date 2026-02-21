import pandas as pd
import numpy as np
import joblib
import os

# -------- PATH --------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'car_price_model.pkl')

# load trained pipeline

model = joblib.load(MODEL_PATH)

print("===== CAR PRICE PREDICTION =====")

# -------- USER INPUT --------

brand = input("Brand: ")
engine = input("Engine Type (Petrol/Diesel/CNG): ")
age = int(input("Car Age (years): "))
mileage = float(input("Mileage (KM driven): "))
power = float(input("Power (BHP): "))
seats = int(input("Seats: "))

# -------- FEATURE ENGINEERING (VERY IMPORTANT) --------

km_per_year = mileage / (age + 1)
power_per_seat = power / seats
is_new = 1 if age <= 3 else 0

# -------- CREATE DATAFRAME (THIS FIXES YOUR ERROR) --------

input_data = pd.DataFrame([{
"Brand": brand,
"Engine_Type": engine,
"Age_Years": age,
"Mileage_KM": mileage,
"Power_BHP": power,
"Seats": seats,
"KM_per_Year": km_per_year,
"Power_per_Seat": power_per_seat,
"Is_New": is_new
}])

# -------- PREDICTION --------

prediction = model.predict(input_data)

# reverse log transform

price = np.expm1(prediction[0])

print("\nEstimated Car Price: ₹", int(price))
