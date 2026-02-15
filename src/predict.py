import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'model', 'car_price_model.pkl'))
brand_encoder = joblib.load(os.path.join(BASE_DIR, 'model', 'brand_encoder.pkl'))
engine_encoder = joblib.load(os.path.join(BASE_DIR, 'model', 'engine_encoder.pkl'))

# Example Input
brand = "Honda"
engine = "Petrol"
age = 5
mileage = 40000
power = 120
seats = 5

# Convert to numbers
brand = brand_encoder.transform([brand])[0]
engine = engine_encoder.transform([engine])[0]

input_data = np.array([[brand, engine, age, mileage, power, seats]])

prediction = model.predict(input_data)

print("Estimated Car Price: ₹", int(prediction[0]))
