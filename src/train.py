import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# -------- PATH SETTINGS --------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'car_price_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'car_price_model.pkl')

# Load Dataset
data = pd.read_csv(DATA_PATH)
print("Dataset Loaded Successfully!")

# ---------- ENCODING (IMPORTANT) ----------
# Machine learning text nahi samajhta (BMW, Honda etc)
# Isliye numbers me convert karenge

brand_encoder = LabelEncoder()
engine_encoder = LabelEncoder()

data['Brand'] = brand_encoder.fit_transform(data['Brand'])
data['Engine_Type'] = engine_encoder.fit_transform(data['Engine_Type'])

# Features and Target
X = data.drop('Price_INR', axis=1)
y = data['Price_INR']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Accuracy
score = model.score(X_test, y_test)
print("Model Accuracy (R2 Score):", round(score*100,2), "%")

# Save model + encoders
joblib.dump(model, MODEL_PATH)
joblib.dump(brand_encoder, os.path.join(BASE_DIR, 'model', 'brand_encoder.pkl'))
joblib.dump(engine_encoder, os.path.join(BASE_DIR, 'model', 'engine_encoder.pkl'))

print("Model Saved Successfully!")

