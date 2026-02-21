import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'car_price_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'car_price_model.pkl')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'model', 'preprocessor.pkl')

# ---------------- LOAD DATA ----------------
data = pd.read_csv(DATA_PATH)
print("Dataset Loaded Successfully!")

# ---------------- BASIC CLEANING ----------------
# remove duplicates
data = data.drop_duplicates()

# remove missing values
data = data.dropna()

# ---------------- OUTLIER REMOVAL ----------------
data = data[data['Price_INR'] < data['Price_INR'].quantile(0.95)]
data = data[data['Mileage_KM'] < data['Mileage_KM'].quantile(0.99)]
data = data[data['Power_BHP'] < data['Power_BHP'].quantile(0.99)]

# ---------------- FEATURE ENGINEERING ----------------
# Important engineered features
data['KM_per_Year'] = data['Mileage_KM'] / (data['Age_Years'] + 1)
data['Power_per_Seat'] = data['Power_BHP'] / data['Seats']
data['Is_New'] = data['Age_Years'].apply(lambda x: 1 if x <= 3 else 0)

# ---------------- FEATURES & TARGET ----------------
X = data.drop('Price_INR', axis=1)

# log transform target (very important)
y = np.log1p(data['Price_INR'])

# categorical and numeric columns
categorical_cols = ['Brand', 'Engine_Type']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# ---------------- PREPROCESSOR ----------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

# ---------------- MODEL ----------------
model = RandomForestRegressor(
    n_estimators=1000,
    max_depth=25,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

# create pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- TRAIN ----------------
pipeline.fit(X_train, y_train)

# ---------------- PREDICT ----------------
y_pred = pipeline.predict(X_test)

# convert back from log
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred)

# ---------------- ACCURACY ----------------
r2 = r2_score(y_test_actual, y_pred_actual)
print("Final Model Accuracy (R2 Score):", round(r2 * 100, 2), "%")

# ---------------- SAVE MODEL ----------------
joblib.dump(pipeline, MODEL_PATH)
print("Model Saved Successfully!")