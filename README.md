# Car Price Prediction 🚗💰

## Overview
This project is a **Machine Learning model** that predicts car prices based on features such as brand, engine type, mileage, power, age, and number of seats. It uses **Python** and popular libraries like **scikit-learn**, **pandas**, and **pickle** for model serialization.  

---

## Project Structure
CAR_PRICE_PREDICTION/
│
├── data/
│ └── car_price_dataset.csv
│
├── model/
│ ├── brand_encoder.pkl
│ ├── engine_encoder.pkl
│ └── car_price_model.pkl
│
├── src/
│ ├── train.py
│ ├── predict.py
│ └── app.py
│
└── requirements.txt
|__ README.md


---

## Features
- Predict car prices using:
  - Brand
  - Car Age
  - Mileage
  - Engine Type
  - Power
  - Seats
- Encoders for categorical features
- Separate scripts for training and prediction
- Interactive app for live predictions (`app.py`)

---

## Installation
1. Clone the repository:
```bash
git clone <your-github-repo-url>
cd CAR_PRICE_PREDICTION

2. Install dependencies:
pip install -r requirements.txt


Usage
1. Train the Model
python src/train.py

2. Make Predictions
python src/predict.py

3. Run the Interactive App
python src/app.py


Enter car details interactively to get predicted prices.



Technologies Used
Python
Pandas
Scikit-learn
Pickle


Skills Highlight
Data Preprocessing & Feature Engineering
Categorical Encoding
Machine Learning Model Training & Evaluation
Model Serialization & Deployment
Python Scripting & Application Development



Author

Manasvi Bauskar 
LinkedIn - 
GitHub - 
