from flask import Flask, request
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, 'model', 'car_price_model.pkl'))
brand_encoder = joblib.load(os.path.join(BASE_DIR, 'model', 'brand_encoder.pkl'))
engine_encoder = joblib.load(os.path.join(BASE_DIR, 'model', 'engine_encoder.pkl'))

@app.route('/')
def home():
    return """
    <h2>Car Price Prediction</h2>
    <form action='/predict' method='post'>
        Brand: <input name='brand'><br><br>
        Engine Type: <input name='engine'><br><br>
        Age (years): <input name='age'><br><br>
        Mileage (km): <input name='mileage'><br><br>
        Power (BHP): <input name='power'><br><br>
        Seats: <input name='seats'><br><br>
        <input type='submit'>
    </form>
    """

@app.route('/predict', methods=['POST'])
def predict():
    brand = request.form['brand']
    engine = request.form['engine']
    age = float(request.form['age'])
    mileage = float(request.form['mileage'])
    power = float(request.form['power'])
    seats = float(request.form['seats'])

    brand = brand_encoder.transform([brand])[0]
    engine = engine_encoder.transform([engine])[0]

    features = np.array([[brand, engine, age, mileage, power, seats]])

    prediction = model.predict(features)

    return f"<h2>Estimated Car Price: ₹ {int(prediction[0])}</h2>"

if __name__ == "__main__":
    app.run(debug=True)
