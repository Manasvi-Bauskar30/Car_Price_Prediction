from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'car_price_model.pkl')
model = joblib.load(MODEL_PATH)

# --- SIMPLE PROFESSIONAL UI ---
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Car Price Tool</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f4f4f4; padding-top: 50px; }
        .main-card { max-width: 600px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h2 { color: #333; margin-bottom: 25px; text-align: center; font-weight: 600; }
        .form-label { font-weight: 500; color: #555; }
        .btn-predict { width: 100%; background-color: #0056b3; color: white; border: none; padding: 10px; border-radius: 5px; font-size: 18px; }
        .btn-predict:hover { background-color: #004494; }
    </style>
</head>
<body>
    <div class="container">
        <div class="main-card">
            <h2>Car Price Predictor</h2>
            <form action="/predict" method="post">
                <div class="row">
                    <div class="col-md-6 mb-3">
                        <label class="form-label">Brand</label>
                        <input type="text" name="Brand" class="form-control" placeholder="Maruti, Honda..." required>
                    </div>
                    <div class="col-md-6 mb-3">
                        <label class="form-label">Engine Type</label>
                        <select name="Engine" class="form-select">
                            <option value="Petrol">Petrol</option>
                            <option value="Diesel">Diesel</option>
                            <option value="CNG">CNG</option>
                        </select>
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-6 mb-3">
                        <label class="form-label">Age (Years)</label>
                        <input type="number" name="Age" class="form-control" value="1" required>
                    </div>
                    <div class="col-md-6 mb-3">
                        <label class="form-label">Mileage (KM)</label>
                        <input type="number" name="Mileage" class="form-control" placeholder="20000" required>
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-6 mb-3">
                        <label class="form-label">Power (BHP)</label>
                        <input type="number" name="Power" class="form-control" value="80" required>
                    </div>
                    <div class="col-md-6 mb-3">
                        <label class="form-label">Seats</label>
                        <input type="number" name="Seats" class="form-control" value="5" required>
                    </div>
                </div>
                <button type="submit" class="btn btn-predict mt-3">Predict Price</button>
            </form>
        </div>
    </div>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Form data collect karna
        brand = request.form['Brand']
        engine = request.form['Engine']
        age = int(request.form['Age'])
        mileage = float(request.form['Mileage'])
        power = float(request.form['Power'])
        seats = int(request.form['Seats'])

        # Feature Engineering (Dataset ke columns ke hisab se)
        input_df = pd.DataFrame([{
            "Brand": brand,
            "Engine_Type": engine,
            "Age_Years": age,
            "Mileage_KM": mileage,
            "Power_BHP": power,
            "Seats": seats,
            "KM_per_Year": mileage / (age + 1),
            "Power_per_Seat": power / seats,
            "Is_New": 1 if age <= 3 else 0
        }])

        prediction = model.predict(input_df)
        price = np.expm1(prediction[0])

        return f'''
        <div style="text-align: center; padding: 50px; font-family: sans-serif;">
            <h3 style="color: #666;">Estimated Price</h3>
            <h1 style="color: #0056b3;">₹ {int(price):,}</h1>
            <br><a href="/" style="text-decoration: none; color: #0056b3;">← Back to Predictor</a>
        </div>
        '''
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)