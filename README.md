#  Car Price Prediction System

A professional Machine Learning web application built with **Flask** that predicts the market value of a car. The project features a robust data pipeline, custom feature engineering, and a clean, minimalist web interface.



##  Project Structure
Based on the current implementation, the directory is organized as follows:

CAR_PRICE_PREDICTION/
│
├── data/
│   └── car_price_dataset.csv       # Raw training data
│
├── model/
│   └── car_price_model.pkl         # Saved Scikit-Learn Pipeline
│
├── src/
│   ├── train.py                    # Script to train and save the model
│   └── predict.py                  # CLI script for testing predictions
│
├── app.py                          # Flask Web Application
├── EDA_Car_Price.ipynb             # Exploratory Data Analysis notebook
├── requirements.txt                # List of required Python libraries
└── README.md                       # Project documentation



##  Features & Engineering
- **Web Interface**: Clean and minimalist Flask-based UI built with Bootstrap for professional look and feel.
- **Advanced Feature Engineering**: 
    - `KM_per_Year`: Evaluates car usage intensity.
    - `Power_per_Seat`: Performance-to-size ratio.
    - `Is_New`: Boolean flag for cars under 3 years old.
- **Smart Pipeline**: Uses a `ColumnTransformer` with `OneHotEncoder` to handle categorical features like `Brand` and `Engine_Type` automatically.
- **Target Transformation**: Utilizes `np.log1p` during training to normalize price distribution and `np.expm1` during prediction for accurate output.

---

##  Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/Manasvi-Bauskar30/Car_Price_Prediction.git](https://github.com/Manasvi-Bauskar30/Car_Price_Prediction.git)
   cd CAR_PRICE_PREDICTION


2. **Install Dependencies:**
pip install -r requirements.txt

3. **Train the model (optional):**
python src/train.py




##  Usage 

1. **Web Application:**
python app.py

After running, open your browser and navigate to: http://127.0.0.1:5000

2. **Command Line Prediction:**
To test predictions directly in the terminal without the web interface: 
python src/predict.py 



##  Technologies Used

Backend: Python, Flask

Machine Learning: Scikit-learn (Random Forest Regressor)

Data Handling: Pandas, NumPy

Serialization: Joblib / Pickle

Frontend: HTML5, Bootstrap 5



##  Author 

Manasvi Bauskar
LinkedIn : https://www.linkedin.com/in/manasvi-bauskar-14b2b2201/
GitHub : https://www.google.com/search?q=https://github.com/Manasvi-Bauskar30


