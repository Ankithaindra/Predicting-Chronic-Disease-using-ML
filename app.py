from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model # type: ignore

app = Flask(__name__)

# Load pre-trained models and scalers
diabetes_model_path = 'C:/Users/sneha/major project/model1/cnn_model.h5'
loaded_diabetes_model = load_model(diabetes_model_path)

# Kidney disease model and scaler paths
kidney_model_path = 'C:/Users/sneha/major project/model_save/kidney_xgb_model.pkl'
kidney_scaler_path = 'C:/Users/sneha/major project/model_save/kidney_scaler.pkl'
with open(kidney_model_path, 'rb') as f:
    kidney_model = pickle.load(f)
with open(kidney_scaler_path, 'rb') as f:
    kidney_scaler = pickle.load(f)

# Heart disease model and scaler paths
heart_model_path = 'C:/Users/sneha/major project/model_save/heart_gbc_model.pkl'
heart_scaler_path = 'C:/Users/sneha/major project/model_save/heart_scaler.pkl'
with open(heart_model_path, 'rb') as f:
    heart_model = pickle.load(f)
with open(heart_scaler_path, 'rb') as f:
    heart_scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def predict_diabetes():
    if request.method == 'POST':
        # Retrieve user input for diabetes prediction
        Pregnancies = int(request.form['pregnancies'])
        Glucose = float(request.form['glucose'])
        BloodPressure = float(request.form['bloodpressure'])
        SkinThickness = float(request.form['skinthickness'])
        Insulin = float(request.form['insulin'])
        BMI = float(request.form['bmi'])
        DiabetesPedigreeFunction = float(request.form['dpf'])
        Age = int(request.form['age'])

        

        # Prepare the feature set
        user_input = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1, -1)

        # Reshape for CNN and predict
        user_input = user_input.reshape((1, 8, 1))
        prediction = loaded_diabetes_model.predict(user_input)
        result = (prediction[0] > 0.5).astype(int)
        diagnosis = "The person is predicted to have diabetes." if result[0] == 1 else "The person is predicted not to have diabetes."
 # Recommendations for diabetes
        recommendations = {
            1: "1. Follow a low-carb diet.\n2. Monitor your blood sugar levels regularly.\n3. Engage in daily physical exercise.\n4. Consider insulin therapy based on doctor advice.",
            0: "1. Maintain a healthy weight.\n2. Follow a balanced diet rich in fruits, vegetables, and whole grains.\n3. Regular exercise to prevent diabetes."
        }

        return render_template('predict.html', prediction=diagnosis, recommendations=recommendations[result[0]], disease='Diabetes')

    return render_template('diabetes.html')

@app.route('/kidney', methods=['GET', 'POST'])
def predict_kidney():
    if request.method == 'POST':
        features = np.array([
            float(request.form['white_blood_cell_count']),
            float(request.form['blood_urea']),
            float(request.form['blood_glucose_random']),
            float(request.form['serum_creatinine']),
            float(request.form['packed_cell_volume']),
            float(request.form['albumin']),
            float(request.form['age']),
            float(request.form['haemoglobin']),
            float(request.form['sugar']),
            int(request.form['hypertension'])
        ]).reshape(1, -1)

        scaled_features = kidney_scaler.transform(features)
        prediction = kidney_model.predict(scaled_features)
        result = int(prediction[0])
        diagnosis = "The person has kidney disease." if result == 1 else "The person does not have kidney disease."

        # Recommendations for kidney disease
        recommendations = {
            1: "1. Reduce salt intake.\n2. Limit protein in your diet.\n3. Stay hydrated but avoid excessive water intake.\n4. Regular kidney function monitoring and medical consultations.",
            0: "1. Maintain a balanced diet.\n2. Regular check-ups to monitor kidney function.\n3. Stay hydrated."
        }

        return render_template('predict.html', prediction=diagnosis, recommendations=recommendations[result], disease='Kidney Disease')

    return render_template('kidney.html')

@app.route('/heart', methods=['GET', 'POST'])
def predict_heart():
    if request.method == 'POST':
        features = np.array([
            float(request.form['age']),
            int(request.form['sex']),
            int(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            int(request.form['fbs']),
            int(request.form['restecg']),
            float(request.form['thalach']),
            int(request.form['exang']),
            float(request.form['oldpeak']),
            int(request.form['slope']),
            int(request.form['ca']),
            int(request.form['thal'])
        ]).reshape(1, -1)

        scaled_features = heart_scaler.transform(features)
        prediction = heart_model.predict(scaled_features)
        result = int(prediction[0])
        diagnosis = "The person has heart disease." if result == 1 else "The person does not have heart disease."

        # Recommendations for heart disease
        recommendations = {
            1: "1. Adopt a heart-healthy diet rich in fruits and vegetables.\n2. Engage in regular cardiovascular exercise.\n3. Avoid smoking and excessive alcohol consumption.\n4. Regular blood pressure and cholesterol monitoring.",
            0: "1. Continue a healthy lifestyle to prevent heart disease.\n2. Keep track of blood pressure and cholesterol levels."
        }

        return render_template('predict.html', prediction=diagnosis, recommendations=recommendations[result], disease='Heart Disease')

    return render_template('heart.html')

if __name__ == '__main__':
    app.run(debug=True)

        