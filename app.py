import streamlit as st
import pandas as pd
import joblib 

# Load the trained model
model = joblib.load("heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Creating the UI
st.title("Heart Disease Risk Prediction")
st.markdown("Enter patient information below")

age = st.slider("Age", 18, 100, 40) # min, max, range
Gender = st.selectbox("Gender", ['male', 'female'])

chest_pain = st.selectbox("Chest Pain", ['typical', 'atypical', 'non-anginal', 'asymptomatic'])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholestrol = st.number_input("Cholestrol (mm Hg)", 100, 600, 200)
fasting_bp = st.selectbox("Fasting Blood Pressure > 120 mg/dl", [0,1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "left ventricular hypertrophy"])

max_heart_rate = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

st_slope = st.selectbox("ST Slope", ["upsloping", "flat", "downsloping"])

if st.button("Predict"):
    # creating raw input data
    input_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholestrol,
        "MaxHR": max_heart_rate,
        "Oldpeak": oldpeak,
        "FastingBS": fasting_bp,
        "Gender_" + Gender: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + str(exercise_angina): 1,
        "ST_Slope_" + st_slope: 1
    }
    
    # for predicting we need everything as dataframe
    input_df = pd.DataFrame([input_data])

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
        
    input_df = input_df[columns] # data frame is ready


    # scaling the data
    scaled_input_data = scaler.transform(input_df)
    prediction = model.predict(scaled_input_data)[0]


    if prediction == 1:
        st.error("⚠️ High risk of heart stroke. Please consult a doctor immediately.")
    else:
        st.success("✅ Low risk of heart stroke. Keep maintaining a healthy lifestyle!")
