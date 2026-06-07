import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Load Model Files
# -----------------------------
try:
    model = joblib.load("heart.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# -----------------------------
# Title
# -----------------------------
st.title("❤️ Heart Disease Risk Predictor")
st.markdown(
    """
    Predict the likelihood of heart disease using patient health indicators.
    """
)

st.divider()

# -----------------------------
# Input Section
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)

    Gender = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["typical", "atypical", "non-anginal", "asymptomatic"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        80,
        200,
        120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dl)",
        100,
        600,
        200
    )

with col2:
    fasting_bp = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "left ventricular hypertrophy"]
    )

    max_heart_rate = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        [0, 1]
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        0.0,
        6.0,
        1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["upsloping", "flat", "downsloping"]
    )

st.divider()

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Risk", use_container_width=True):

    input_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "MaxHR": max_heart_rate,
        "Oldpeak": oldpeak,
        "FastingBS": fasting_bp,
        "Gender_" + Gender: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + str(exercise_angina): 1,
        "ST_Slope_" + st_slope: 1
    }

    input_df = pd.DataFrame([input_data])

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    # Probability Scores
    probability = model.predict_proba(scaled_input)[0]

    low_risk_prob = probability[0] * 100
    high_risk_prob = probability[1] * 100

    st.divider()

    st.subheader("📊 Prediction Results")

    if prediction == 1:

        st.error("⚠️ High Risk of Heart Disease")

        st.metric(
            label="Risk Score",
            value=f"{high_risk_prob:.2f}%"
        )

        st.progress(min(int(high_risk_prob), 100))

        st.markdown(
            f"""
            **Model Confidence:** {high_risk_prob:.2f}%

            The model indicates an elevated risk of heart disease.
            Consider consulting a healthcare professional for further evaluation.
            """
        )

    else:

        st.success("✅ Low Risk of Heart Disease")

        st.metric(
            label="Safety Score",
            value=f"{low_risk_prob:.2f}%"
        )

        st.progress(min(int(low_risk_prob), 100))

        st.markdown(
            f"""
            **Model Confidence:** {low_risk_prob:.2f}%

            The model indicates a lower risk of heart disease.
            Continue maintaining a healthy lifestyle.
            """
        )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Low Risk Probability",
            f"{low_risk_prob:.2f}%"
        )

    with c2:
        st.metric(
            "High Risk Probability",
            f"{high_risk_prob:.2f}%"
        )

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "This application is intended for educational purposes only and should not be used as a substitute for professional medical advice."
)