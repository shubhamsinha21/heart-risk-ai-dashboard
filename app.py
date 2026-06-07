import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from io import BytesIO
from reportlab.pdfgen import canvas

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Heart Risk AI Dashboard",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# -----------------------------
# Header UI
# -----------------------------
st.markdown("""
    <div style="text-align:center">
        <h1>❤️ Heart Risk AI Dashboard</h1>
        <p style="color:gray;">Explainable AI-powered clinical decision support system</p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------
# INPUT UI
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    gender = st.selectbox("Gender", ["male", "female"])
    chest_pain = st.selectbox("Chest Pain Type",
        ["typical", "atypical", "non-anginal", "asymptomatic"])
    resting_bp = st.number_input("Resting BP", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    resting_ecg = st.selectbox("Resting ECG",
        ["Normal", "ST", "left ventricular hypertrophy"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Angina", [0, 1])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope",
        ["upsloping", "flat", "downsloping"])

st.divider()

# -----------------------------
# PDF FUNCTION (NEW)
# -----------------------------
def generate_pdf(data_dict):
    buffer = BytesIO()
    p = canvas.Canvas(buffer)

    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, 800, "Heart Disease Risk Report")

    p.setFont("Helvetica", 11)

    y = 760
    for key, value in data_dict.items():
        p.drawString(50, y, f"{key}: {value}")
        y -= 20

    p.save()
    buffer.seek(0)
    return buffer

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict Risk", use_container_width=True):

    input_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "FastingBS": fasting_bs,
        "Gender_" + gender: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + str(exercise_angina): 1,
        "ST_Slope_" + st_slope: 1
    }

    df = pd.DataFrame([input_data])

    for c in columns:
        if c not in df.columns:
            df[c] = 0

    df = df[columns]

    scaled = scaler.transform(df)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]

    low = prob[0] * 100
    high = prob[1] * 100

    st.divider()

    # -----------------------------
    # RESULT SECTION
    # -----------------------------
    st.subheader("📊 Risk Analysis")

    left, right = st.columns(2)

    with left:
        st.metric("Low Risk", f"{low:.2f}%")
        st.metric("High Risk", f"{high:.2f}%")

    with right:
        st.progress(int(high))
        st.write("Risk Probability Visualization")

    st.divider()

    # -----------------------------
    # CHART 1
    # -----------------------------
    st.subheader("📈 Risk Probability Chart")

    chart_data = pd.DataFrame({
        "Category": ["Low Risk", "High Risk"],
        "Probability": [low, high]
    })

    st.bar_chart(chart_data.set_index("Category"))

    st.divider()

    # -----------------------------
    # CHART 2
    # -----------------------------
    st.subheader("🧠 Key Health Indicators")

    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        importance = np.zeros(len(columns))

    feat_df = pd.DataFrame({
        "Feature": columns,
        "Impact": importance
    }).sort_values("Impact", ascending=False).head(8)

    st.bar_chart(feat_df.set_index("Feature"))

    st.divider()

    # -----------------------------
    # FINAL SUMMARY
    # -----------------------------
    st.success("Prediction completed using ML + probability + explainability")

    # -----------------------------
    # NEW: PDF DOWNLOAD SECTION
    # -----------------------------
    st.subheader("📄 Download Report")

    report_data = {
        "Age": age,
        "Gender": gender,
        "Chest Pain": chest_pain,
        "Resting BP": resting_bp,
        "Cholesterol": cholesterol,
        "Max Heart Rate": max_hr,
        "Oldpeak": oldpeak,
        "Prediction": "High Risk" if pred == 1 else "Low Risk",
        "High Risk %": f"{high:.2f}",
        "Low Risk %": f"{low:.2f}",
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    pdf_file = generate_pdf(report_data)

    st.download_button(
        label="📥 Download Medical Report (PDF)",
        data=pdf_file,
        file_name="heart_risk_report.pdf",
        mime="application/pdf"
    )

# -----------------------------
# FOOTER
# -----------------------------
st.caption("Made with ❤️ by Shubham Sinha | AI Engineer")