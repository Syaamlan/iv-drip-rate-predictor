import streamlit as st
import pandas as pd
import joblib
import datetime
import gspread
from google.oauth2.service_account import Credentials

# -----------------------------
# Load trained model & encoder
# -----------------------------
model = joblib.load("iv_drip_model.pkl")
le = joblib.load("label_encoder.pkl")

# -----------------------------
# Google Sheets Authentication
# -----------------------------
creds_dict = st.secrets["google_sheets"]
scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
client = gspread.authorize(creds)

# Open the sheet (make sure sheet name is correct)
sheet = client.open("iv_drip_log").worksheet("Sheet1")

# -----------------------------
# Streamlit Web Interface
# -----------------------------
st.title("💧 IV Drip Rate Predictor")

# Input fields
patient_name = st.text_input("Patient Name")
medication = st.selectbox("Select Medication", le.classes_)
dosage = st.number_input("Dosage (mcg/kg/min)", min_value=0.0, step=0.1)
weight = st.number_input("Patient Weight (kg)", min_value=1.0, step=0.5)
concentration = st.number_input("Concentration (mcg/ml)", min_value=1.0, step=10.0)

# Predict button
if st.button("Predict Drip Rate"):
    try:
        # Encode medication
        med_code = le.transform([medication])[0]

        # Prepare input
        input_df = pd.DataFrame([[medication, dosage, weight, concentration]],
            columns=["Medication", "Dosage (mcg/kg/min)", "Patient Weight (kg)", "Concentration (mcg/ml)"])

        # Predict
        predicted_rate = model.predict([[med_code, dosage, weight, concentration]])[0]
        st.success(f"💧 Predicted Drip Rate: {predicted_rate:.2f} ml/hr")

        # Log to Google Sheet
        row = [
            datetime.datetime.now().isoformat(),
            patient_name,
            medication,
            dosage,
            weight,
            concentration,
            round(predicted_rate, 2)
        ]
        sheet.append_row(row)
        st.info("✅ Prediction logged to Google Sheets")

    except Exception as e:
        st.error("❌ Prediction or logging error:")
        st.code(str(e))
