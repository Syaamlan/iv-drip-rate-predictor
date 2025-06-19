import streamlit as st
import pandas as pd
import joblib
import datetime
import gspread
from google.oauth2.service_account import Credentials

# -------------------------------
# Load the trained model & encoder
# -------------------------------
model = joblib.load("iv_drip_model.pkl")
le = joblib.load("label_encoder.pkl")

# -------------------------------
# Google Sheets Setup
# -------------------------------
scope = ["https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive"]

creds = Credentials.from_service_account_file(
    "your-service-account.json",  # ← Replace with actual JSON filename
    scopes=scope
)

client = gspread.authorize(creds)
sheet = client.open("iv_drip_log").worksheet("Sheet1")  # Replace with your sheet name if different

# -------------------------------
# Streamlit App Interface
# -------------------------------
st.title("💧 IV Drip Rate Predictor")

patient_name = st.text_input("Patient Name")
medication = st.selectbox("Select Medication", le.classes_)
dosage = st.number_input("Dosage (mcg/kg/min)", min_value=0.0, step=0.1)
weight = st.number_input("Patient Weight (kg)", min_value=1.0, step=0.5)
concentration = st.number_input("Concentration (mcg/ml)", min_value=1.0, step=10.0)

if st.button("Predict Drip Rate"):
    try:
        # Encode medication
        med_code = le.transform([medication])[0]

        # Match feature names used during training
        X_new = pd.DataFrame([[med_code, dosage, weight, concentration]],
                             columns=["Med_Code", "Dosage (mcg/kg/min)", "Patient Weight (kg)", "Concentration (mcg/ml)"])

        # Predict
        predicted_rate = model.predict(X_new)[0]
        st.success(f"💧 Predicted Drip Rate: {predicted_rate:.2f} ml/hr")

        # Log to Google Sheets
        log_row = [
            patient_name,
            medication,
            dosage,
            weight,
            concentration,
            round(predicted_rate, 2),
            datetime.datetime.now().isoformat()
        ]

        sheet.append_row(log_row)
        st.info("✅ Prediction logged to Google Sheets.")

    except Exception as e:
        st.error(f"❌ Prediction or logging error:\n\n{e}")
