import streamlit as st
import pandas as pd
import joblib
import datetime
import requests
import gspread
from google.oauth2.service_account import Credentials

# -----------------------------
# Load trained model & encoder
# -----------------------------
model = joblib.load("iv_drip_model.pkl")
le = joblib.load("label_encoder.pkl")

# -----------------------------
# Google Sheets Auth from Streamlit secrets
# -----------------------------
creds_dict = {
    "type": st.secrets["GSHEET_TYPE"],
    "project_id": st.secrets["GSHEET_PROJECT_ID"],
    "private_key_id": st.secrets["GSHEET_PRIVATE_KEY_ID"],
    "private_key": st.secrets["GSHEET_PRIVATE_KEY"].replace('\\n', '\n'),
    "client_email": st.secrets["GSHEET_CLIENT_EMAIL"],
    "client_id": st.secrets["GSHEET_CLIENT_ID"],
    "auth_uri": st.secrets["GSHEET_AUTH_URI"],
    "token_uri": st.secrets["GSHEET_TOKEN_URI"],
    "auth_provider_x509_cert_url": st.secrets["GSHEET_AUTH_PROVIDER_X509_CERT_URL"],
    "client_x509_cert_url": st.secrets["GSHEET_CLIENT_X509_CERT_URL"]
}

scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
gc = gspread.authorize(credentials)
worksheet = gc.open("iv_drip_log").worksheet("Sheet1")  # adjust if needed

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
        input_df = pd.DataFrame([[med_code, dosage, weight, concentration]],
                                columns=["Med_Code", "Dosage (mcg/kg/min)", "Patient Weight (kg)", "Concentration (mcg/ml)"])

        # Predict
        predicted_rate = model.predict(input_df)[0]
        st.success(f"💧 Predicted Drip Rate: {predicted_rate:.2f} ml/hr")

        # Log data to Google Sheets
        log_row = [
            patient_name,
            medication,
            dosage,
            weight,
            concentration,
            round(predicted_rate, 2),
            datetime.datetime.now().isoformat()
        ]
        worksheet.append_row(log_row)
        st.info("✅ Logged to Google Sheets successfully.")

    except Exception as e:
        st.error("❌ Prediction or logging error:")
        st.code(str(e))
