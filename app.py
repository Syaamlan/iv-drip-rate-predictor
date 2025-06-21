import streamlit as st
import pandas as pd
import joblib
import datetime
import requests

# -----------------------------
# Load model and label encoder
# -----------------------------
try:
    model = joblib.load("iv_drip_model.pkl")
    le = joblib.load("label_encoder.pkl")
except Exception as e:
    st.error("❌ Failed to load model or label encoder.")
    st.stop()

# -----------------------------
# Webhook URL (Google Apps Script)
# -----------------------------
WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbxV4ApISXns96-lDvHjj7-XBiorA24gxkoH7uuOkvwOYzwltoFRROfEa8i07ezu2a1siQ/exec"  # Replace if different

# -----------------------------
# Streamlit Interface
# -----------------------------
st.title("💧 IV Drip Rate Predictor")

# Inputs
patient_name = st.text_input("Patient Name")
medication = st.selectbox("Select Medication", le.classes_)
dosage = st.number_input("Dosage (mcg/kg/min)", min_value=0.0, step=0.1)
weight = st.number_input("Patient Weight (kg)", min_value=1.0, step=0.5)
concentration = st.number_input("Concentration (mcg/ml)", min_value=1.0, step=10.0)

if st.button("Predict Drip Rate"):
    try:
        med_code = le.transform([medication])[0]

        input_df = pd.DataFrame([[med_code, dosage, weight, concentration]],
                                columns=["Med_Code", "Dosage (mcg/kg/min)", "Patient Weight (kg)", "Concentration (mcg/ml)"])

        predicted_rate = model.predict(input_df)[0]
        st.success(f"💧 Predicted Drip Rate: {predicted_rate:.2f} ml/hr")

        # Payload for Google Sheets
        payload = {
            "patient_name": patient_name,
            "medication": medication,
            "dosage": dosage,
            "weight": weight,
            "concentration": concentration,
            "predicted_rate": round(predicted_rate, 2),
            "timestamp": datetime.datetime.now().isoformat()
        }

        response = requests.post(WEBHOOK_URL, json=payload)

        if response.status_code == 200:
            st.info("✅ Logged to Google Sheets")
        else:
            st.error("❌ Logging failed. Check Apps Script permissions or URL.")
            st.code(response.text)

    except Exception as e:
        st.error("❌ Prediction or Logging Error")
        st.code(str(e))
