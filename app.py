import streamlit as st
import pandas as pd
import joblib
import datetime
import requests

# -----------------------------
# Load model and encoder
# -----------------------------
model = joblib.load("iv_drip_model.pkl")
le = joblib.load("label_encoder.pkl")

# -----------------------------
# Webhook URL from Google Apps Script
# -----------------------------
WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbxV4ApISXns96-lDvHjj7-XBiorA24gxkoH7uuOkvwOYzwltoFRROfEa8i07ezu2a1siQ/exec"  # ✅ Replace with your actual one if needed

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("💧 IV Drip Rate Predictor")

# Input fields
patient_name = st.text_input("Patient Name")
medication = st.selectbox("Select Medication", le.classes_)
dosage = st.number_input("Dosage (mcg/kg/min)", min_value=0.0, step=0.1)
weight = st.number_input("Patient Weight (kg)", min_value=1.0, step=0.5)
concentration = st.number_input("Concentration (mcg/ml)", min_value=1.0, step=50.0)

# Predict button
if st.button("Predict Drip Rate"):
    try:
        # Encode medication into numeric code
        med_code = le.transform([medication])[0]

        # Prepare input for prediction (match trained model's feature names!)
        X_new = pd.DataFrame([[med_code, dosage, weight, concentration]],
                             columns=["Med_Code", "Dosage (mcg/kg/min)", "Patient Weight (kg)", "Concentration (mcg/ml)"])
        
        # Make prediction
        predicted_rate = model.predict(X_new)[0]
        st.success(f"💧 Predicted Drip Rate: {predicted_rate:.2f} ml/hr")

        # Payload to send to Google Sheets via Apps Script webhook
        payload = {
            "patient_name": patient_name,
            "medication": medication,
            "dosage": dosage,
            "weight": weight,
            "concentration": concentration,
            "predicted_rate": round(predicted_rate, 2),
            "timestamp": datetime.datetime.now().isoformat()
        }

        # Send POST request to webhook
        response = requests.post(WEBHOOK_URL, json=payload)

        # Result handling
        if response.status_code == 200:
            st.info("✅ Prediction logged to Google Sheets")
        else:
            st.error("❌ Failed to log. Check your webhook URL.")
            st.code(response.text)

    except Exception as e:
        st.error("❌ Prediction or logging error:")
        st.code(str(e))
