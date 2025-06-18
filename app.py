import streamlit as st
import joblib
import pandas as pd
import datetime
import firebase_admin
from firebase_admin import credentials, db

# --- Firebase: Load from secrets ---
firebase_keys = {
    "type": st.secrets["FIREBASE_TYPE"],
    "project_id": st.secrets["FIREBASE_PROJECT_ID"],
    "private_key_id": st.secrets["FIREBASE_PRIVATE_KEY_ID"],
    "private_key": st.secrets["FIREBASE_PRIVATE_KEY"].replace('\\n', '\n'),
    "client_email": st.secrets["FIREBASE_CLIENT_EMAIL"],
    "client_id": st.secrets["FIREBASE_CLIENT_ID"],
    "auth_uri": st.secrets["FIREBASE_AUTH_URI"],
    "token_uri": st.secrets["FIREBASE_TOKEN_URI"],
    "auth_provider_x509_cert_url": st.secrets["FIREBASE_AUTH_PROVIDER_X509_CERT_URL"],
    "client_x509_cert_url": st.secrets["FIREBASE_CLIENT_X509_CERT_URL"]
}

# Initialize Firebase (only once)
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_keys)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://' + firebase_keys["project_id"] + '.firebaseio.com'
    })


# --- Load the model and label encoder ---
try:
    model = joblib.load("iv_drip_model.pkl")
    le = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    st.error("Model or label encoder not found. Please make sure both 'iv_drip_model.pkl' and 'label_encoder.pkl' are in the repo.")
    st.stop()

st.title("💉 IV Drip Rate Predictor (Medication-Based)")

# --- User Inputs ---
st.subheader("Patient & Medication Info")

medication = st.selectbox("Select Medication", le.classes_)
dosage = st.number_input("Dosage (mcg/kg/min)", min_value=0.0, step=0.1)
weight = st.number_input("Patient Weight (kg)", min_value=1.0, step=0.5)
concentration = st.number_input("Concentration (mcg/ml)", min_value=1.0, step=50.0)

# --- Predict Button ---
if st.button("Predict Drip Rate"):
    # Encode medication
    med_code = le.transform([medication])[0]

    # Prepare input DataFrame
    input_data = pd.DataFrame([[med_code, dosage, weight, concentration]],
        columns=['Med_Code', 'Dosage (mcg/kg/min)', 'Patient Weight (kg)', 'Concentration (mcg/ml)'])

    # Make prediction
    predicted_rate = model.predict(input_data)[0]

    st.success(f"💧 Predicted Drip Rate: {predicted_rate:.2f} ml/hr")

