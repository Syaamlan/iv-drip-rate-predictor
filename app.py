import streamlit as st
import joblib
import pandas as pd

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

