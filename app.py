import streamlit as st
import joblib
import pandas as pd

# Load the trained model and column names
model = joblib.load("drip_model.pkl")
columns = joblib.load("columns.pkl")

# Medication dropdown setup
medication_options = {
    "Adrenaline": 0,
    "Noradrenaline": 1,
    "Dobutamine": 2
}

st.set_page_config(page_title="IV Drip Rate Predictor", page_icon="💉")
st.title("💧 IV Drip Rate Predictor")

st.markdown("""
This app uses a trained machine learning model to predict the **drip rate (ml/hr)** for IV medication based on user inputs.
""")

# User input
medication = st.selectbox("Medication Type", list(medication_options.keys()))
dosage = st.number_input("Dosage (mcg/kg/min)", min_value=0.0, step=0.1)
weight = st.number_input("Patient Weight (kg)", min_value=0.0, step=0.5)
concentration = st.number_input("Concentration (mcg/ml)", min_value=0.0, step=1.0)

if st.button("Predict Drip Rate"):
    # Convert medication to numeric value
    med_value = medication_options[medication]
    
    # Prepare input data
    input_data = pd.DataFrame([[med_value, dosage, weight, concentration]], columns=columns)

    # Predict
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(f"💧 Predicted Drip Rate: **{prediction:.2f} ml/hr**")
