import streamlit as st
import pandas as pd
import joblib
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# -----------------------------
# Load ML model and encoder
# -----------------------------
model = joblib.load("iv_drip_model.pkl")
le = joblib.load("label_encoder.pkl")

# -----------------------------
# Google Sheets Authentication
# -----------------------------
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("google-credentials.json", scope)
client = gspread.authorize(creds)

# Open the sheet (replace with your exact sheet name)
sheet = client.open("iv_drip_log").worksheet("Sheet1")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("💧 IV Drip Rate Predictor")

medication = st.selectbox("Select Medication", le.classes_)
dosage = st.number_input("Dosage (mcg/kg/min)", min_value=0.0, step=0.1)
weight = st.number_input("Patient Weight (kg)", min_value=1.0, step=0.5)
concentration = st.number_input("Concentration (mcg/ml)", min_value=1.0, step=50.0)

if st.button("Predict Drip Rate"):
    try:
        # Encode input
        med_code = le.transform([medication])[0]
        X_new = pd.DataFrame([[med_code, dosage, weight, concentration]],
                             columns=["Med_Code", "Dosage (mcg/kg/min)", "Patient Weight (kg)", "Concentration (mcg/ml)"])

        # Predict
        predicted_rate = model.predict(X_new)[0]
        st.success(f"💧 Predicted Drip Rate: {predicted_rate:.2f} ml/hr")

        # Log to Google Sheet (column order must match header)
        row = [
            medication,
            dosage,
            weight,
            concentration,
            round(predicted_rate, 2),
            datetime.datetime.now().isoformat()
        ]
        sheet.append_row(row)
        st.info("✅ Prediction logged to Google Sheets.")

    except Exception as e:
        st.error("❌ Error during prediction or logging:")
        st.code(str(e))


    except Exception as model_error:
        st.error("❌ Model prediction failed:")
        st.code(repr(model_error))
