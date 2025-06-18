import streamlit as st
import pandas as pd
import joblib
import datetime
import firebase_admin
from firebase_admin import credentials, db

# -----------------------------
# Load model and encoder
# -----------------------------
model = joblib.load("iv_drip_model.pkl")
le = joblib.load("label_encoder.pkl")

# -----------------------------
# Firebase credentials from secrets
# -----------------------------
firebase_keys = {
    "type": st.secrets["FIREBASE_TYPE"],
    "project_id": st.secrets["FIREBASE_PROJECT_ID"],
    "private_key_id": st.secrets["FIREBASE_PRIVATE_KEY_ID"],
    "private_key": st.secrets["FIREBASE_PRIVATE_KEY"].replace("\\n", "\n"),
    "client_email": st.secrets["FIREBASE_CLIENT_EMAIL"],
    "client_id": st.secrets["FIREBASE_CLIENT_ID"],
    "auth_uri": st.secrets["FIREBASE_AUTH_URI"],
    "token_uri": st.secrets["FIREBASE_TOKEN_URI"],
    "auth_provider_x509_cert_url": st.secrets["FIREBASE_AUTH_PROVIDER_X509_CERT_URL"],
    "client_x509_cert_url": st.secrets["FIREBASE_CLIENT_X509_CERT_URL"],
    "universe_domain": "googleapis.com"
}

# ✅ Absolute correct Realtime DB URL
FIREBASE_DB_URL = "https://iv-drip-ml-log-default-rtdb.firebaseio.com"

# -----------------------------
# Initialize Firebase (hardcoded DB URL to avoid fallback errors)
# -----------------------------
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_keys)
        firebase_admin.initialize_app(cred, {
            "databaseURL": FIREBASE_DB_URL
        })
except Exception as init_error:
    st.error("❌ Firebase initialization failed.")
    st.code(str(init_error))

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("💧 IV Drip Rate Predictor")
st.markdown("Enter patient and medication details below:")

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

        # Prediction
        predicted_rate = model.predict(X_new)[0]
        st.success(f"💧 Predicted Drip Rate: {predicted_rate:.2f} ml/hr")

        # Prepare data
        log_data = {
            "medication": medication,
            "dosage": dosage,
            "weight": weight,
            "concentration": concentration,
            "predicted_rate": predicted_rate,
            "timestamp": datetime.datetime.now().isoformat()
        }

        # Firebase push
        try:
            ref = db.reference("/predictions")
            ref.push(log_data)
            st.info("✅ Data logged to Firebase.")
        except Exception as firebase_push_error:
            st.error("🔥 Firebase push failed:")
            st.code(str(firebase_push_error))

    except Exception as prediction_error:
        st.error("❌ Error during prediction:")
        st.code(str(prediction_error))

    except Exception as prediction_error:
        st.error(f"❌ Error during prediction: {prediction_error}")
