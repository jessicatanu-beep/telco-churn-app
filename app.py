import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

# Load artifacts
@st.cache_resource
def load_model():
    return joblib.load("model_pipeline.joblib")

def load_threshold():
    with open("threshold.json", "r") as f:
        return json.load(f)["threshold"]

def load_schema():
    with open("schema.json", "r") as f:
        return json.load(f)["feature_columns"]

model = load_model()
THR = load_threshold()
FEATURE_COLS = load_schema()

# Risk segmentation
def risk_level(p):
    if p > 0.7:
        return "High Risk"
    elif p > 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

# Recommendation
def recommendation(risk):
    if risk == "High Risk":
        return "Prioritaskan retensi pelanggan (promo / upgrade layanan)"
    elif risk == "Medium Risk":
        return "Berikan promo ringan atau reminder benefit"
    else:
        return "Maintain relationship"

# Cleaning helper
def clean_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    # Drop ID kalau ikut di CSV
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    # Pastikan kolom numeric penting benar-benar numeric
    # TotalCharges sering berisi " " (spasi) -> jadi NaN
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = (
            df_clean["TotalCharges"]
            .astype(str)
            .str.strip()
            .replace("", None)
        )
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

    # jaga-jaga kalau kebaca object/string
    if "MonthlyCharges" in df_clean.columns:
        df_clean["MonthlyCharges"] = pd.to_numeric(df_clean["MonthlyCharges"], errors="coerce")

    if "tenure" in df_clean.columns:
        df_clean["tenure"] = pd.to_numeric(df_clean["tenure"], errors="coerce")

    if "SeniorCitizen" in df_clean.columns:
        df_clean["SeniorCitizen"] = pd.to_numeric(df_clean["SeniorCitizen"], errors="coerce")

    return df_clean

# Prediction function
def predict_df(df):
    # 1) bersihin dulu biar aman
    df_clean = clean_input_df(df)

    # 2) validasi kolom wajib sesuai schema
    missing = [c for c in FEATURE_COLS if c not in df_clean.columns]
    if missing:
        raise ValueError(f"Kolom berikut tidak ada di file: {missing}")

    X = df_clean[FEATURE_COLS].copy()
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= THR).astype(int)

    out = df.copy()
    out["churn_probability"] = proba
    out["churn_prediction"] = pd.Series(pred).map({1: "Yes", 0: "No"})  # biar rapi Yes/No
    out["risk_level"] = out["churn_probability"].apply(risk_level)
    out["recommendation"] = out["risk_level"].apply(recommendation)

    return out


st.title("Telco Customer Churn Predictor")

st.write(
"""
Aplikasi ini digunakan untuk memprediksi kemungkinan pelanggan melakukan churn.

Output:
- Prediksi churn
- Probabilitas churn
- Risk level
- Rekomendasi aksi
"""
)

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# Single prediction
with tab1:
    st.subheader("Single Customer Prediction")

    input_data = {}
    for col in FEATURE_COLS:
        input_data[col] = st.text_input(col)

    if st.button("Predict"):
        df_input = pd.DataFrame([input_data])

        # optional: bersihin input single juga (biar numeric kebaca)
        df_input = clean_input_df(df_input)

        result = predict_df(df_input).iloc[0]

        st.success("Prediction generated")
        st.write("### Result")
        st.write("Churn Prediction:", result["churn_prediction"])
        st.write("Churn Probability:", result["churn_probability"])
        st.write("Risk Level:", result["risk_level"])
        st.write("Recommendation:", result["recommendation"])


# Batch prediction
with tab2:
    st.subheader("Batch Prediction (Upload CSV)")

    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)

        st.write("Preview Data")
        st.dataframe(df.head())

        if st.button("Run Prediction"):
            scored = predict_df(df)

            st.success("Batch prediction selesai")
            st.dataframe(scored.head())

            csv = scored.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Result CSV",
                csv,
                "churn_prediction_result.csv",
                "text/csv"
            )