import streamlit as st
import pandas as pd
import joblib

# Configure the page
st.set_page_config(page_title="Model Monitoring Dashboard", layout="wide")
st.title("Model Monitoring and Drift Detection Dashboard")

# --- Performance Metrics Section ---
st.header("Performance Metrics")

# Load datasets and the trained model.
ref = pd.read_csv('reference_data.csv', parse_dates=['date'])
curr = pd.read_csv('current_data.csv', parse_dates=['date'])
model = joblib.load('model.pkl')

# Add predictions to both datasets.
ref['prediction'] = model.predict(ref[['feature1', 'feature2']])
curr['prediction'] = model.predict(curr[['feature1', 'feature2']])

# Compute mean squared error (MSE)
mse_ref = ((ref['target'] - ref['prediction'])**2).mean()
mse_curr = ((curr['target'] - curr['prediction'])**2).mean()

col1, col2 = st.columns(2)
col1.metric("Reference MSE", f"{mse_ref:.2f}")
col2.metric("Current MSE", f"{mse_curr:.2f}")

if mse_curr > 1.5 * mse_ref:
    st.error("ALERT: Performance degradation detected!")
else:
    st.success("Model performance is stable.")

# --- Evidently AI Report Section ---
st.header("Evidently AI Report")
try:
    with open("evidently_report.html", "r", encoding="utf-8") as f:
        report_html = f.read()
    st.components.v1.html(report_html, height=800, scrolling=True)
except Exception as e:
    st.error("Evidently report not found. Please run monitor.py first.")

# --- Feature Importance Drift Section ---
st.header("Feature Importance Drift")
try:
    st.image("feature_importance_drift.png", caption="Feature Importance: Reference vs Current")
except Exception as e:
    st.error("Feature importance plot not found. Please run monitor.py first.")
