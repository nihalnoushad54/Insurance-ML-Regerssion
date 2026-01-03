import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open("gbr_insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("💰 Insurance Charge Prediction")
st.write("Predict medical insurance charges using Machine Learning")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# Encoding (must match training)
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

# Feature order MUST match training
input_data = np.array([[
    age,
    sex,
    bmi,
    children,
    smoker,
    region_northwest,
    region_southeast,
    region_southwest
]])

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Insurance Charge"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"💵 Estimated Insurance Charge: ₹ {prediction:,.2f}")
