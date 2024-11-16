import streamlit as st
import joblib
import numpy as np

# Modelni yuklash
model = joblib.load('lung_cancer_prediction_model.pkl')

# Streamlit interfeysi
st.title("Lung Cancer Prediction")
st.write("Please enter the following details to predict lung cancer risk:")

# Kirish formasi
gender = st.selectbox("Gender", options=["Female", "Male"])
age = st.number_input("Age:", min_value=1, max_value=120, step=1)
smoking = st.selectbox("Smoking:", options=["No", "Yes"])
yellow_fingers = st.selectbox("Yellow Fingers:", options=["No", "Yes"])
anxiety = st.selectbox("Anxiety:", options=["No", "Yes"])
peer_pressure = st.selectbox("Peer Pressure:", options=["No", "Yes"])
chronic_disease = st.selectbox("Chronic Disease:", options=["No", "Yes"])
fatigue = st.selectbox("Fatigue:", options=["No", "Yes"])
allergy = st.selectbox("Allergy:", options=["No", "Yes"])
wheezing = st.selectbox("Wheezing:", options=["No", "Yes"])
alcohol_consuming = st.selectbox("Alcohol Consuming:", options=["No", "Yes"])
coughing = st.selectbox("Coughing:", options=["No", "Yes"])
shortness_of_breath = st.selectbox("Shortness of Breath:", options=["No", "Yes"])
swallowing_difficulty = st.selectbox("Swallowing Difficulty:", options=["No", "Yes"])
chest_pain = st.selectbox("Chest Pain:", options=["No", "Yes"])

# Ma'lumotlarni tayyorlash
features = np.array([[
    1 if gender == "Male" else 0,
    age,
    1 if smoking == "Yes" else 0,
    1 if yellow_fingers == "Yes" else 0,
    1 if anxiety == "Yes" else 0,
    1 if peer_pressure == "Yes" else 0,
    1 if chronic_disease == "Yes" else 0,
    1 if fatigue == "Yes" else 0,
    1 if allergy == "Yes" else 0,
    1 if wheezing == "Yes" else 0,
    1 if alcohol_consuming == "Yes" else 0,
    1 if coughing == "Yes" else 0,
    1 if shortness_of_breath == "Yes" else 0,
    1 if swallowing_difficulty == "Yes" else 0,
    1 if chest_pain == "Yes" else 0,
]])

# Bashorat qilish
if st.button("Predict"):
    prediction = model.predict(features)
    result = "Lung Cancer Detected" if prediction[0] == 1 else "No Lung Cancer Detected"
    st.success(f"Prediction: {result}")
