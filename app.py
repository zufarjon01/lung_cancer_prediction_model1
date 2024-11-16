import streamlit as st
import joblib
import numpy as np

# Modelni yuklash
model = joblib.load('lung_cancer_prediction_model.pkl')

# Streamlit interfeysi
st.title("Lung Cancer Prediction")
st.write("Kiritilgan ma'lumotlarga asoslanib, o'pka saratonini bashorat qiluvchi dastur")

# Foydalanuvchi uchun kirish maydonlari
gender = st.selectbox("Gender (0=Female, 1=Male):", [0, 1])
age = st.number_input("Age:", min_value=1, max_value=120, step=1)
smoking = st.selectbox("Smoking (0=No, 1=Yes):", [0, 1])
yellow_fingers = st.selectbox("Yellow Fingers (0=No, 1=Yes):", [0, 1])
anxiety = st.selectbox("Anxiety (0=No, 1=Yes):", [0, 1])
peer_pressure = st.selectbox("Peer Pressure (0=No, 1=Yes):", [0, 1])
chronic_disease = st.selectbox("Chronic Disease (0=No, 1=Yes):", [0, 1])
fatigue = st.selectbox("Fatigue (0=No, 1=Yes):", [0, 1])
allergy = st.selectbox("Allergy (0=No, 1=Yes):", [0, 1])
wheezing = st.selectbox("Wheezing (0=No, 1=Yes):", [0, 1])
alcohol_consuming = st.selectbox("Alcohol Consuming (0=No, 1=Yes):", [0, 1])
coughing = st.selectbox("Coughing (0=No, 1=Yes):", [0, 1])
shortness_of_breath = st.selectbox("Shortness of Breath (0=No, 1=Yes):", [0, 1])
swallowing_difficulty = st.selectbox("Swallowing Difficulty (0=No, 1=Yes):", [0, 1])
chest_pain = st.selectbox("Chest Pain (0=No, 1=Yes):", [0, 1])

# Bashorat qilish
if st.button("Predict"):
    features = np.array([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                          chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                          coughing, shortness_of_breath, swallowing_difficulty, chest_pain]])
    
    prediction = model.predict(features)
    result = "Lung Cancer Detected" if prediction[0] == 1 else "No Lung Cancer Detected"
    st.subheader(f"Prediction Result: {result}")
