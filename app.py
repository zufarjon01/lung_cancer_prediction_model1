import streamlit as st
import joblib
import numpy as np

# Modelni yuklash
try:
    model = joblib.load('lung_cancer_prediction_model.pkl')
except Exception as e:
    st.error(f"Modelni yuklashda xatolik yuz berdi: {e}")
    st.stop()

# Streamlit interfeysi
st.title("Lung Cancer Prediction")
st.markdown("""
Bu ilova bemorning simptomlari va boshqa omillarni hisobga olib, o'pka saratoni ehtimolini bashorat qiladi.
Iltimos, quyidagi maydonlarni to'ldiring.
""")

# Input maydonlar
gender = st.selectbox("Jinsingiz:", options=[("Ayol", 0), ("Erkak", 1)], format_func=lambda x: x[0])[1]
age = st.number_input("Yoshingiz:", min_value=0, max_value=120, value=30)
smoking = st.selectbox("Chekasizmi?", options=[("Yo'q", 0), ("Ha", 1)])[1]
yellow_fingers = st.selectbox("Barmoqlaringiz sarg'ayganmi?", options=[("Yo'q", 0), ("Ha", 1)])[1]
anxiety = st.selectbox("Bezovtalanish bo'lganmi?", options=[("Yo'q", 0), ("Ha", 1)])[1]
peer_pressure = st.selectbox("Yaqinlaringiz chekishga majburlaganmi?", options=[("Yo'q", 0), ("Ha", 1)])[1]
chronic_disease = st.selectbox("Surunkali kasalliklar mavjudmi?", options=[("Yo'q", 0), ("Ha", 1)])[1]
fatigue = st.selectbox("Charchoqni his qilasizmi?", options=[("Yo'q", 0), ("Ha", 1)])[1]
allergy = st.selectbox("Allergiya bormi?", options=[("Yo'q", 0), ("Ha", 1)])[1]
wheezing = st.selectbox("Xansirash bo'lganmi?", options=[("Yo'q", 0), ("Ha", 1)])[1]
alcohol_consuming = st.selectbox("Spirtli ichimlik iste'mol qilasizmi?", options=[("Yo'q", 0), ("Ha", 1)])[1]
coughing = st.selectbox("Yo'tal bo'lganmi?", options=[("Yo'q", 0), ("Ha", 1)])[1]
shortness_of_breath = st.selectbox("Nafas qisishi bo'lganmi?", options=[("Yo'q", 0), ("Ha", 1)])[1]
swallowing_difficulty = st.selectbox("Yutishda qiyinchiliklar bormi?", options=[("Yo'q", 0), ("Ha", 1)])[1]
chest_pain = st.selectbox("Ko'krak og'rig'i bo'lganmi?", options=[("Yo'q", 0), ("Ha", 1)])[1]

# Bashorat tugmasi
if st.button("Bashorat qilish"):
    # Foydalanuvchi kiritgan ma'lumotlarni tayyorlash
    features = np.array([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                          chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                          coughing, shortness_of_breath, swallowing_difficulty, chest_pain]])
    
    # Bashorat qilish
    try:
        prediction = model.predict(features)
        result = "O'pka saratoni aniqlangan" if prediction[0] == 1 else "O'pka saratoni aniqlanmadi"
        st.success(f"Bashorat natijasi: {result}")
    except Exception as e:
        st.error(f"Bashorat qilishda xatolik yuz berdi: {e}")
