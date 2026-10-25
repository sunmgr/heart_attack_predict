import streamlit as st
import pandas as pd
import joblib


model = joblib.load("LogisticRegression.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")



st.title("Heart Disease Prediction")
st.markdown("this app predicts the likelihood of heart disease based on user input features.")


age = st.slider("Age",18,100,40)
sex = st.selectbox("Sex",["M","F"])
chest_pain = st.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",[0,1])
resting_ecg = st.selectbox("Resting ECG Results",["Normal","ST","LVH"])
max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina",["Y","N"])
oldpeak = st.slider("Oldpeak (ST depression induced by exercise)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope",["Up","Flat","Down"])


if st.button("Predict"):
    raw_input = {
        'Age':age,
        'RestingBP':resting_bp,
        'Cholesterol':cholesterol,
        'FastingBS':fasting_bs,
        'MaxHR':max_hr,
        'Oldpeak':oldpeak,
        'Sex_'+sex:1,
        'ChestPainType_'+chest_pain:1,
        'RestingECG_'+resting_ecg:1,
        'ExerciseAngina_'+exercise_angina:1,
        'ST_Slope_'+st_slope:1
    }
    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)

    if prediction == 1:
        st.error("The model predicts that you are likely to have heart disease. Please consult a healthcare professional for further evaluation.")
    else:
        st.success("The model predicts that you are unlikely to have heart disease. However, please consult a healthcare professional for a comprehensive assessment.")
        