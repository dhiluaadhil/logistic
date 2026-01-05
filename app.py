import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Titanic Survival (LR)", page_icon="🚢")
st.title("🚢 Titanic Survival - Logistic Regression")

model_path = 'titanic_model.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
    
    st.write("Enter passenger details to predict survival probability.")

    col1, col2 = st.columns(2)
    with col1:
        p_class = st.selectbox("Class (1=High, 3=Low)", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 100, 25)
        sib_sp = st.number_input("Siblings Aboard", 0, 10, 0)
    
    with col2:
        parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
        fare = st.number_input("Fare", 0.0, 600.0, 32.0)
        embarked = st.selectbox("Port", ["S", "C", "Q"])

    if st.button("Predict"):
        # Encoding
        sex_num = 1 if sex == "male" else 0
        emb_map = {"C": 0, "Q": 1, "S": 2}
        emb_num = emb_map[embarked]
        
        # Features: [p_class, sex, age, sib_sp, parch, fare, embarked]
        features = np.array([[p_class, sex_num, age, sib_sp, parch, fare, emb_num]])
        prediction = model.predict(features)
        
        if prediction[0] == 1:
            st.success("✅ Prediction: SURVIVED")
        else:
            st.error("💀 Prediction: NOT SURVIVED")
else:
    st.warning("Please upload 'titanic_model.pkl' to GitHub.")