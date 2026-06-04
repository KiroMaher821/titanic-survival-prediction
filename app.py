import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained model
model = joblib.load('model.joblib')

# 2. Set up the web app title and text
st.title("🚢 Titanic Survival Predictor")
st.write("Enter the passenger's details below to predict if they would have survived the disaster.")

# 3. Create user input fields (Notice: Embarked is gone!)
pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0.0, 100.0, 30.0)
family_size = st.slider("Family Size (Siblings/Spouse + Parents/Children)", 0, 10, 0)
fare = st.number_input("Fare Paid (£)", 0.0, 600.0, 32.0)

# 4. Preprocess the user's inputs
sex_encoded = 1 if sex == "Female" else 0

# 5. Format the data perfectly for the model
# MATCHING EXACTLY: ['Sex', 'Fare', 'Age', 'Pclass', 'FamilySize']
input_data = pd.DataFrame({
    'Sex': [sex_encoded],
    'Fare': [fare],
    'Age': [age],
    'Pclass': [pclass],
    'FamilySize': [family_size]
})

# 6. Add a button to trigger the prediction
if st.button("Predict Survival"):
    # Get prediction and probabilities
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    if prediction == 1:
        st.success(f"This passenger likely **SURVIVED**! 🛶 (Probability: {prediction_proba[1]:.2%})")
    else:
        st.error(f"This passenger likely **DID NOT SURVIVE**. 🧊 (Probability: {prediction_proba[0]:.2%})")
