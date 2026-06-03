import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained model
# (This expects 'model.joblib' to be in the same folder)
model = joblib.load('model.joblib')

# 2. Set up the web app title and text
st.title("🚢 Titanic Survival Predictor")
st.write("Enter the passenger's details below to predict if they would have survived the disaster.")

# 3. Create user input fields
# We use standard Streamlit widgets like selectboxes, sliders, and number inputs
pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0.0, 100.0, 30.0)
family_size = st.slider("Family Size (Siblings/Spouse + Parents/Children)", 0, 10, 0)
fare = st.number_input("Fare Paid (£)", 0.0, 600.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])

# 4. Preprocess the user's inputs
# Models only understand numbers, so we must encode the text just like we did in the notebook
sex_encoded = 1 if sex == "Female" else 0

# Assuming you used One-Hot Encoding for the Embarked column
embarked_C = 1 if embarked == "Cherbourg (C)" else 0
embarked_Q = 1 if embarked == "Queenstown (Q)" else 0
embarked_S = 1 if embarked == "Southampton (S)" else 0

# 5. Format the data perfectly for the model
# CRITICAL: These column names MUST exactly match the columns your model was trained on!
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_encoded],
    'Age': [age],
    'Fare': [fare],
    'FamilySize': [family_size],
    'Embarked_C': [embarked_C],
    'Embarked_Q': [embarked_Q],
    'Embarked_S': [embarked_S]
})

# 6. Add a button to trigger the prediction
if st.button("Predict Survival"):
    # The model returns an array, we want the first item [0]
    prediction = model.predict(input_data)[0]
    
    # Optional: Get the probability percentages
    prediction_proba = model.predict_proba(input_data)[0]
    
    if prediction == 1:
        st.success(f"This passenger likely **SURVIVED**! 🛶 (Probability: {prediction_proba[1]:.2%})")
    else:
        st.error(f"This passenger likely **DID NOT SURVIVE**. 🧊 (Probability: {prediction_proba[0]:.2%})")
