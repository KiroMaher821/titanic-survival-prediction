# titanic-survival-prediction
A machine learning web application built with Streamlit that predicts passenger survival on the Titanic using a Random Forest Classifier.
# 🚢 Titanic Survival Prediction Web App

## 📌 Project Overview
This project applies machine learning techniques to analyze the passenger manifest of the RMS Titanic and predict the likelihood of survival based on various demographic and socio-economic factors. The final predictive model is deployed as an interactive web application using Streamlit.

## 📊 Data Exploration & Engineering
During the Exploratory Data Analysis (EDA) phase, we investigated how different variables impacted survival rates, including:
* **Socio-economic status** (Ticket Class)
* **Demographics** (Age and Sex)
* **Ticket Fare**
* **Embarkation Port**

**Feature Engineering:** A new feature, `FamilySize`, was created by combining the 'SibSp' (siblings/spouses) and 'Parch' (parents/children) columns to determine if traveling with family influenced survival chances.

## 🤖 Machine Learning Models
To find the best predictive algorithm, we trained and evaluated three different models:
1. **Random Forest Classifier** (Selected as the final model and optimized via Hyperparameter Tuning)
2. **Logistic Regression** 
3. **Support Vector Machine (SVM)**

Feature selection was performed using an Embedded Method (Random Forest Feature Importances) to isolate the most impactful variables and reduce noise before final model training.

## 🛠️ Technologies Used
* **Python**
* **Pandas & NumPy** (Data manipulation)
* **Matplotlib & Seaborn** (Data visualization)
* **Scikit-Learn** (Machine learning, GridSearchCV, Label Encoding)
* **Streamlit** (Web application deployment)

## 🌐 Live Application
The project has been deployed to the web! You can interact with the live model here: 
**[Insert Your Streamlit Link Here]**

# presentation video 

https://drive.google.com/file/d/1_l0Awvh2FlpSpxu7S9Wm9lpWTcFldUGN/view?usp=sharing
