from tensorflow.keras.models import load_model
import pickle
import streamlit as st
import pandas as pd

# Load the trained model
model = load_model('data_files/churn_model.h5')

# Load the encoder and scaler
with open('data_files/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('data_files/onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geography = pickle.load(file)

with open('data_files/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Title of the app
st.title("Customer Churn Prediction")

# User input form
credit_score = st.number_input("Credit Score")
geography = st.selectbox("Geography", onehot_encoder_geography.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)  
balance = st.number_input("Balance")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)  
num_of_products = st.number_input("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Input Data
input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
}

input_df = pd.DataFrame(input_data)

# Convert 'Geography' using OneHotEncoder (France, Germany, Spain)
geography_encoded = onehot_encoder_geography.transform([[geography]])
geography_encoded_df = pd.DataFrame(geography_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

# Combine the encoded 'Geography' column with the original input data
input_df = input_df.reset_index(drop=True)
input_df = pd.concat([input_df, geography_encoded_df], axis=1)

# Standardize the numerical input data using the scaler
input_scaled = scaler.transform(input_df)

# Make a prediction with the trained model
prediction = model.predict(input_scaled)

# Display the prediction in Streamlit
st.write("Prediction Probability:", prediction[0][0])

churn_prediction = 1 if prediction[0][0] > 0.05 else 0
if churn_prediction == 1:
    st.write("Prediction: The customer is **likely to churn**.")
else:
    st.write("Prediction: The customer is **not likely to churn**.")

