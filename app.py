import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import keras

# Load the model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open("one_hot_encoder_geo.pkl", "rb") as file:
    one_hot_encoder_geo = pickle.load(file)

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Churn Predictor')

# User input
geography = st.selectbox("Geography", ["Germany", "France", "Spain"])  # Keep exact trained values
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=42)
tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=2)
balance = st.number_input("Balance", min_value=0.0, value=0.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.radio("Has Credit Card?", [0, 1])
is_active_member = st.radio("Is Active Member?", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=101348.88)

# Convert Gender to numerical
gender_encoded = label_encoder_gender.transform([gender])[0]

# Convert Geography using One-Hot Encoding (Fixed)
geo_transformed = one_hot_encoder_geo.transform([[geography]])  # No .toarray()

# Assign feature names correctly
geo_feature_names = one_hot_encoder_geo.get_feature_names_out(["Geography"])
geo_df = pd.DataFrame(geo_transformed, columns=geo_feature_names)

# Prepare input data (without Credit Score)
input_data = {
    "Gender": gender_encoded,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary,
}

# Merge One-Hot Encoded Geography
input_data.update(dict(zip(geo_feature_names, geo_transformed[0])))

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Expected column order (without Credit Score)
expected_columns = ["Gender", "Age", "Tenure", "Balance", "NumOfProducts", 
                    "HasCrCard", "IsActiveMember", "EstimatedSalary", 
                    "Geography_France", "Geography_Germany", "Geography_Spain"]

# Add missing columns (if any)
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing columns with 0

# Reorder to match the trained scaler
input_df = input_df[expected_columns]

# Apply Scaling
scaled_input = scaler.transform(input_df)

st.write("Processed Input Data:")
st.dataframe(pd.DataFrame(scaled_input, columns=expected_columns))

# Make prediction
prediction = model.predict(scaled_input)

# Display Result
st.subheader("Prediction Result")
if prediction[0][0] > 0.5:
    st.write("The customer is likely to **Churn** 🚨")
else:
    st.write("The customer is likely to **Stay** ✅")
