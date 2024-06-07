import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the trained model
lr = pickle.load(open('models/lr.pkl', 'rb'))

# Streamlit app title and description
st.title("Credit Card Fraud Detection")
st.write("It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.")

# User input for required features
input_str = st.text_input("Enter required features separated by comma (Time, V1, V2, ..., V28, Amount)")

# Split the input string into a list of features
input_features = input_str.split(',')

# Convert features to numpy array
features = np.asarray(input_features, dtype=np.float64)

# Predict using the loaded model
if st.button('Submit'):
    prediction = lr.predict(features.reshape(1, -1))
    if prediction[0] == 0:
        st.write("Legitimate Transaction")
    else:
        st.write("Fraud Transaction")
