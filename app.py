import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained LSTM model
model = load_model("StockPricefinal.h5")
scaler = joblib.load('scaler.pkl')

# Define the input fields
st.write("# Stock Price Prediction App")
st.write("Enter the last 10 days' closing prices for AAPL:")

inputs = []
for i in range(10):
    inputs.append(st.number_input(f"Day {i+1}", min_value=0.0))

if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = np.array(inputs).reshape(1, 10, 1)
    
    # Make the prediction
    prediction = model.predict(input_data)
    
    # Inverse transform the prediction to get the actual stock price
    predicted_price = scaler.inverse_transform(prediction)
    
    st.write(f"Predicted Stock Price for the next day: ${predicted_price[0][0]:.2f}")
