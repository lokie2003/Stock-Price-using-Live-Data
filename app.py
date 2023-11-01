import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained LSTM model
model = load_model('StockPrice.h5')  # Replace 'your_model.h5' with the actual path to your model

# Define a function to make predictions
def make_predictions(start_date, num_days):
    # Initialize the input sequence with data from 'start_date'
    input_sequence = df[df['date'] >= start_date]['close'].values.tolist()
    
    # Scaling the input sequence
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_sequence = scaler.fit_transform(np.array(input_sequence).reshape(-1, 1))
    
    predictions = []
    for _ in range(num_days):
        # Prepare the input data for the model
        x_input = np.array(input_sequence[-100:]).reshape(1, 100, 1)
        
        # Make a prediction for the next day
        predicted_price = model.predict(x_input)
        
        # Inverse transform the predicted price
        predicted_price = scaler.inverse_transform(predicted_price)
        
        # Append the prediction to the result list
        predictions.append(predicted_price[0][0])
        
        # Extend the input sequence with the new prediction
        input_sequence = np.append(input_sequence, predicted_price)
    
    return predictions

# Load the data
df = pd.read_csv('AAPL.csv')

# Streamlit app
st.title('Stock Price Prediction App')

# Sidebar with user input
st.sidebar.header('User Input')
start_date = st.sidebar.date_input('Select a start date', pd.to_datetime('2023-01-01'))
num_days = st.sidebar.slider('Number of Days to Predict', 1, 30, 10)

# Main content
st.write(f"Predicting {num_days} days of stock prices starting from {start_date}")

# Make predictions
if st.button('Make Predictions'):
    predictions = make_predictions(start_date, num_days)
    st.write(predictions)

# Data table
st.subheader('Data Table')
st.write(df.tail())

# Plot the historical data
st.subheader('Historical Data')
st.line_chart(df[df['date'] >= start_date]['close'])

# Plot the predicted prices
st.subheader('Predicted Prices')
if st.button('Plot Predicted Prices'):
    predicted_dates = pd.date_range(start=start_date, periods=num_days)
    predicted_data = pd.DataFrame({'date': predicted_dates, 'predicted_price': predictions})
    st.line_chart(predicted_data)

