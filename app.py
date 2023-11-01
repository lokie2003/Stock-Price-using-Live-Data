import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained LSTM model
model = load_model('StockPrice.h5')  # Replace 'your_model.h5' with the actual path to your model

# Define a function to make predictions
def make_predictions(input_sequence, num_days):
    input_sequence = np.array(input_sequence).reshape(-1, 1)
    # Scaling the input sequence
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_sequence = scaler.fit_transform(input_sequence)
    
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

# Streamlit app
st.title('Stock Price Prediction App')

# Sidebar with user input
st.sidebar.header('User Input')
st.sidebar.markdown('Enter historical stock prices (comma-separated):')
input_sequence = st.sidebar.text_input('Historical Prices', '207.48, 201.59, 203.77, 209.95, 208.49')

# Convert user input to a list of floats
input_sequence = [float(x) for x in input_sequence.split(',')]

num_days = st.sidebar.slider('Number of Days to Predict', 1, 30, 10)

# Main content
st.write(f"Predicting {num_days} days of stock prices based on user input")

# Make predictions
if st.button('Make Predictions'):
    predictions = make_predictions(input_sequence, num_days)
    st.write(predictions)

# Plot the predicted prices
st.subheader('Predicted Prices')
if st.button('Plot Predicted Prices'):
    st.line_chart(predictions)
