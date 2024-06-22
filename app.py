import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# Load your trained model and preprocessing tools
model = load_model('best_model.h5')
with open('label_encoder.pkl', 'rb') as le:
    label_encoder = pickle.load(le)
with open('minmax_scaler.pkl', 'rb') as sc:
    scaler = pickle.load(sc)

def preprocess_input(data):
    # Scale the features using the loaded scaler
    scale_columns = ['maxtempC', 'mintempC', 'avgtempC', 'totalSnow_cm', 'sunHour', 'uvIndex',
                     'tempC', 'precipMM', 'humidity', 'cloudcover', 'windspeedKmph',
                     'visibility', 'pressure']
    non_scale_columns = ['month', 'day_of_week', 'day_of_year']  # These columns do not need scaling

    scaled_data = scaler.transform(data[scale_columns])
    scaled_data = pd.DataFrame(scaled_data, columns=scale_columns)
    non_scaled_data = data[non_scale_columns]

    scaled_data.reset_index(drop=True, inplace=True)
    non_scaled_data.reset_index(drop=True, inplace=True)

    processed_data = pd.concat([scaled_data, non_scaled_data], axis=1)
    return processed_data

def create_sequences(X, sequence_length=4):
    # Function to create a single sequence from the last 4 days of data
    if len(X) >= sequence_length:
        return np.array([X[-sequence_length:]])
    else:
        st.error("Not enough data to create a sequence.")
        return None

def make_prediction(processed_seq):
    # Make the prediction using the loaded model
    prediction = model.predict(processed_seq)
    pred_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return pred_label

def calculate_derived_date_features(date):
    date = pd.to_datetime(date)
    return {
        'month': date.month,
        'day_of_week': date.dayofweek,
        'day_of_year': date.dayofyear
    }

# Streamlit interface
st.title('Weather Forecasting App')


st.markdown("#### PROJECT WORK BY: Abdulmalik Shafiu Ozovehe")


st.write('Please enter the starting date and weather data for the past 4 days:')

# Collecting date input
start_date = st.date_input("Start Date", value=datetime.now())

# Generate dates for the 4 days
dates = [start_date + timedelta(days=i) for i in range(4)]
derived_features = pd.DataFrame([calculate_derived_date_features(date) for date in dates])

# Define columns for each day's inputs
columns = st.columns(4)
data_list = []  # List to collect data dictionaries for each day

for i, col in enumerate(columns):
    with col:
        st.subheader(f"Day {i+1}")
        day_data = {}  # Dictionary to hold data for each day
        for feature in ['maxtempC', 'mintempC', 'avgtempC', 'totalSnow_cm', 'sunHour', 'uvIndex', 'tempC', 'precipMM', 'humidity', 'cloudcover', 'windspeedKmph', 'visibility', 'pressure']:
            # Collect input and store in dictionary with correct feature names
            day_data[feature] = st.number_input(f"{feature}", step=0.1, format="%.2f", key=f"{feature}_{i}")
        data_list.append(day_data)

# Convert the list of dictionaries to DataFrame
input_data = pd.DataFrame(data_list)

# Combine input data with derived features
input_df = pd.concat([input_data, derived_features], axis=1)

if st.button('Predict Weather for Tomorrow'):
    processed_data = preprocess_input(input_df)
    processed_seq = create_sequences(processed_data)
    if processed_seq is not None:
        prediction = make_prediction(processed_seq)
        st.write(f'The forecasted weather condition for tomorrow is: {prediction[0]}')
