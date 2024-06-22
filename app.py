
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_resources():
    try:
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, 'best_model.h5')
        label_encoder_path = os.path.join(current_dir, 'label_encoder.pkl')
        scaler_path = os.path.join(current_dir, 'minmax_scaler.pkl')
        
        model = load_model(model_path)
        with open(label_encoder_path, 'rb') as le:
            label_encoder = pickle.load(le)
        with open(scaler_path, 'rb') as sc:
            scaler = pickle.load(sc)
        logging.info("Resources loaded successfully.")
        return model, label_encoder, scaler
    except Exception as e:
        logging.error(f"Failed to load resources: {e}")
        raise

model, label_encoder, scaler = load_resources()

def preprocess_input(data):
    try:
        scale_columns = ['maxtempC', 'mintempC', 'avgtempC', 'totalSnow_cm', 'sunHour', 'uvIndex',
                         'tempC', 'precipMM', 'humidity', 'cloudcover', 'windspeedKmph',
                         'visibility', 'pressure']
        non_scale_columns = ['month', 'day_of_week', 'day_of_year']

        scaled_data = scaler.transform(data[scale_columns])
        scaled_data = pd.DataFrame(scaled_data, columns=scale_columns)
        non_scaled_data = data[non_scale_columns]

        scaled_data.reset_index(drop=True, inplace=True)
        non_scaled_data.reset_index(drop=True, inplace=True)

        processed_data = pd.concat([scaled_data, non_scaled_data], axis=1)
        logging.info("Input preprocessing completed successfully.")
        return processed_data
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        st.error("Failed to preprocess input data.")
        return None

def create_sequences(X, sequence_length=4):
    try:
        if len(X) >= sequence_length:
            return np.array([X[-sequence_length:]])
        else:
            st.error("Not enough data to create a sequence.")
            return None
    except Exception as e:
        logging.error(f"Error during sequence creation: {e}")
        st.error("Failed to create sequences from input data.")
        return None

def make_prediction(processed_seq):
    try:
        prediction = model.predict(processed_seq)
        pred_label = label_encoder.inverse_transform([np.argmax(prediction)])
        logging.info("Prediction made successfully.")
        return pred_label
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        st.error("Failed to make prediction.")
        return None

def calculate_derived_date_features(date):
    try:
        date = pd.to_datetime(date)
        features = {
            'month': date.month,
            'day_of_week': date.dayofweek,
            'day_of_year': date.dayofyear
        }
        logging.info("Date features derived successfully.")
        return features
    except Exception as e:
        logging.error(f"Error calculating derived date features: {e}")
        st.error("Failed to calculate date features.")
        return {}

def main():
    st.title('Weather Forecasting App')
    st.markdown("#### PROJECT WORK BY: Abdulmalik Shafiu Ozovehe")
    st.write('Please enter the starting date and weather data for the past 4 days:')

    with st.form("input_form"):
        start_date = st.date_input("Start Date", value=datetime.now())
        dates = [start_date + timedelta(days=i) for i in range(4)]
        derived_features = pd.DataFrame([calculate_derived_date_features(date) for date in dates])

        columns = st.columns(4)
        data_list = []

        for i, col in enumerate(columns):
            with col:
                st.subheader(f"Day {i+1}")
                day_data = {}
                for feature in ['maxtempC', 'mintempC', 'avgtempC', 'totalSnow_cm', 'sunHour', 'uvIndex', 'tempC', 'precipMM', 'humidity', 'cloudcover', 'windspeedKmph', 'visibility', 'pressure']:
                    day_data[feature] = st.number_input(f"{feature}", step=0.1, format="%.2f", key=f"{feature}_{i}")
                data_list.append(day_data)

        submit_button = st.form_submit_button("Predict Weather for Tomorrow")

    if submit_button:
        input_data = pd.DataFrame(data_list)
        input_df = pd.concat([input_data, derived_features], axis=1)

        processed_data = preprocess_input(input_df)
        if processed_data is not None:
            processed_seq = create_sequences(processed_data)
            if processed_seq is not None:
                prediction = make_prediction(processed_seq)
                if prediction is not None:
                    st.write(f'The forecasted weather condition for tomorrow is: {prediction[0]}')

if __name__ == '__main__':
    main()
