import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Downloading the model from the Model Hub
model_path = hf_hub_download(repo_id="Pammi123/engine-model", filename="best_model_v1.joblib")

# Loading the model
model = joblib.load(model_path)

# Streamlit UI for Engine Health Prediction
st.title("Engine Health Prediction App")
st.write("This application predicts the condition of an engine based on operating parameters such as RPM, pressures, and temperatures.")
st.write("Kindly enter the engine operating parameters to predict the engine health.")

# Collecting user inputs

engine_rpm = st.number_input("Engine RPM", min_value=0, max_value=2500, value=800)
lub_oil_pressure = st.number_input("Lubricating Oil Pressure", min_value=0.0, max_value=10.0, value=3.0)
fuel_pressure = st.number_input("Fuel Pressure", min_value=0.0, max_value=25.0, value=6.0)
coolant_pressure = st.number_input( "Coolant Pressure", min_value=0.0, max_value=10.0, value=2.0)
lub_oil_temp = st.number_input( "Lubricating Oil Temperature (°C)", min_value=60.0, max_value=100.0, value=77.0)
coolant_temp = st.number_input( "Coolant Temperature (°C)", min_value=50.0, max_value=200.0, value=78.0)

# Create dataframe
input_data = pd.DataFrame([{
    'Engine rpm': engine_rpm,
    'Lub oil pressure': lub_oil_pressure,
    'Fuel pressure': fuel_pressure,
    'Coolant pressure': coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': coolant_temp
}])

# Set the classification threshold
classification_threshold = 0.5

# Predict button
if st.button("Predict Engine Condition"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "On/True/Faulty" if prediction == 1 else "Off/False/Active"
    st.write(f"Based on the information provided, the engine is most likely {result}.")
