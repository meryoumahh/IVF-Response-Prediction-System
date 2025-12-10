"""IVF Response Prediction - Streamlit Web Application

Interactive web interface for predicting patient response to ovarian stimulation.
Provides real-time predictions with probability outputs and visualizations.

Features:
- Input validation
- Probabilistic predictions
- Confidence scores
- Probability distribution charts


"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import os

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import MODEL_FILE

# === MODEL LOADING ===
# Load trained Random Forest model with error handling
try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_FILE}. Please train the model first by running random_forest_classifier.py")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# === APP HEADER ===
st.title("🧬 IVF Response Predictor")
st.write("Enter patient details to predict ovarian stimulation response.")
st.write("Model provides probabilistic predictions: Low, Optimal, or High response.")

# === USER INPUT SECTION ===
# Sidebar contains all input fields with validation
st.sidebar.header("Patient Data")
age = st.sidebar.slider("Age", 20, 45, 30)
amh = st.sidebar.number_input("AMH Level", min_value=0.0, max_value=20.0, value=3.64, step=0.1)
follicles = st.sidebar.number_input("Antral Follicles (AFC)", min_value=0, max_value=50, value=9, step=1)
cycle = st.sidebar.selectbox("Cycle Number", [1, 2, 3, 4, 5])
protocol = st.sidebar.selectbox("Protocol", ["flexible antagonist", "fixed antagonist", "agonist"])

# === INPUT VALIDATION ===
# Prevent negative values that would be clinically impossible
if age < 0:
    st.error("❌ Age cannot be negative")
    st.stop()
if amh < 0:
    st.error("❌ AMH level cannot be negative")
    st.stop()
if follicles < 0:
    st.error("❌ AFC cannot be negative")
    st.stop()

# === INPUT PREPROCESSING ===
# Prepare input data to match model's expected format
# Get exact feature names and order from trained model
model_features = model.feature_names_in_

# Create input data matching the exact training features
input_data = {
    'cycle_number': cycle,
    'Age': age,
    'AMH': amh,
    'n_Follicles': follicles,
    'E2_day5': 0,  # Default value
    'AFC': follicles,
    'row_rank': 1,  # Default value
    'Protocol_Fixed Antagonist': 1 if protocol == "fixed antagonist" else 0,
    'Protocol_Flexible Antagonist': 1 if protocol == "flexible antagonist" else 0
}

# Convert to DataFrame with exact column order from training
features = pd.DataFrame([input_data])

# Ensure all model features exist and are in the correct order
for col in model_features:
    if col not in features.columns:
        features[col] = 0

features = features[model_features]

# === PREDICTION SECTION ===
if st.button("🔮 Predict Response"):
    # Get prediction and probabilities
    prediction_idx = model.predict(features)[0]  # Predicted class index
    probs = model.predict_proba(features)[0]     # Probability for each class
    
    # Map prediction index to class name
    classes = ['Low', 'Optimal', 'High']
    predicted_class = classes[prediction_idx]
    
    st.subheader(f"Prediction: {predicted_class} Response")
    st.write(f"Confidence: {probs[prediction_idx]*100:.2f}%")
    
    # Simple Bar Chart of Probabilities
    prob_df = pd.DataFrame(probs, index=classes, columns=["Probability"])
    st.bar_chart(prob_df)
