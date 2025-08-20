import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model, feature_names = pickle.load(f)

st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")

st.title("🏡 House Price Prediction App")

st.write("Enter the house details below to predict its price.")

# Sidebar inputs dynamically
user_input = {}
for col in feature_names:
    if "area" in col.lower() or "sqft" in col.lower():
        user_input[col] = st.sidebar.number_input(f"{col}", min_value=200, max_value=10000, value=1000)
    elif "room" in col.lower() or "bed" in col.lower():
        user_input[col] = st.sidebar.number_input(f"{col}", min_value=1, max_value=10, value=3)
    elif "bath" in col.lower():
        user_input[col] = st.sidebar.number_input(f"{col}", min_value=1, max_value=5, value=2)
    else:
        user_input[col] = st.sidebar.number_input(f"{col}", value=0)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Prediction button
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated House Price: **${prediction:,.2f}**")
