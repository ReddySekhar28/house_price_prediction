import streamlit as st
import pandas as pd
import pickle
import os

# Load files safely
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# App title
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")
st.title("🏠 House Price Prediction App")

st.write("Fill the details below to estimate the house price")

# --- INPUTS ---
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1.0, 10.0, 2.0)
sqft_living = st.number_input("Living Area (sqft)", 500, 10000, 1500)
sqft_lot = st.number_input("Lot Area (sqft)", 1000, 50000, 5000)
floors = st.number_input("Floors", 1.0, 3.0, 1.0)
waterfront = st.selectbox("Waterfront (0 = No, 1 = Yes)", [0, 1])
view = st.slider("View Rating", 0, 4, 0)
condition = st.slider("Condition", 1, 5, 3)
sqft_above = st.number_input("Sqft Above", 500, 10000, 1200)
sqft_basement = st.number_input("Sqft Basement", 0, 5000, 300)
yr_built = st.number_input("Year Built", 1900, 2024, 2000)
yr_renovated = st.number_input("Year Renovated", 0, 2024, 0)

# Derived features (same as training)
year = 2014
month = 5

# --- PREDICTION ---
if st.button("Predict Price"):
    try:
        input_data = {
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft_living': sqft_living,
            'sqft_lot': sqft_lot,
            'floors': floors,
            'waterfront': waterfront,
            'view': view,
            'condition': condition,
            'sqft_above': sqft_above,
            'sqft_basement': sqft_basement,
            'yr_built': yr_built,
            'yr_renovated': yr_renovated,
            'year': year,
            'month': month
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Handle categorical features
        input_df = pd.get_dummies(input_df)

        # Align with training columns
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)

        st.success(f"💰 Estimated Price: ${prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
