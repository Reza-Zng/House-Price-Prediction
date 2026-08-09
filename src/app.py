import streamlit as st
import pandas as pd
import joblib
import os

# Page Configuration (MUST be the first Streamlit command)
st.set_page_config(page_title="Tehran House Price Prediction", layout="wide")

# Custom CSS to force text, headings, and checkboxes to align center
st.markdown("""
    <style>
    /* Center main titles and text */
    .stApp h1, .stApp p {
        text-align: center;
    }
    /* Center checkboxes */
    [data-testid="stCheckbox"] {
        display: flex;
        justify-content: center;
    }
    /* Center the button */
    div.stButton {
        display: flex;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load Trained Assets
@st.cache_resource
def load_assets():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    base_path = os.path.join(BASE_DIR, 'saved_models')
    
    model_path = os.path.join(base_path, 'tehran_housing_best_model.pkl')
    features_path = os.path.join(base_path, 'model_features.pkl')
    
    model = joblib.load(model_path)
    model_features = joblib.load(features_path)

    addresses = [col.replace('Address_', '') for col in model_features if col.startswith('Address_')]
    addresses = sorted(addresses)

    return model, model_features, addresses

model, model_features, addresses = load_assets()

# Center all content using a middle column layout
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    # Banner
    url = "https://media.cntraveler.com/photos/6124643cc9e624849c7a44bd/16:9/w_2240,c_limit/LICENSE_Mehrdad-Mzadeh-Tehran_(c)-Getty-Images_CNT-UK_Sophie-Knight.jpg"
    st.image(url, use_container_width=True)

    # Title & Description
    st.title("Tehran House Price Prediction")
    st.write("Enter the property details below to calculate the estimated price in USD.")

    st.markdown("---")

    # Form Inputs
    area = st.number_input("Area (m²)", min_value=20, max_value=1000, value=80, step=5)
    room = st.selectbox("Number of Rooms", options=[1, 2, 3, 4, 5], index=2)
    address = st.selectbox("Address", options=addresses)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        parking = st.checkbox("Parking", value=True)
    with col_b:
        warehouse = st.checkbox("Warehouse", value=True)
    with col_c:
        elevator = st.checkbox("Elevator", value=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction Button
    if st.button("Calculate Price", type="primary", use_container_width=True):
        # 1. Initialize input dictionary with zero for all model features
        input_data = {col: 0 for col in model_features}
        
        # 2. Assign values for numerical and boolean features
        input_data['Area'] = area
        input_data['Room'] = room
        input_data['Parking'] = int(parking)
        input_data['Warehouse'] = int(warehouse)
        input_data['Elevator'] = int(elevator)
        
        # 3. Apply one-hot encoding for the selected address
        address_col = f"Address_{address}"
        if address_col in input_data:
            input_data[address_col] = 1
            
        # 4. Convert input dictionary to DataFrame aligned with model features
        input_df = pd.DataFrame([input_data])[model_features]
        
        # 5. Generate prediction
        predicted_price = model.predict(input_df)[0]
        
        # Display Results Centered
        st.success(f" **Estimated Price:** ${predicted_price:,.0f}")