import datetime
import os
import sys
import __main__
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Custom Transformer Class for Unpickling
from model import Iqrclipper

sys.modules["__main__"].Iqrclipper = Iqrclipper
__main__.Iqrclipper = Iqrclipper

# 2. Get Absolute Path of the Model File
MODEL_PATH = os.path.join(os.path.dirname(__file__), "car_price_model.pkl")

# PAGE CONFIGURATION
st.set_page_config(
    page_title="AutoValue — Car Price Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CUSTOM STYLING & RESPONSIVE ANIMATIONS
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: 
        radial-gradient(circle at 20% 20%, rgba(0, 180, 255, 0.10), transparent 30%),
        radial-gradient(circle at 80% 80%, rgba(255, 70, 70, 0.08), transparent 30%),
        #070b12;
    color: white;
    overflow-x: hidden;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

.block-container {
    padding: 2rem 5rem 4rem 5rem;
    max-width: 1400px;
}

.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 0 30px 0;
}

.logo {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -1px;
}

.logo span {
    color: #3abff8;
}

.nav-badge {
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.04);
    padding: 8px 15px;
    border-radius: 50px;
    color: #aeb8c7;
    font-size: 13px;
}

.hero {
    position: relative;
    overflow: hidden;
    min-height: 460px;
    border-radius: 28px;
    border: 1px solid rgba(255,255,255,0.09);
    background: 
        linear-gradient(110deg, rgba(5,10,18,0.98), rgba(8,19,32,0.82)),
        #0a111c;
    box-shadow: 0 30px 80px rgba(0,0,0,0.45);
    padding: 60px;
    margin-bottom: 40px;
}

.hero-content {
    position: relative;
    z-index: 5;
    max-width: 650px;
}

.eyebrow {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 50px;
    background: rgba(58,191,248,0.10);
    border: 1px solid rgba(58,191,248,0.25);
    color: #62d2ff;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.hero h1 {
    font-size: 58px;
    line-height: 1.05;
    margin: 22px 0 18px 0;
    letter-spacing: -3px;
}

.hero h1 span {
    color: #3abff8;
}

.hero p {
    color: #aab6c5;
    font-size: 17px;
    line-height: 1.7;
    max-width: 590px;
}

.scroll-cue {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 25px;
    padding: 8px 16px;
    background: rgba(58, 191, 248, 0.08);
    border: 1px dashed rgba(58, 191, 248, 0.4);
    border-radius: 30px;
    color: #3abff8;
    font-size: 13px;
    font-weight: 500;
}

.section-title {
    margin: 45px 0 8px 0;
    font-size: 28px;
    font-weight: 700;
}

.section-subtitle {
    color: #8793a3;
    margin-bottom: 25px;
    font-size: 15px;
}

.info-card {
    padding: 20px;
    border-radius: 16px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
}

.info-card-title {
    color: #8290a2;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.info-card-value {
    margin-top: 6px;
    font-size: 18px;
    font-weight: 700;
}

div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    background-color: rgba(255, 255, 255, 0.04) !important;
    border-color: rgba(255, 255, 255, 0.1) !important;
    color: white !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# NAVBAR
st.markdown(
    """
<div class="navbar">
    <div class="logo">Auto<span>Value</span></div>
    <div class="nav-badge">AI POWERED · CAR VALUATION</div>
</div>
""",
    unsafe_allow_html=True,
)

# HERO SECTION
st.markdown(
    """
<div class="hero">
    <div class="hero-content">
        <div class="eyebrow">Intelligent Car Valuation</div>
        <h1>Know what your<br><span>car is really worth.</span></h1>
        <p>Estimate the market value of a used car using advanced machine learning.</p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    model = None
    st.error(f"🚨 Model Load Error: {e}")

# INPUT SECTION
st.markdown(
    '<div class="section-title">Vehicle Valuation Parameters</div>',
    unsafe_allow_html=True,
)

input_col1, input_col2 = st.columns(2, gap="large")

with input_col1:
    name = st.selectbox(
        "Car Brand / Name",
        [
            "Maruti",
            "Skoda",
            "Honda",
            "Hyundai",
            "Toyota",
            "Ford",
            "Renault",
            "Mahindra",
            "Tata",
            "Chevrolet",
            "Datsun",
            "Jeep",
            "Mercedes",
            "Mitsubishi",
            "Audi",
            "Volkswagen",
            "BMW",
            "Nissan",
            "Lexus",
            "Jaguar",
            "Land",
            "MG",
            "Volvo",
            "Daewoo",
            "Kia",
            "Fiat",
            "Force",
            "Ambassador",
            "Ashok",
            "Isuzu",
            "Opel",
            "other",
        ],
    )
    year = st.slider("Manufacturing Year", min_value=1994, max_value=2020, value=2015)
    km_driven = st.number_input(
        "Kilometers Driven (km)", min_value=1, max_value=2500000, value=50000, step=1000
    )
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
    seller_type = st.selectbox(
        "Seller Type", ["Individual", "Dealer", "Trustmark_Dealer"]
    )

with input_col2:
    transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
    owner = st.selectbox(
        "Owner Category",
        [
            "First_Owner",
            "Second_Owner",
            "Third_Owner",
            "Fourth_Above_Owner",
            "Test_Drive_Car",
        ],
    )
    mileage = st.number_input(
        "Mileage (kmpl)", min_value=0.0, max_value=42.0, value=19.3, step=0.1
    )
    engine = st.number_input(
        "Engine Capacity (CC)", min_value=624, max_value=3604, value=1248, step=50
    )
    max_power = st.number_input(
        "Max Power (bhp)", min_value=32.8, max_value=400.0, value=82.0, step=1.0
    )

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "Calculate Predicted Price", type="primary", use_container_width=True
)

if predict_button:
    if model is not None:
        current_year = 2020
        age_of_car = current_year - year

        input_data = pd.DataFrame(
            {
                "name": [name],
                "km_driven": [km_driven],
                "fuel": [fuel],
                "seller_type": [seller_type],
                "transmission": [transmission],
                "owner": [owner],
                "mileage": [mileage],
                "engine": [engine],
                "max_power": [max_power],
                "age_of_car": [age_of_car],
            }
        )

        try:
            prediction = model.predict(input_data)
            pred_price = prediction[0]

            st.markdown("<br>", unsafe_allow_html=True)
            st.success(
                f"### Estimated Market Selling Price: ₹ {pred_price:,.2f}"
            )
        except Exception as e:
            st.error(f"Error during prediction execution: {e}")
    else:
        st.error("Model file could not be loaded.")
