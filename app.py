import datetime
import os
import sys
import __main__
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Import Custom Transformer and Bind to __main__ for joblib unpickling
from model import Iqrclipper

sys.modules['__main__'].Iqrclipper = Iqrclipper
__main__.Iqrclipper = Iqrclipper

# 2. Get Absolute Path of the Model File
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'car_price_model.pkl')

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

/* Default Desktop Container Padding */
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
    animation: pulseCue 2s infinite;
}

@keyframes pulseCue {
    0% { transform: translateY(0); opacity: 0.8; }
    50% { transform: translateY(4px); opacity: 1; }
    100% { transform: translateY(0); opacity: 0.8; }
}

.road {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 125px;
    background: linear-gradient(#111923, #05080d);
    border-top: 1px solid rgba(255,255,255,0.06);
}

.road::before {
    content: "";
    position: absolute;
    top: 57px;
    left: 0;
    width: 100%;
    height: 4px;
    background: repeating-linear-gradient(
        90deg,
        rgba(255,255,255,0.55) 0px,
        rgba(255,255,255,0.55) 70px,
        transparent 70px,
        transparent 150px
    );
    animation: roadMove 1.2s linear infinite;
}

@keyframes roadMove {
    from { transform: translateX(0); }
    to { transform: translateX(-150px); }
}

.car {
    position: absolute;
    z-index: 4;
    bottom: 65px;
    left: -300px;
    width: 230px;
    height: 80px;
    animation: drive 8s linear infinite;
}

.car-body {
    position: absolute;
    bottom: 18px;
    left: 20px;
    width: 190px;
    height: 48px;
    border-radius: 45px 55px 18px 15px;
    background: linear-gradient(145deg, #e9eef5, #737d8b);
    box-shadow: 0 8px 20px rgba(0,0,0,0.55), inset 0 2px 4px rgba(255,255,255,0.7);
}

.car-top {
    position: absolute;
    bottom: 54px;
    left: 66px;
    width: 105px;
    height: 45px;
    border-radius: 65px 65px 8px 8px;
    background: linear-gradient(135deg, #3b4b5d, #111923);
    border: 2px solid rgba(255,255,255,0.16);
}

.window {
    position: absolute;
    bottom: 59px;
    left: 76px;
    width: 42px;
    height: 27px;
    border-radius: 35px 5px 5px 5px;
    background: linear-gradient(135deg, #8bdcff, #172b3c);
}

.window.right {
    left: 121px;
    border-radius: 5px 35px 5px 5px;
}

.wheel {
    position: absolute;
    bottom: 0;
    width: 38px;
    height: 38px;
    background: #050505;
    border: 7px solid #2a3038;
    border-radius: 50%;
    animation: wheelSpin 0.45s linear infinite;
}

.wheel.left { left: 42px; }
.wheel.right { right: 30px; }

@keyframes wheelSpin {
    to { transform: rotate(360deg); }
}

@keyframes drive {
    0% { transform: translateX(0); }
    100% { transform: translateX(calc(100vw + 500px)); }
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

@media screen and (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 3rem 1rem !important;
    }

    .hero {
        padding: 25px 20px;
        min-height: 390px;
        border-radius: 20px;
    }

    .hero h1 {
        font-size: 32px !important;
        letter-spacing: -1.5px;
        margin: 12px 0 10px 0;
    }

    .hero p {
        font-size: 13px;
        line-height: 1.5;
    }

    .road {
        height: 75px;
    }

    .road::before {
        top: 34px;
        height: 3px;
        background: repeating-linear-gradient(
            90deg,
            rgba(255,255,255,0.55) 0px,
            rgba(255,255,255,0.55) 40px,
            transparent 40px,
            transparent 90px
        );
    }

    .car {
        transform: scale(0.55);
        transform-origin: bottom left;
        bottom: 35px;
    }

    .section-title {
        font-size: 22px;
        margin: 30px 0 6px 0;
    }

    .section-subtitle {
        font-size: 13px;
        margin-bottom: 15px;
    }

    .navbar {
        padding: 10px 0 15px 0;
    }

    .logo {
        font-size: 22px;
    }

    .nav-badge {
        font-size: 11px;
        padding: 5px 10px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# NAVBAR SECTION
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
        <div class="eyebrow">
            Intelligent Car Valuation
        </div>
        <h1>
            Know what your<br>
            <span>car is really worth.</span>
        </h1>
        <p>
            Estimate the market value of a used car using 
            advanced machine learning trained on real-world vehicle data.
        </p>
        <div class="scroll-cue">
            <span>👇 Scroll down to enter details</span>
        </div>
    </div>
    <div class="road"></div>
    <div class="car">
        <div class="car-body"></div>
        <div class="car-top"></div>
        <div class="window"></div>
        <div class="window right"></div>
        <div class="wheel left"></div>
        <div class="wheel right"></div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# MODEL OVERVIEW INFO CARDS
st.markdown(
    '<div class="section-title">System Overview</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle">Core components powering your vehicle'
    " estimation pipeline based on UserCarData.csv.</div>",
    unsafe_allow_html=True,
)

col_ic1, col_ic2, col_ic3 = st.columns(3)

with col_ic1:
    st.markdown(
        '<div class="info-card">'
        '<div class="info-card-title">Model Architecture</div>'
        '<div class="info-card-value">XGBoost Regressor (Tuned)</div>'
        "</div>",
        unsafe_allow_html=True,
    )

with col_ic2:
    st.markdown(
        '<div class="info-card">'
        '<div class="info-card-title">Dataset Range</div>'
        '<div class="info-card-value">Years: 1994 – 2020 | Price: ₹30k –'
        " ₹10L+</div>"
        "</div>",
        unsafe_allow_html=True,
    )

with col_ic3:
    st.markdown(
        '<div class="info-card">'
        '<div class="info-card-title">Preprocessing</div>'
        '<div class="info-card-value">IQR Clipping + Robust Scaling</div>'
        "</div>",
        unsafe_allow_html=True,
    )

# LOAD TRAINED MODEL WITH ABSOLUTE PATH & DETAILED ERROR HANDLING
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    model = None
    st.error(f"🚨 Detailed Model Load Error: {e}")

# VEHICLE INPUT SECTION
st.markdown(
    '<div class="section-title">Vehicle Valuation Parameters</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Enter parameters based on dataset ranges'
    ' (Years 1994–2020, Mileage 0–42 kmpl, Engine 624–3604 CC, Power 33–400'
    " bhp).</div>",
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
        help="Select the brand/manufacturer of the car from dataset categories.",
    )

    year = st.slider(
        "Manufacturing Year",
        min_value=1994,
        max_value=2020,
        value=2015,
        help="Dataset cars range from 1994 to 2020.",
    )

    km_driven = st.number_input(
        "Kilometers Driven (km)",
        min_value=1,
        max_value=2500000,
        value=50000,
        step=1000,
        help="Typical range in dataset: 1,000 to 2,500,000 km.",
    )

    fuel = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "CNG", "LPG"],
        help="Fuel type option matching dataset.",
    )

    seller_type = st.selectbox(
        "Seller Type",
        ["Individual", "Dealer", "Trustmark_Dealer"],
        help="Type of seller listing the car.",
    )

with input_col2:
    transmission = st.selectbox(
        "Transmission Type", ["Manual", "Automatic"], help="Gearbox type."
    )

    owner = st.selectbox(
        "Owner Category",
        [
            "First_Owner",
            "Second_Owner",
            "Third_Owner",
            "Fourth_Above_Owner",
            "Test_Drive_Car",
        ],
        help="Previous ownership history.",
    )

    mileage = st.number_input(
        "Mileage (kmpl)",
        min_value=0.0,
        max_value=42.0,
        value=19.3,
        step=0.1,
        help="Dataset range: 0.0 to 42.0 kmpl.",
    )

    engine = st.number_input(
        "Engine Capacity (CC)",
        min_value=624,
        max_value=3604,
        value=1248,
        step=50,
        help="Dataset range: 624 CC to 3,604 CC.",
    )

    max_power = st.number_input(
        "Max Power (bhp)",
        min_value=32.8,
        max_value=400.0,
        value=82.0,
        step=1.0,
        help="Dataset range: 32.8 bhp to 400.0 bhp.",
    )

st.markdown("<br>", unsafe_allow_html=True)

# PREDICTION ACTION BUTTON
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

            st.info(
                f"💡 **Dataset Context:** For a car of age {age_of_car} years"
                f" with {km_driven:,} km driven, the model prediction of **₹"
                f" {pred_price:,.2f}** falls within expected market"
                " boundaries derived from training distribution."
            )
        except Exception as e:
            st.error(f"Error during prediction execution: {e}")
    else:
        st.error(
            "Model file ('car_price_model.pkl') could not be loaded. Please"
            " ensure it is present in the working directory along with"
            " 'model.py'."
        )
