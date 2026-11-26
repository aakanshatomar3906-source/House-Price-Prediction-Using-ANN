import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers

# 1. Page Configuration
st.set_page_config(
    page_title="NestWorth AI | House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Exact Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_ann.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "min_max_scaler.pkl")
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "housing.csv")

# 3. Custom CSS for Modern UI, Glassmorphism & Smooth Cards
st.markdown("""
    <style>
    .hero-banner {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2.2rem;
        border-radius: 18px;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        margin-top: 0.6rem;
        font-size: 1.1rem;
        color: #d0e1e9;
    }
    .input-header {
        background: rgba(255, 255, 255, 0.04);
        padding: 0.6rem 1rem;
        border-radius: 10px;
        border-left: 4px solid #00c6ff;
        margin-bottom: 1rem;
        font-weight: 600;
        font-size: 1.15rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        padding: 2.2rem;
        border-radius: 18px;
        color: #081c15;
        text-align: center;
        box-shadow: 0 12px 32px rgba(0, 176, 155, 0.3);
        margin-bottom: 1rem;
    }
    .prediction-card .label {
        font-size: 1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        opacity: 0.85;
    }
    .prediction-card .price {
        font-size: 3.3rem;
        font-weight: 900;
        margin: 0.4rem 0;
        line-height: 1.1;
    }
    .prediction-card .inr {
        font-size: 1.1rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.35);
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        margin-top: 0.4rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        font-weight: 700;
        font-size: 1.15rem;
        padding: 0.8rem;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 114, 255, 0.45);
    }
    </style>
""", unsafe_allow_html=True)

# 4. Load Model Weights & Scaler (Bypasses Keras 3.13 quantization_config mismatch)
@st.cache_resource
def load_artifacts():
    model = tf.keras.Sequential([
        layers.Input(shape=(9,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.load_weights(MODEL_PATH)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(
        f"⚠️ Files load nahi ho payi!\n\n"
        f"Expected paths:\n- `{MODEL_PATH}`\n- `{SCALER_PATH}`\n\nError: {e}"
    )
    st.stop()

# 5. Hero Header
st.markdown("""
    <div class="hero-banner">
        <h1>🏡 NestWorth AI — House Price Predictor</h1>
        <p>Deep Learning (ANN) powered real estate valuation for California properties ✨</p>
    </div>
""", unsafe_allow_html=True)

# 6. Sidebar - Model Insights & Controls
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/619/619153.png", width=70)
    st.title("Model Control Panel")
    st.success("✅ Loaded from `models/`")
    
    st.markdown("---")
    st.subheader("⚙️ Preprocessing Setting")
    use_scaler = st.toggle(
        "Apply MinMaxScaler", 
        value=True, 
        help="Scaled data par trained model ke liye isko ON rakhein."
    )
    
    usd_to_inr = st.number_input("USD to INR Rate (₹)", min_value=70.0, max_value=100.0, value=84.5, step=0.5)

    st.markdown("---")
    st.subheader("🧠 ANN Architecture")
    st.code("Input (9 Features)\n  ↓\nDense(64, ReLU)\n  ↓\nDense(32, ReLU)\n  ↓\nDense(16, ReLU)\n  ↓\nOutput(1, Linear)", language="text")

# 7. Navigation Tabs
tab1, tab2 = st.tabs(["🎯 Single Property Valuation", "📂 Batch CSV Prediction"])

with tab1:
    st.caption("⚡ Quick Test Presets (Ek click mein sample values bharne ke liye):")
    p_col1, p_col2, p_col3 = st.columns(3)
    
    if "preset" not in st.session_state:
        st.session_state.preset = "mid"

    if p_col1.button("🌊 Bay Area Luxury Home"):
        st.session_state.preset = "luxury"
    if p_col2.button("🏙️ LA Suburb Family Home"):
        st.session_state.preset = "mid"
    if p_col3.button("🌾 Inland Budget Home"):
        st.session_state.preset = "budget"

    presets = {
        "luxury": {"ocean": 3, "lat": 37.85, "lon": -122.24, "age": 35, "rooms": 3200, "beds": 520, "pop": 950, "hh": 480, "inc": 9.5},
        "mid":    {"ocean": 0, "lat": 34.05, "lon": -118.24, "age": 25, "rooms": 2200, "beds": 430, "pop": 1200, "hh": 400, "inc": 4.8},
        "budget": {"ocean": 1, "lat": 36.75, "lon": -119.77, "age": 18, "rooms": 1650, "beds": 360, "pop": 1450, "hh": 340, "inc": 2.3},
    }
    cur = presets[st.session_state.preset]

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown('<div class="input-header">📍 Location & Coastline</div>', unsafe_allow_html=True)
        ocean_options = {
            "<1H OCEAN (Samundar se < 1 hr)": 0,
            "INLAND (Andaruni ilaka)": 1,
            "NEAR OCEAN (Samundar ke paas)": 2,
            "NEAR BAY (Bay area ke paas)": 3,
            "ISLAND (Tapu par)": 4
        }
        ocean_label = st.selectbox(
            "Ocean Proximity", 
            list(ocean_options.keys()), 
            index=cur["ocean"]
        )
        ocean_proximity = ocean_options[ocean_label]

        latitude = st.slider("Latitude (North-South)", 32.50, 42.00, float(cur["lat"]), 0.01)
        longitude = st.slider("Longitude (East-West)", -124.35, -114.30, float(cur["lon"]), 0.01)

    with col2:
        st.markdown('<div class="input-header">🏠 House & Block Specs</div>', unsafe_allow_html=True)
        housing_median_age = st.slider("Building Median Age (Years)", 1, 52, int(cur["age"]))
        total_rooms = st.number_input("Total Rooms (in Block)", 2, 40000, int(cur["rooms"]), step=50)
        total_bedrooms = st.number_input("Total Bedrooms (in Block)", 1, 6500, int(cur["beds"]), step=10)

    with col3:
        st.markdown('<div class="input-header">👨‍👩‍👧‍👦 Neighborhood & Income</div>', unsafe_allow_html=True)
        population = st.number_input("Block Population", 3, 36000, int(cur["pop"]), step=50)
        households = st.number_input("Total Households", 1, 6100, int(cur["hh"]), step=10)
        median_income = st.slider(
            "Median Income ($10,000s)", 
            0.5, 15.0, float(cur["inc"]), 0.1,
            help="Jaise 5.0 ka matlab $50,000 annual income hai"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Calculate Estimated Price"):
        raw_features = np.array([[
            longitude,
            latitude,
            housing_median_age,
            total_rooms,
            total_bedrooms,
            population,
            households,
            median_income,
            ocean_proximity
        ]], dtype=np.float32)

        final_features = scaler.transform(raw_features) if use_scaler else raw_features
        pred_price = max(0.0, float(model.predict(final_features, verbose=0)[0][0]))
        inr_lakhs = (pred_price * usd_to_inr) / 100000

        st.markdown("---")
        res_left, res_right = st.columns([1.3, 1], gap="large")

        with res_left:
            st.markdown(f"""
                <div class="prediction-card">
                    <div class="label">Estimated Property Value</div>
                    <div class="price">${pred_price:,.2f}</div>
                    <div class="inr">🇮🇳 Approx. ₹{inr_lakhs:,.2f} Lakhs</div>
                </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Rooms / House", f"{total_rooms / max(1, households):.1f}")
            m2.metric("People / House", f"{population / max(1, households):.1f}")
            m3.metric("Annual Income", f"${median_income * 10000:,.0f}")

            if pred_price >= 350000:
                st.success("🌟 **Prime Luxury Segment:** Yeh high-demand area hai! High median income aur coastal proximity iski value ko boost kar rahe hain.")
            elif pred_price >= 180000:
                st.info("👍 **Solid Mid-Range Value:** Yeh balanced residential zone hai—families aur steady appreciation ke liye badhiya choice.")
            else:
                st.warning("💰 **Affordable / Budget Zone:** Yeh entry-level price bracket hai—first-time buyers ke liye kaafi pocket-friendly option.")

        with res_right:
            st.subheader("🗺️ Property Pinpoint")
            st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=6)

with tab2:
    st.subheader("📂 Batch House Price Predictions")
    st.write("Apni koi CSV upload karein ya direct `Dataset/housing.csv` ko test karein:")

    source_choice = st.radio(
        "Data Source Chunein:",
        ["Use Project Dataset (`Dataset/housing.csv`)", "Upload Custom CSV"],
        horizontal=True
    )

    batch_df = None
    if source_choice == "Use Project Dataset (`Dataset/housing.csv`)":
        if os.path.exists(DEFAULT_DATASET_PATH):
            batch_df = pd.read_csv(DEFAULT_DATASET_PATH)
        else:
            st.warning(f"`{DEFAULT_DATASET_PATH}` nahi mili.")
    else:
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)

    if batch_df is not None:
        st.write("**Data Preview:**", batch_df.head(5))

        required_cols = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'
        ]

        if all(col in batch_df.columns for col in required_cols):
            if st.button("⚡ Predict All Rows"):
                proc_df = batch_df[required_cols].copy().dropna()
                
                if proc_df['ocean_proximity'].dtype == 'object':
                    mapping = {'<1H OCEAN': 0, 'INLAND': 1, 'NEAR OCEAN': 2, 'NEAR BAY': 3, 'ISLAND': 4}
                    proc_df['ocean_proximity'] = proc_df['ocean_proximity'].map(mapping)

                X_batch = proc_df.values.astype(np.float32)
                if use_scaler:
                    X_batch = scaler.transform(X_batch)

                preds = model.predict(X_batch, verbose=0).flatten()
                result_df = batch_df.loc[proc_df.index].copy()
                result_df['Predicted_House_Value ($)'] = np.round(np.maximum(0, preds), 2)

                st.success(f"✅ {len(result_df)} properties ke prices successfully predict ho gaye!")
                st.dataframe(result_df.head(15), use_container_width=True)

                csv_out = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_out,
                    file_name="predicted_house_prices.csv",
                    mime="text/csv"
                )
        else:
            st.error(f"CSV mein ye columns hone chahiye: `{required_cols}`")