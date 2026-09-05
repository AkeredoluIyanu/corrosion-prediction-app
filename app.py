import streamlit as st
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_lottie import st_lottie

# ==========================================
# 1. PAGE CONFIGURATION & CACHING
# ==========================================
st.set_page_config(page_title="Corrosion Predictor", page_icon="⚙️", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('corrosion_gb_model.pkl')

@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

model = load_model()

# Load a cool factory/engineering animation from Lottie
lottie_factory = load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_vybexzxa.json')

# ==========================================
# 2. APP HEADER & UI
# ==========================================
col_header1, col_header2 = st.columns([1, 4])
with col_header1:
    if lottie_factory:
        st_lottie(lottie_factory, height=150, key="factory")
with col_header2:
    st.title("Mild Carbon Steel Corrosion Predictor")
    st.markdown("""
    This application predicts the **annual corrosion rate** and **cumulative mass loss** of mild carbon steel.
    It uses a Gradient Boosting machine learning model trained on physics-grounded environmental and alloy data.
    """)

st.divider()

# ==========================================
# 3. SIDEBAR: USER INPUTS
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2083/2083204.png", width=100) # Optional aesthetic icon
st.sidebar.header("⚙️ Set Service Conditions")

with st.sidebar.expander("🌤️ Environmental Parameters", expanded=True):
    temp = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0)
    rh = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=82.0)
    so2 = st.number_input("SO₂ Deposition (mg/m²/day)", min_value=0.0, max_value=200.0, value=45.0)
    cl = st.number_input("Chloride Deposition (mg/m²/day)", min_value=0.0, max_value=500.0, value=120.0)
    time_yrs = st.number_input("Exposure Time (Years)", min_value=0.1, max_value=50.0, value=5.0)

with st.sidebar.expander("🔬 Alloy Composition (wt%)", expanded=False):
    cu = st.slider("Copper (Cu)", 0.0, 0.5, 0.10)
    cr = st.slider("Chromium (Cr)", 0.0, 1.2, 0.08)
    ni = st.slider("Nickel (Ni)", 0.0, 1.0, 0.08)
    p = st.slider("Phosphorus (P)", 0.0, 0.1, 0.02)
    si = st.slider("Silicon (Si)", 0.0, 1.0, 0.20)
    mn = st.slider("Manganese (Mn)", 0.0, 2.0, 0.60)

# Format input mapping for prediction
input_features = ['temperature_C', 'relative_humidity_pct', 'SO2_deposition_mg_m2_day', 
                  'Cl_deposition_mg_m2_day', 'exposure_time_years', 'Cu_content_wt_pct', 
                  'Cr_content_wt_pct', 'Ni_content_wt_pct', 'P_content_wt_pct', 
                  'Si_content_wt_pct', 'Mn_content_wt_pct']

input_data = pd.DataFrame([[temp, rh, so2, cl, time_yrs, cu, cr, ni, p, si, mn]], columns=input_features)

# ==========================================
# 4. PREDICTION & TABS
# ==========================================
# We use st.tabs to cleanly separate the output from the model explanation
tab1, tab2 = st.tabs(["📊 Prediction Results", "🧠 Model Insights (Feature Importance)"])

with tab1:
    st.subheader("Estimated Corrosion Severity")
    
    # Generate Prediction
    predicted_rate = model.predict(input_data)[0]
    predicted_loss = predicted_rate * time_yrs * 1000  # Convert to µm
    
    # Create beautiful metric cards
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Average Corrosion Rate", value=f"{predicted_rate:.4f} mm/yr")
    col2.metric(label=f"Cumulative Loss ({time_yrs} yrs)", value=f"{predicted_loss:.1f} µm")
    
    # Determine ISO Category dynamically based on rate
    if predicted_rate <= 0.0013: category, severity = "C1", "Very Low"
    elif predicted_rate <= 0.025: category, severity = "C2", "Low"
    elif predicted_rate <= 0.05: category, severity = "C3", "Medium"
    elif predicted_rate <= 0.08: category, severity = "C4", "High"
    elif predicted_rate <= 0.2: category, severity = "C5", "Very High"
    else: category, severity = "CX", "Extreme"
    
    col3.metric(label="ISO 9223 Corrosivity Category", value=category, delta=severity, delta_color="off")
    
    st.info(f"**Interpretation:** Based on the environmental inputs, this environment falls into the **{category} ({severity})** category. The total material thickness lost after {time_yrs} years is estimated to be {predicted_loss:.1f} micrometers.")

with tab2:
    st.subheader("What is driving this prediction?")
    st.write("This chart extracts the internal decision weights directly from the Gradient Boosting model to show which variables have the largest impact on the corrosion rate.")
    
    # Extract Feature Importances directly from the loaded model
    importances = model.feature_importances_
    display_names = ['Temperature', 'Relative Humidity', 'SO₂ Deposition', 'Cl⁻ Deposition',
                     'Exposure Time', 'Copper (Cu)', 'Chromium (Cr)', 'Nickel (Ni)', 
                     'Phosphorus (P)', 'Silicon (Si)', 'Manganese (Mn)']
    
    imp_df = pd.DataFrame({'Feature': display_names, 'Importance': importances})
    imp_df = imp_df.sort_values(by='Importance', ascending=False)
    
    # Plotting
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis', ax=ax)
    ax.set_xlabel('Relative Importance (Contribution to Prediction)')
    ax.set_ylabel('')
    
    st.pyplot(fig)