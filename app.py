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
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except requests.RequestException:
        return None

model = load_model()

# Animation is optional: the app still works if it cannot be loaded
lottie_factory = load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_vybexzxa.json')

# ==========================================
# 2. TRAINING RANGES (from the dataset the model was trained on)
# ==========================================
# Tree-based models cannot extrapolate: outside these ranges the prediction
# stays at the value given at the edge of the training data. Inputs are
# therefore limited to the ranges below.
RANGES = {
    'temperature_C':            (-8.0, 35.0),
    'relative_humidity_pct':    (55.0, 95.0),
    'SO2_deposition_mg_m2_day': (0.7, 150.0),
    'Cl_deposition_mg_m2_day':  (0.4, 300.0),
    'exposure_time_years':      (1.0, 10.0),
    'Cu_content_wt_pct':        (0.01, 0.20),
    'Cr_content_wt_pct':        (0.01, 0.15),
    'Ni_content_wt_pct':        (0.01, 0.15),
    'P_content_wt_pct':         (0.005, 0.045),
    'Si_content_wt_pct':        (0.10, 0.35),
    'Mn_content_wt_pct':        (0.30, 0.90),
}

# ISO 9223 limits for the FIRST-YEAR corrosion loss of carbon steel (µm)
ISO_LIMITS = [
    (1.3,   "C1", "Very Low"),
    (25.0,  "C2", "Low"),
    (50.0,  "C3", "Medium"),
    (80.0,  "C4", "High"),
    (200.0, "C5", "Very High"),
]

def iso_category(first_year_loss_um):
    for upper, category, severity in ISO_LIMITS:
        if first_year_loss_um <= upper:
            return category, severity
    return "CX", "Extreme"

# ==========================================
# 3. APP HEADER & UI
# ==========================================
col_header1, col_header2 = st.columns([1, 4])
with col_header1:
    if lottie_factory:
        st_lottie(lottie_factory, height=150, key="factory")
with col_header2:
    st.title("Mild Carbon Steel Corrosion Predictor")
    st.markdown("""
    This application predicts the **average corrosion rate** and **cumulative thickness loss** of mild carbon steel.
    It uses a Gradient Boosting machine learning model trained on physics-grounded environmental and alloy data
    (ISO 9223 / ISO 9224). It is a screening tool and has not been validated against field measurements.
    """)

st.divider()

# ==========================================
# 4. SIDEBAR: USER INPUTS
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2083/2083204.png", width=100)
st.sidebar.header("⚙️ Set Service Conditions")
st.sidebar.caption("Inputs are limited to the ranges the model was trained on.")

with st.sidebar.expander("🌤️ Environmental Parameters", expanded=True):
    lo, hi = RANGES['temperature_C']
    temp = st.number_input("Temperature (°C)", min_value=lo, max_value=hi, value=25.0)
    lo, hi = RANGES['relative_humidity_pct']
    rh = st.number_input("Relative Humidity (%)", min_value=lo, max_value=hi, value=82.0)
    lo, hi = RANGES['SO2_deposition_mg_m2_day']
    so2 = st.number_input("SO₂ Deposition (mg/m²/day)", min_value=lo, max_value=hi, value=45.0)
    lo, hi = RANGES['Cl_deposition_mg_m2_day']
    cl = st.number_input("Chloride Deposition (mg/m²/day)", min_value=lo, max_value=hi, value=120.0)
    lo, hi = RANGES['exposure_time_years']
    time_yrs = st.number_input("Exposure Time (Years)", min_value=lo, max_value=hi, value=5.0)

with st.sidebar.expander("🔬 Alloy Composition (wt%)", expanded=False):
    cu = st.slider("Copper (Cu)", *RANGES['Cu_content_wt_pct'], 0.10)
    cr = st.slider("Chromium (Cr)", *RANGES['Cr_content_wt_pct'], 0.08)
    ni = st.slider("Nickel (Ni)", *RANGES['Ni_content_wt_pct'], 0.08)
    p = st.slider("Phosphorus (P)", *RANGES['P_content_wt_pct'], 0.02)
    si = st.slider("Silicon (Si)", *RANGES['Si_content_wt_pct'], 0.20)
    mn = st.slider("Manganese (Mn)", *RANGES['Mn_content_wt_pct'], 0.60)

# Format input mapping for prediction
input_features = list(RANGES.keys())
input_data = pd.DataFrame([[temp, rh, so2, cl, time_yrs, cu, cr, ni, p, si, mn]], columns=input_features)

# Safety check: warn if any value is outside the training range
out_of_range = [name for name in input_features
                if not (RANGES[name][0] <= input_data[name].iloc[0] <= RANGES[name][1])]

# ==========================================
# 5. PREDICTION & TABS
# ==========================================
tab1, tab2 = st.tabs(["📊 Prediction Results", "🧠 Model Insights (Feature Importance)"])

with tab1:
    st.subheader("Estimated Corrosion Severity")

    if out_of_range:
        st.warning("These inputs are outside the range the model was trained on, so the "
                   "prediction is unreliable: " + ", ".join(out_of_range))

    # Generate prediction
    predicted_rate = model.predict(input_data)[0]            # average rate, mm/year
    predicted_loss = predicted_rate * time_yrs * 1000        # cumulative loss, µm

    # ISO 9223 classifies on the FIRST-YEAR loss. With the ISO 9224 power law
    # C(t) = C1 * t^0.5, the first-year loss is C(t) / sqrt(t).
    first_year_loss = predicted_loss / (time_yrs ** 0.5)     # µm
    category, severity = iso_category(first_year_loss)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Average Corrosion Rate", value=f"{predicted_rate:.4f} mm/yr")
    col2.metric(label=f"Cumulative Loss ({time_yrs:g} yrs)", value=f"{predicted_loss:.1f} µm")
    col3.metric(label="Estimated First-Year Loss", value=f"{first_year_loss:.1f} µm")
    col4.metric(label="ISO 9223 Corrosivity Category", value=category, delta=severity, delta_color="off")

    st.info(f"**Interpretation:** The estimated first-year loss is {first_year_loss:.1f} µm, which places this "
            f"environment in category **{category} ({severity})** of ISO 9223. The total thickness lost after "
            f"{time_yrs:g} years is estimated to be {predicted_loss:.1f} µm.")
    st.caption("The category is found by converting the predicted average rate back to a first-year loss, "
               "as ISO 9223 defines it. Predictions come from a model trained on ISO-based generated data.")

with tab2:
    st.subheader("What is driving this prediction?")
    st.write("This chart shows the internal importance of each variable in the Gradient Boosting model. "
             "It describes how the model was built, not the effect of a change in one input.")

    importances = model.feature_importances_
    display_names = ['Temperature', 'Relative Humidity', 'SO₂ Deposition', 'Cl⁻ Deposition',
                     'Exposure Time', 'Copper (Cu)', 'Chromium (Cr)', 'Nickel (Ni)',
                     'Phosphorus (P)', 'Silicon (Si)', 'Manganese (Mn)']

    imp_df = pd.DataFrame({'Feature': display_names, 'Importance': importances})
    imp_df = imp_df.sort_values(by='Importance', ascending=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=imp_df,
                palette='viridis', legend=False, ax=ax)
    ax.set_xlabel('Relative Importance (Contribution to Prediction)')
    ax.set_ylabel('')

    st.pyplot(fig)