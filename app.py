import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Corrosion Rate Predictor", layout="wide")

# 2. Load the trained model
# Ensure 'corrosion_gb_model.pkl' is in the same directory as this script
@st.cache_resource
def load_model():
    return joblib.load('corrosion_gb_model.pkl')

model = load_model()

# 3. App Header
st.title("Mild Carbon Steel Corrosion Predictor")
st.write("""
This application predicts the **annual corrosion rate** and **cumulative mass loss** of mild carbon steel 
based on environmental conditions and alloy composition. 
Adjust the parameters in the sidebar to simulate different exposure environments.
""")

# 4. User Inputs (Sidebar)
st.sidebar.header("Environmental Parameters")
temp = st.sidebar.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0)
rh = st.sidebar.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=82.0)
so2 = st.sidebar.number_input("SO₂ Deposition (mg/m²/day)", min_value=0.0, max_value=200.0, value=45.0)
cl = st.sidebar.number_input("Chloride Deposition (mg/m²/day)", min_value=0.0, max_value=500.0, value=120.0)
time_yrs = st.sidebar.number_input("Exposure Time (Years)", min_value=0.1, max_value=50.0, value=5.0)

st.sidebar.header("Alloy Composition (wt%)")
cu = st.sidebar.slider("Copper (Cu)", 0.0, 0.5, 0.1)
cr = st.sidebar.slider("Chromium (Cr)", 0.0, 1.2, 0.08)
ni = st.sidebar.slider("Nickel (Ni)", 0.0, 1.0, 0.08)
p = st.sidebar.slider("Phosphorus (P)", 0.0, 0.1, 0.02)
si = st.sidebar.slider("Silicon (Si)", 0.0, 1.0, 0.2)
mn = st.sidebar.slider("Manganese (Mn)", 0.0, 2.0, 0.6)

# 5. Format input for the model
input_data = pd.DataFrame([{
    'temperature_C': temp,
    'relative_humidity_pct': rh,
    'SO2_deposition_mg_m2_day': so2,
    'Cl_deposition_mg_m2_day': cl,
    'exposure_time_years': time_yrs,
    'Cu_content_wt_pct': cu,
    'Cr_content_wt_pct': cr,
    'Ni_content_wt_pct': ni,
    'P_content_wt_pct': p,
    'Si_content_wt_pct': si,
    'Mn_content_wt_pct': mn
}])

# 6. Prediction & Output Display
if st.sidebar.button("Predict Corrosion"):
    # Run prediction
    predicted_rate = model.predict(input_data)[0]
    predicted_loss = predicted_rate * time_yrs * 1000  # Convert to µm
    
    # Display Results in Columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"### Predicted Average Rate\n# {predicted_rate:.4f} mm/yr")
        st.write("This is the average material loss per year over the specified exposure period.")
        
    with col2:
        st.error(f"### Cumulative Material Loss\n# {predicted_loss:.2f} µm")
        st.write(f"This is the total estimated material loss after {time_yrs} years of exposure.")