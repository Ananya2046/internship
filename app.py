import streamlit as st
from src.predict import predict_crop_production

# Configure a clean, dark-mode friendly layout page
st.set_page_config(page_title="Agri-Production Predictor", layout="centered")

st.title("🌾 Agricultural Crop Yield Prediction Engine")
st.write("Fill in the fields manually to calculate machine learning production forecasts dynamically.")
st.markdown("---")

# Layout input elements side-by-side using columns
col1, col2 = st.columns(2)

with col1:
    crop = st.selectbox("Select Crop Type", ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"])
    state = st.selectbox("Select State", ["Uttar Pradesh", "Andhra Pradesh", "Punjab", "Haryana", "Tamil Nadu"])
    season = st.text_input("Seasonal Cycle / Duration Code", value="152")

with col2:
    cost_a2_fl = st.number_input("Cost of Cultivation A2+FL (Rs.)", min_value=0.0, value=9794.05)
    cost_c2 = st.number_input("Cost of Cultivation C2 (Rs.)", min_value=0.0, value=23076.74)
    cost_prod = st.number_input("Cost of Production C2 (Per Unit)", min_value=0.0, value=1941.55)

st.markdown("---")

# Submit button to trigger prediction backend
if st.button("Calculate Production Output", type="primary"):
    try:
        # Passes manual user input values directly to the prediction logic
        predicted_yield = predict_crop_production(crop, state, season, cost_a2_fl, cost_c2, cost_prod)
        st.success(f"### Predicted Production: **{predicted_yield:.4f} Tonnes**")
    except Exception as e:
        st.error(f"Inference Engine Error: {e}")
