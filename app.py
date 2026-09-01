import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("house_price_model.pkl")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Predictor")
st.write("Based on paper - 8 key features. Trained with RandomForest (RMSE ~ 29k)")

# Inputs
col1, col2 = st.columns(2)

with col1:
    OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 7)
    GrLivArea = st.number_input("Living Area (sq ft)", 500, 5000, 1710)
    GarageCars = st.slider("Garage Cars", 0, 4, 2)
    GarageArea = st.number_input("Garage Area (sq ft)", 0, 1500, 548)

with col2:
    TotalBsmtSF = st.number_input("Basement Area (sq ft)", 0, 3000, 856)
    FirstFlrSF = st.number_input("1st Floor Area (sq ft)", 0, 4000, 856)
    FullBath = st.slider("Full Bathrooms", 0, 4, 2)
    YearBuilt = st.number_input("Year Built", 1870, 2025, 2003)

if st.button("Predict Price", type="primary"):
    data = pd.DataFrame([{
        'OverallQual': OverallQual,
        'GrLivArea': GrLivArea,
        'GarageCars': GarageCars,
        'GarageArea': GarageArea,
        'TotalBsmtSF': TotalBsmtSF,
        '1stFlrSF': FirstFlrSF,
        'FullBath': FullBath,
        'YearBuilt': YearBuilt
    }])
    
    price = model.predict(data)[0]
    st.success(f"### Estimated Price: ${price:,.2f}")
    st.balloons()

st.markdown("---")
st.caption("Built with Streamlit • Free deployment")
