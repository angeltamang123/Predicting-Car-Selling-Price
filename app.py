import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# --- Load Model and Data ---
@st.cache_resource
def load_model_and_data():
    """
    Loads the trained model pipeline and the dataset for populating dropdowns.
    Using st.cache_resource to load only once.
    """
    try:
        model = joblib.load('final_random_forest_model.joblib')
        # We need the original dataframe to get the lists for our dropdowns
        df = pd.read_csv('./Data/car_processed.csv')
    except FileNotFoundError:
        st.error("Model or data file not found. Make sure 'final_random_forest_model.joblib' and './Data/car_processed.csv' are in the correct directories.")
        return None, None
    return model, df

model_pipeline, df = load_model_and_data()

# --- Application Title and Description ---
st.title("🚗 Used Car Price Prediction App")
st.markdown("This application predicts the selling price of a used car based on its features. Please provide the details of the car below.")

# --- User Input Fields ---
if model_pipeline and df is not None:
    # Create columns for a cleaner layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Primary Details")
        
        # Brand dropdown (this is the first user choice)
        brands = sorted(df['brand'].unique())
        brand = st.selectbox('Brand', options=brands, key='brand_select')
        
        # Model dropdown (dependent on brand)
        models = sorted(df[df['brand'] == brand]['model'].unique())
        model = st.selectbox('Model', options=models, key='model_select')

        #-- Fetch default specs based on selected model ---
        default_specs = {}
        if brand and model:
            model_df = df[(df['brand'] == brand) & (df['model'] == model)]
            if not model_df.empty:
                default_specs['transmission'] = model_df['transmission'].mode()[0]
                default_specs['fuel'] = model_df['fuel'].mode()[0]
                default_specs['seats'] = int(model_df['seats'].mode()[0])
                default_specs['engine'] = int(model_df['engine'].median())
                default_specs['max_power'] = round(model_df['max_power'].median(), 2)
                default_specs['mileage'] = round(model_df['mileage(km/ltr/kg)'].median(), 2)

    with col2:
        st.header("Usage Details")
        
        # Year of manufacture
        current_year = datetime.now().year
        year = st.number_input('Year of Manufacture', min_value=1980, max_value=current_year, value=2015, step=1)
        car_age = current_year - year if current_year > year else 1 # Ensure age is at least 1
        
        # Kilometers driven
        km_driven = st.number_input('Kilometers Driven', min_value=0, max_value=600000, value=70000, step=1000)
        
        # Owner dropdown
        owner = st.selectbox('Owner', options=sorted(df['owner'].unique()))
        
        # Seller type dropdown
        seller_type = st.selectbox('Seller Type', options=sorted(df['seller_type'].unique()))

    with col3:
        st.header("Technical Specs (Auto-filled)")
        
        # Get the index for the default value to pre-select it in the dropdown
        fuel_options = list(df['fuel'].unique())
        trans_options = list(df['transmission'].unique())
        
        fuel = st.selectbox('Fuel Type', options=fuel_options, 
                            index=fuel_options.index(default_specs.get('fuel', fuel_options[0])), 
                            disabled=True)
        
        transmission = st.selectbox('Transmission', options=trans_options, 
                                    index=trans_options.index(default_specs.get('transmission', trans_options[0])), 
                                    disabled=True)
        
        seats = st.number_input('Seats', min_value=2, max_value=14, 
                                value=default_specs.get('seats', 5), 
                                disabled=True)
        
        engine = st.number_input('Engine Size (CC)', min_value=600, max_value=4000, 
                                 value=default_specs.get('engine', 1200), 
                                 disabled=True)

        max_power = st.number_input('Max Power (bhp)', min_value=30.0, max_value=400.0, 
                                  value=default_specs.get('max_power', 85.0), 
                                  disabled=True)
        
        mileage = st.number_input('Mileage (km/ltr or kg)', min_value=5.0, max_value=45.0, 
                                value=default_specs.get('mileage', 19.0), 
                                disabled=True)

    # --- Prediction Logic ---
    if st.button('Predict Selling Price', use_container_width=True):
        
        # Create km_per_year feature
        km_per_year = km_driven / car_age if car_age > 0 else km_driven

        # Create a DataFrame from the user input in the correct order
        input_data = pd.DataFrame({
            'km_driven': [km_driven],
            'fuel': [fuel],
            'seller_type': [seller_type],
            'transmission': [transmission],
            'owner': [owner],
            'mileage(km/ltr/kg)': [mileage],
            'engine': [engine],
            'max_power': [max_power],
            'seats': [seats],
            'car_age': [car_age],
            'brand': [brand],
            'model': [model],
            'km_per_year': [km_per_year]
        })
        
        try:
            prediction = model_pipeline.predict(input_data)[0]
            st.success(f"**Predicted Selling Price: ₹ {prediction:,.0f}**")
            st.info("Note: This prediction is an estimate based on market data for a typical version of this model. It does not account for the vehicle's specific physical condition or service history.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("Please ensure model and data files are available to run the app.")