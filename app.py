import streamlit as st
import pandas as pd
import joblib

# 1. Load your trained model
model = joblib.load("random_forest_model.joblib")

st.set_page_config(page_title = 'Wattwise - Electricity Predictor', page_icon = '⚡', layout = 'wide')
st.title('Wattwise - Electricity Predictor')


@st.cache_resource
def load_model():
  model = joblib.load("random_forest_model.joblib")
  return model

model = load_model()



# 2. Build input widgets in the sidebar for ALL required features
st.sidebar.header("Input Weather & Temporal Features")

awnd = st.sidebar.number_input("Average Wind Speed (AWND)", value=0.0)
prcp = st.sidebar.number_input("Precipitation (PRCP)", value=0.0)
tmax = st.sidebar.number_input("Max Temperature (TMAX)", value=75.0)
tmin = st.sidebar.number_input("Min Temperature (TMIN)", value=50.0)
day_of_week = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 0)
month = st.sidebar.slider("Month (1-12)", 1, 12, 1)
is_weekend = st.sidebar.selectbox("Is Weekend?", [0, 1])

consumption_lag_1 = st.sidebar.number_input("Consumption Lag 1 Day", value=1500.0)
consumption_lag_7 = st.sidebar.number_input("Consumption Lag 7 Days", value=1500.0)
rolling_mean_7 = st.sidebar.number_input("Rolling Mean 7 Days", value=1500.0)
rolling_max_7 = st.sidebar.number_input("Rolling Max 7 Days", value=1800.0)

# 3. Trigger the prediction matrix
if st.sidebar.button("Predict"):
    # Organize data as a dictionary matching the EXACT name strings seen at fit time
    input_data = {
        'AWND': [awnd],
        'PRCP': [prcp],
        'TMAX': [tmax],
        'TMIN': [tmin],
        'day_of_week': [day_of_week],
        'month': [month],
        'is_weekend': [is_weekend],
        'consumpution_lag_1': [consumption_lag_1],
        'consumpution_lag_7': [consumption_lag_7],
        'rolling_mean_7': [rolling_mean_7],
        'rolling_max_7': [rolling_max_7]
    }
    
    # Convert into a pandas Dataframe
    input_df = pd.DataFrame(input_data)
    
    # CRITICAL STEP: Align the columns to the model's exact fit sequence
    # This automatically matches the array layout to model.feature_names_in_
    input_df = input_df[model.feature_names_in_]
    
    # 4. Execute safe prediction row pipeline
    prediction = model.predict(input_df)[0]
    
    st.success(f"⚡ Predicted Electricity Consumption: {prediction:.2f} kW")





















