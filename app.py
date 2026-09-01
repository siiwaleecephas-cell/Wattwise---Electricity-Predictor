import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title = 'Wattwise - Electricity Predictor', page_icon = '⚡', layout = 'wide')
st.title('Wattwise - Electricity Predictor')

@st.cache_resource
def load_model():
  model = joblib.load("random_forest_model.joblib")
  return model

model = load_model()

if hasattr(model, 'feature_names_in_'):
  feature_names = model.feature_names_in_
else:
  feature_names = []

with st.expander('expected model features'):
  st.write(feature_names)

user_input = {}

for feature in feature_names:
  f = feature.lower()
if f == 'month':
  user_input[feature] = st.sidebar.selectbox(feature, range(1,13))
elif f == 'day_of_week':
  user_input[feature] = st.sidebar.selectbox(feature, range(7))
elif f == 'is_weekend':
  user_input[feature] = st.sidebar.selectbox(feature, [0,1])
else:
  user_input[feature] = st.sidebar.number_input(feature,value = 0.0,format = "%.2f")


if st.sidebar.button("Predict"):
  input_df = pd.DataFrame([user_input])
  st.write("input sent to model")
  st.dataframe(input_df)

  prediction = model.predict(input_df)[0]
  st.success('Prediction value:', prediction)