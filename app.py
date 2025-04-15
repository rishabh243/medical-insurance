
import streamlit as st
import pickle
import json
import numpy as np

# Load the trained model and metadata
@st.cache(allow_output_mutation=True)
def load_model_and_data(model_path='linear_regression.pkl', data_path='proj_data.json'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(data_path, 'r') as f:
        project_data = json.load(f)
    return model, project_data

model, project_data = load_model_and_data()

# Streamlit UI
st.title('Medical Insurance Charge Predictor')
st.write('Provide the following details to estimate the insurance charge:')

# Input widgets
age = st.slider('Age', min_value=18, max_value=100, value=30)
gender = st.selectbox('Gender', list(project_data['Gender'].keys()))
bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=28.265, format="%.2f")
children = st.slider('Number of Children', min_value=0, max_value=10, value=0)
smoker = st.selectbox('Smoker Status', list(project_data['Smoker'].keys()))
region_choices = [col.replace('region_', '') for col in project_data['Column Names'] if col.startswith('region_')]
region = st.selectbox('Region', region_choices)

# Predict button
if st.button('Predict Insurance Charge'):
    # Encode inputs
    gender_num = project_data['Gender'][gender]
    smoker_num = project_data['Smoker'][smoker]
    # Build feature array
    features = np.zeros((1, model.n_features_in_))
    features[0, 0] = age
    features[0, 1] = gender_num
    features[0, 2] = bmi
    features[0, 3] = children
    features[0, 4] = smoker_num
    # Set region dummy
    region_col = f'region_{region}'
    region_index = project_data['Column Names'].index(region_col)
    features[0, region_index] = 1

    # Make prediction
    prediction = model.predict(features)[0]
    st.success(f'Estimated Annual Insurance Charge: ${prediction:,.2f}')

st.write('---')
st.write('**Note:** This prediction is based on a linear regression model trained on the Medical Insurance dataset.')
