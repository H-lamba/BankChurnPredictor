import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd 
import pickle 

model = tf.keras.models.load_model('model.h5')  

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('ohe_geography.pkl', 'rb') as f:
    ohe = pickle.load(f)   
st.title("Bank Customer Churn Prediction")

geography = st.selectbox("Select Geography", ohe.categories_[0])
gender = st.selectbox('Gender', encoder.classes_)
age = st.slider('Age',18,100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card', ['1', '0'])    
is_active_member = st.selectbox('Is Active Member', ['1', '0'])

input_data = pd.DateFrame({
    'CreditScore': [credit_score],
    'Gender' : [encoder.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]})

geo_encoded = ohe.transform([['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)

predction = model.predict(input_data_scaled)
predction_proba = predction[0][0]
if st.button('Predict'):
    if predction_proba > 0.5:
        st.success(f'The customer is likely to churn with a probability of {predction_proba:.2f}')
    else:
        st.success(f'The customer is unlikely to churn with a probability of {1 - predction_proba:.2f}')

