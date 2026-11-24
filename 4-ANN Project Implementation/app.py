import pickle

import pandas as pd
import streamlit as st
import tensorflow as tf


# ---------- Load artifacts (cached so they load once) ----------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('model.h5', compile=False)

    with open('label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        onehot_encoder_geo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return model, label_encoder_gender, onehot_encoder_geo, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()


# ---------- UI ----------
st.title('Customer Churn Prediction')
st.subheader('Enter Customer Details')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, 40)
balance = st.number_input('Balance', min_value=0.0, value=60000.0, step=1000.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0, step=1000.0)
tenure = st.slider('Tenure', 0, 10, 3)
num_of_products = st.slider('Number of Products', 1, 4, 2)
has_cr_card = st.selectbox('Has Credit Card', [0, 1], index=1)
is_active_member = st.selectbox('Is Active Member', [0, 1], index=1)


# ---------- Prepare input ----------
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

# One-hot encode Geography (DataFrame input keeps feature names -> no warning)
geo_encoded = onehot_encoder_geo.transform(
    pd.DataFrame({'Geography': [geography]})
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography']),
)

input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Match the exact training feature order
input_data = input_data.reindex(columns=scaler.feature_names_in_)

# Scale + predict
input_data_scaled = scaler.transform(input_data)
prediction_proba = float(model.predict(input_data_scaled, verbose=0)[0][0])


# ---------- Results ----------
st.subheader('Prediction Results')
st.metric('Churn Probability', f'{prediction_proba:.2%}')

if prediction_proba > 0.5:
    st.error('The customer is likely to churn.')
else:
    st.success('The customer is not likely to churn.')

with st.expander('View processed customer data'):
    st.dataframe(input_data)