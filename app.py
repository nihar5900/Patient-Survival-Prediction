import streamlit as st
import numpy as np
from preprocess import process,get_prediction,ordinal
import joblib
from tensorflow.keras.models import model_from_json

st.set_page_config(page_title="Petient Survival Prediction App",
                   page_icon="🛏", layout="wide")

file=open(r'model/model.json')
model_json=file.read()
file.close()
model=model_from_json(model_json)
model.load_weights("model/model.h5")


dictionary=joblib.load(r'data/directory.pkl')

option_gender=list(dictionary['gender'].keys())
option_icu_type=list(dictionary['icu_type'].keys())
option_ethnicity=list(dictionary['ethnicity'].keys())
option_apache_2_bodysystem=list(dictionary['apache_2_bodysystem'].keys())
option_apache_3j_bodysystem=list(dictionary['apache_3j_bodysystem'].keys())

option_intubated_apache=[0,1]
option_apache_post_operative=[0,1]

st.markdown("<h1 style='text-align: center;'>Petient Survival Prediction App</h1>", unsafe_allow_html=True)
def main():
    with st.form('predictin_form'):
        st.subheader("Enter the below Features:")

        gender=st.selectbox("Gender: ",options=option_gender)
        icu_type=st.selectbox("ICU Type: ",options=option_icu_type)
        ethnicity=st.selectbox("Ethnicity Type: ",options=option_ethnicity)
        apache_2_bodysystem=st.selectbox("APACHE 2 Body System Type: ",options=option_apache_2_bodysystem)
        apache_3j_bodysystem=st.selectbox("APACHE 3j Body System Type: ",options=option_apache_3j_bodysystem)
        intubated_apache=st.selectbox("Intubated Apache OR Not: ",options=option_intubated_apache)
        apache_post_operative=st.selectbox("APACHE 3j Body System Type: ",options=option_apache_post_operative)

        age=st.number_input("Age : ",16,89,format="%d")
        gcs_motor_apache=st.number_input("GCS Motor APACHE : ",1,6,format="%d")
        
        
        submit=st.form_submit_button("Predict")
    if submit:
        gender=ordinal(gender,option_gender)
        icu_type=icu_type
        ethnicity=ethnicity
        apache_2_bodysystem=apache_2_bodysystem
        apache_3j_bodysystem=apache_3j_bodysystem

        intubated_apache=intubated_apache
        apache_post_operative=apache_post_operative

        age=age
        gcs_motor_apache=gcs_motor_apache

        value_1=[intubated_apache,age,apache_post_operative,gcs_motor_apache,gender]
        value_2=[icu_type,ethnicity,apache_2_bodysystem,apache_3j_bodysystem]

        value_3=process(value_2)

        final_values=value_1+value_3
        final_values=np.array([final_values]).reshape(1,20)

        pred=get_prediction(data=final_values,model=model)[0][0]
        if pred==0:
            d='Die'
        else:
            d='Survive'

        st.write(f"The Patient will:  {d}")

if __name__=='__main__':
    main()