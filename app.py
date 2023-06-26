# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 20:57:00 2023

@author: Amrita
https://github.com/streamlit/streamlit/wiki/Installing-in-a-virtual-environment
"""


import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
model=pickle.load(open('bankmodel_v2.pkl','rb'))


def predict_bank(age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome):
    input_arr = np.array([[age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome]])
    input_df = pd.DataFrame(input_arr, columns = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome'])
    
    #preprocessing
    one_hot_encoded_test = pd.get_dummies(input_df, columns = ['job','marital','contact','poutcome'])
    data_encoded_unseen = pd.merge(input_df, one_hot_encoded_test, how = "inner")
    data_encoded_unseen.drop(['job','marital','contact','poutcome'], axis=1, inplace=True)
    
    for column in data_encoded_unseen.columns:
        if data_encoded_unseen[column].dtype == np.number:
            continue
        data_encoded_unseen[column] = LabelEncoder().fit_transform(data_encoded_unseen[column])
    
    #data_encoded_unseen = data_encoded_unseen[['age', 'education', 'default', 'balance', 'housing', 'loan', 'day','month', 'duration', 'campaign', 'pdays', 'previous','job_entrepreneur', 'job_management', 'job_technician','marital_married', 'marital_single', 'contact_unknown','poutcome_unknown']]
    
    X = data_encoded_unseen
    X_norm_unseen = MinMaxScaler().fit_transform(X.values)
    normed_features_df = pd.DataFrame(X_norm_unseen, index=X.index, columns=X.columns)
    X_test_columns = ['age', 'education', 'default', 'balance', 'housing', 'loan', 'day',
       'month', 'duration', 'campaign', 'pdays', 'previous', 'job_admin.',
       'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
       'marital_divorced', 'marital_married', 'marital_single',
       'contact_cellular', 'contact_telephone', 'contact_unknown',
       'poutcome_failure', 'poutcome_other', 'poutcome_success',
       'poutcome_unknown']
    list_n = list(set(X_test_columns) - set(normed_features_df.columns))
    
    for i in list_n:
        normed_features_df[i] = 0
    normed_features_df = normed_features_df[X_test_columns]
    
    prediction = model.predict_proba(normed_features_df)
    pred = prediction[0][0]
    return int(pred)

def main():
    st.title("Prediction of Bank Marketing Outcome")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    age = st.text_input("age","Type Here")
    job = st.text_input("job","Type Here")
    marital = st.text_input("marital","Type Here")
    education = st.text_input("education","Type Here")
    default = st.text_input("default","Type Here")
    balance = st.text_input("balance","Type Here")
    housing = st.text_input("housing","Type Here")
    loan = st.text_input("loan","Type Here")
    contact = st.text_input("contact","Type Here")
    day = st.text_input("day","Type Here")
    month = st.text_input("month","Type Here")
    duration = st.text_input("duration","Type Here")
    campaign = st.text_input("campaign","Type Here")
    pdays = st.text_input("pdays","Type Here")
    previous = st.text_input("previous","Type Here")
    poutcome = st.text_input("poutcome","Type Here")
    
    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> The client will subscribe a term deposit</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> The client will not subscribe a term deposit</h2>
       </div>
    """
    

    if st.button("Predict"):
        output=predict_bank(age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome)
        st.success('The final outcome is {}'.format(output))

        if output == 0:
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)

if __name__=='__main__':
    #print("Hello")
    main()
    #print("Bye")