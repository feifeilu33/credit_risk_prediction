import streamlit as st
import pandas as pd
import joblib

model = joblib.load(r"E:\dataprojects\jupyter\credit_risk\best_extra_trees_model.pkl")

base_path = r"E:\dataprojects\jupyter\credit_risk"

encoder = {
    'Sex': joblib.load(r"E:\dataprojects\jupyter\credit_risk\le_Sex_encoder.pkl"),
    'Housing': joblib.load(r"E:\dataprojects\jupyter\credit_risk\le_Housing_encoder.pkl"),
    'Saving accounts': joblib.load(r"E:\dataprojects\jupyter\credit_risk\le_Saving accounts_encoder.pkl"),
    'Checking account': joblib.load(r"E:\dataprojects\jupyter\credit_risk\le_Checking account_encoder.pkl")
}

st.title('Credit Risk Prediction App')
st.write('Enter applicant details to predict credit risk.')

age = st.number_input('Age', min_value=18, max_value=100, value=30)
sex = st.selectbox('Sex', options=['male', 'female'])
job = st.number_input('Job (0-3)', min_value=0, max_value=3, value=1)
housing = st.selectbox('Housing', options=['own', 'rent', 'free'])
saving_accounts = st.selectbox('Saving accounts', options=['little', 'moderate', 'rich', 'quite rich'])
checking_account = st.selectbox('Checking account', options=['little', 'moderate', 'rich'])
credit_amount = st.number_input('Credit Amount', min_value=0, value=100)
duration = st.number_input('Duration (months)', min_value=1, value=12)

input_df = pd.DataFrame({
    'Age': [age],
    'Sex': [encoder['Sex'].transform([sex])[0]],
    'Job': [job],
    'Housing': [encoder['Housing'].transform([housing])[0]],
    'Saving accounts': [encoder['Saving accounts'].transform([saving_accounts])[0]],
    'Checking account': [encoder['Checking account'].transform([checking_account])[0]],
    'Credit amount': [credit_amount],
    'Duration': [duration]
})

if st.button('Predict Credit Risk'):
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success('The applicant is likely to repay the credit (Low Risk).')   
    else:
        st.error('The applicant is likely to default on the credit (High Risk).')


# open terminal and run: streamlit run app.py