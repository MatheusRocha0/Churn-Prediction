from lightgbm import LGBMClassifier
import streamlit as st
import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("https://raw.githubusercontent.com/MatheusRocha0/Churn-Prediction/main/telecom_churn.csv")

model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(layout = "wide")

st.title("Welcome to Churn Predictor")
st.header("Please enter the necessary data")
st.subheader("Here is an example")
st.write(df.drop("Churn", axis = 1)[:1])

st.sidebar.markdown("**Select the values:**")

account_weeks = st.sidebar.slider("AccountWeeks", min_value = 1.0, max_value = 243.0)

conct_renewal = st.sidebar.selectbox("ContractRenewal", options = ["True", "False"])
if conct_renewal == "True":
   contract_renewal = 1

else:
   contract_renewal = 0

dp = st.sidebar.selectbox("DataPlan", options = ["True", "False"])
if dp == "True":
   data_plan = 1

else:
   data_plan = 0

data_usage = st.sidebar.slider("DataUsage", min_value = 0.0, max_value = 5.4)

cust_serv_calls = st.sidebar.slider("CustServCalls", min_value = 0.0, max_value = 9.0)

day_mins = st.sidebar.slider("DayMins", min_value = 0.0, max_value = 350.8)

day_calls = st.sidebar.slider("DayCalls", min_value = 0.0, max_value = 165.0)

monthly_charge = st.sidebar.slider("MonthlyCharge", min_value = 14.0, max_value = 111.3)

overage_fee = st.sidebar.slider("OverageFee", min_value = 0.0, max_value = 18.19)

roam_mins = st.sidebar.slider("RoamMins", min_value = 0.0, max_value = 20.0)

data = np.array([[account_weeks, contract_renewal, data_plan, data_usage, cust_serv_calls, day_mins, day_calls, monthly_charge, overage_fee, roam_mins]])

if st.sidebar.button("Submit"):
   pred = model.predict(data)
   if pred == 1:
      st.warning("This customer is going to stop buying from us")

   else:
      st.success("This customer is not going to stop buying from us")
      pass
