import sys
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ======================

# ======================
st.title("Customer Churn Prediction 👑")

# ======================

# ======================
data = pd.read_csv("TelcoCustomerChurn.csv")

# ======================

# ======================

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data.dropna(inplace=True)


data.drop("customerID", axis=1, inplace=True)

# ======================

# ======================
data["gender"] = data["gender"].map({"Male": 1, "Female": 0})
data["Partner"] = data["Partner"].map({"Yes": 1, "No": 0})
data["Dependents"] = data["Dependents"].map({"Yes": 1, "No": 0})
data["PhoneService"] = data["PhoneService"].map({"Yes": 1, "No": 0})
data["PaperlessBilling"] = data["PaperlessBilling"].map({"Yes": 1, "No": 0})
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})


data = pd.get_dummies(data)

# ======================

# ======================
X = data.drop("Churn", axis=1)
y = data["Churn"]

# ======================

# ======================
model = RandomForestClassifier()
model.fit(X, y)
import pickle

pickle.dump(model, open("model.pkl", "wb"))
# ======================

# ======================
st.header("Enter Customer Data")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure", 0, 100)
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0)

# ======================

# ======================
input_dict = {
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": SeniorCitizen,
    "Partner": 1 if Partner == "Yes" else 0,
    "Dependents": 1 if Dependents == "Yes" else 0,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
}

input_df = pd.DataFrame([input_dict])


input_df = input_df.reindex(columns=X.columns, fill_value=0)

# ======================
# Prediction
# ======================
if st.button("Predict"):
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠️ Customer will likely CHURN (leave)")
    else:
        st.success("✅ Customer will likely STAY")