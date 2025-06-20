import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load data
data = pd.read_csv('data/sample_input.csv')

# Drop customerID if present
if 'customerID' in data.columns:
    data.drop('customerID', axis=1, inplace=True)

# Feature engineering
data['TotalRevenue'] = data['MonthlyCharges'] * data['tenure']

def tenure_group(tenure):
    if tenure <= 12:
        return 'New'
    elif tenure <= 36:
        return 'Mid'
    else:
        return 'Loyal'

data['TenureGroup'] = data['tenure'].apply(tenure_group)
data['HasStreamingService'] = data[['StreamingTV', 'StreamingMovies']].apply(
    lambda row: 'Yes' if 'Yes' in row.values else 'No', axis=1
)

if 'Churn' in data.columns:
    data.drop('Churn', axis=1, inplace=True)

# One-hot encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# Align with training columns
model_input_columns = scaler.mean_.shape[0]
while data_encoded.shape[1] < model_input_columns:
    data_encoded[data_encoded.columns[-1] + '_pad'] = 0

# Scale and predict
scaled_data = scaler.transform(data_encoded)
predictions = model.predict(scaled_data)
probs = model.predict_proba(scaled_data)[:, 1]

data['Churn Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
data['Churn Probability (%)'] = (probs * 100).round(2)

# 🟨 Add Risk Level Column
def get_risk(prob):
    if prob > 80:
        return '⚠️ High Risk'
    elif prob > 50:
        return 'Moderate'
    else:
        return 'Low'

data['Risk Level'] = data['Churn Probability (%)'].apply(get_risk)

# Streamlit UI
st.set_page_config(page_title="Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")
st.write("Predictions for preloaded customer data.")

# 🔍 FILTER CONTROLS
st.subheader("🔍 Filter Results")

contract_options = data['Contract'].unique().tolist()
selected_contract = st.multiselect("Select Contract Type(s):", contract_options, default=contract_options)

gender_options = data['gender'].unique().tolist()
selected_gender = st.multiselect("Select Gender(s):", gender_options, default=gender_options)

min_tenure = int(data['tenure'].min())
max_tenure = int(data['tenure'].max())
tenure_range = st.slider("Select Tenure Range (months):", min_value=min_tenure, max_value=max_tenure, value=(min_tenure, max_tenure))

# Apply filters
filtered_data = data[
    (data['Contract'].isin(selected_contract)) &
    (data['gender'].isin(selected_gender)) &
    (data['tenure'].between(tenure_range[0], tenure_range[1]))
]

# 📑 Summary Metrics
st.subheader("📋 Summary")
total = len(filtered_data)
churned = filtered_data['Churn Prediction'].value_counts().get('Yes', 0)
not_churned = filtered_data['Churn Prediction'].value_counts().get('No', 0)
churn_rate = round((churned / total) * 100, 2) if total > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", total)
col2.metric("Churned", churned)
col3.metric("Churn Rate (%)", churn_rate)

# 🔍 Optional Filter for churn only
if st.checkbox("Show only churned customers"):
    st.dataframe(filtered_data[filtered_data['Churn Prediction'] == 'Yes'])
else:
    st.dataframe(filtered_data)

# 📊 Pie Chart
if total > 0:
    st.subheader("📌 Churn Distribution")
    fig, ax = plt.subplots()
    labels = ['Churned', 'Retained']
    sizes = [churned, not_churned]
    colors = ['#ff6666', '#66b3ff']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')
    st.pyplot(fig)

# 📤 Download Button
csv = filtered_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Filtered Predictions as CSV",
    data=csv,
    file_name='filtered_churn_predictions.csv',
    mime='text/csv'
)

st.success(f"✅ Predictions completed for {total} customers.")
