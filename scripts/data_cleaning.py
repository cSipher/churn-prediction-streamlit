import pandas as pd

# Load raw dataset
df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.columns = df.columns.str.strip()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# ✅ Feature Engineering:

# 1. Total Revenue = Monthly Charges × Tenure
df['TotalRevenue'] = df['MonthlyCharges'] * df['tenure']

# 2. Tenure Grouping
def tenure_group(tenure):
    if tenure <= 12:
        return 'New'
    elif tenure <= 36:
        return 'Mid'
    else:
        return 'Loyal'

df['TenureGroup'] = df['tenure'].apply(tenure_group)

# 3. Has Streaming Service
df['HasStreamingService'] = df[['StreamingTV', 'StreamingMovies']].apply(
    lambda row: 'Yes' if 'Yes' in row.values else 'No', axis=1
)

# Save the updated file
df.to_csv('../data/cleaned_churn.csv', index=False)
print("✅ Cleaned + engineered data saved.")
