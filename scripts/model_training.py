import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# -----------------------------
# Step 1: Load cleaned dataset
# -----------------------------
df = pd.read_csv('../data/cleaned_churn.csv')

# Step 2: Separate features (X) and target (y)
# Step 2: Encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 3: Separate features and target
X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']


# -----------------------------
# Step 3: Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------
# Step 4: Apply SMOTE to handle class imbalance
# -----------------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# -----------------------------
# Step 5: Scale features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Step 6: Train with GridSearchCV and XGBoost
# -----------------------------
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

xgb = XGBClassifier(
    eval_metric='logloss',
    learning_rate=0.01,
    max_depth=7,
    n_estimators=200,
    subsample=0.8
)

grid = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1)



grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train_res)

# -----------------------------
# Step 7: Evaluate model
# -----------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

print(f"\n✅ Best Parameters: {grid_search.best_params_}")
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred)}\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Step 8: Save model and scaler for Streamlit
# -----------------------------
os.makedirs('../models', exist_ok=True)

joblib.dump(best_model, '../models/churn_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')

print("✅ Tuned XGBoost model and scaler saved successfully.")
