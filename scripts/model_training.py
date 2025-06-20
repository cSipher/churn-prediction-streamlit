import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Step 1: Load the cleaned + feature-engineered dataset
df = pd.read_csv('../data/cleaned_churn.csv')

# Step 2: Separate features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Step 3: One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 5: Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Step 7: Define XGBoost and grid search parameters
xgb = XGBClassifier(eval_metric='logloss', random_state=42)

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0]
}

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    verbose=1,
    n_jobs=-1
)

# Step 8: Fit the best model
grid.fit(X_train_resampled, y_train_resampled)

# Get the best model
model = grid.best_estimator_
print("✅ Best Parameters:", grid.best_params_)

# Step 9: Predict and evaluate on test set
y_pred = model.predict(X_test_scaled)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 10: Save the final model and scaler
joblib.dump(model, '../models/churn_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')

print("✅ Tuned XGBoost model and scaler saved successfully.")
