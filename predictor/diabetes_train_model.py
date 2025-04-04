import os
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "test_data", "diabetes.csv")
df = pd.read_csv(DATA_PATH)

# Remove missing values
df.dropna(inplace=True)

# Define features & target
X = df.drop(columns=['Diabetes_012'])
y = df['Diabetes_012']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Base models
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    eval_metric='logloss',
    use_label_encoder=False
)

logreg = LogisticRegression(max_iter=1000, class_weight='balanced')

# Stacking model
stacking_model = StackingClassifier(
    estimators=[('xgb', xgb)],
    final_estimator=logreg,
    passthrough=True
)

# Model training
stacking_model.fit(X_train, y_train)

# Evaluation
y_pred = stacking_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, stacking_model.predict_proba(X_test), multi_class='ovr')

print(f"Diabetes Prediction Stacking Model Accuracy: {accuracy * 100:.2f}%")
print(f"AUC-ROC Score: {roc_auc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model & scaler
joblib.dump(stacking_model, os.path.join(BASE_DIR, "diabetes_model.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler_diabetes.pkl"))
print("Diabetes Prediction Model and scaler saved successfully!")
