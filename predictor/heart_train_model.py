import os
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
DATA_PATH = os.path.join(BASE_DIR, "test_data", "heart.csv")  
df = pd.read_csv(DATA_PATH)

# Remove missing values
df.dropna(inplace=True)

# Define features & target
X = df.drop(columns=['target'])
y = df['target']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define base models
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
logreg = LogisticRegression()

# Create stacking ensemble
stacking_model = StackingClassifier(
    estimators=[('xgb', xgb)],
    final_estimator=logreg,  # Logistic Regression as meta-model
    passthrough=True  # Pass original features + XGBoost predictions
)

# Train the model
stacking_model.fit(X_train, y_train)

# Evaluate model
y_pred = stacking_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacking Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save model & scaler
joblib.dump(stacking_model, os.path.join(BASE_DIR, "stacking_model.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "heart_scaler.pkl"))
print("Stacking Model and scaler saved successfully!")
