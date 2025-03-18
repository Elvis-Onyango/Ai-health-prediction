import os
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define the correct path for heart.csv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
DATA_PATH = os.path.join(BASE_DIR, "test_data", "heart.csv")  # Full path to heart.csv

# Load dataset
df = pd.read_csv(DATA_PATH)

# Check for missing values
df.dropna(inplace=True)

# Define features (X) and target (y)
X = df.drop(columns=['target'])  # Features
y = df['target']  # Target (1 = Heart Disease, 0 = No Disease)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save model & scaler
joblib.dump(model, os.path.join(BASE_DIR, "heart_disease_model.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
print(" Model and scaler saved successfully!")

# Optional: Visualize feature importance
coef = pd.Series(model.coef_[0], index=df.columns[:-1]).sort_values()
plt.figure(figsize=(10, 5))
sns.barplot(x=coef, y=coef.index)
plt.title("Feature Importance in Heart Disease Prediction")
plt.xlabel("Coefficient Value")
plt.show()
