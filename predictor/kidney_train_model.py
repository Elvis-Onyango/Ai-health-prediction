import os
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "kidney_model.pkl")
SCALER_SAVE_PATH = os.path.join(BASE_DIR, "kidney_scaler.pkl")

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    DATA_PATH = os.path.join(BASE_DIR, "test_data", "kidney.csv")  
    print(f"\nLooking for data at: {DATA_PATH}")
    
    # Verify data path exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Please ensure the file exists.")
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"\nInitial dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check target variable
    target_col = 'Chronic Kidney Disease: yes'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    print(f"\nInitial class distribution:\n{df[target_col].value_counts()}")
    
    # Handle missing values
    print(f"\nMissing values per column:")
    print(df.isnull().sum())
    
    # Drop rows with missing values (consider imputation for production)
    df.dropna(inplace=True)
    print(f"\nDataset shape after dropping missing values: {df.shape}")
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Ensure target is binary (0 and 1)
    y = y.astype(int)
    
    return X, y

def preprocess_features(X_train, X_test):
    """Scale features without data leakage."""
    print("\nPreprocessing Features...")
    
    # Check for non-numeric columns
    non_numeric = X_train.select_dtypes(exclude=['number']).columns
    if len(non_numeric) > 0:
        print(f"Warning: Non-numeric columns found: {non_numeric}")
        X_train = X_train.select_dtypes(include=['number'])
        X_test = X_test.select_dtypes(include=['number'])
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train shape after scaling: {X_train_scaled.shape}")
    print(f"Test shape after scaling: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler

def handle_class_imbalance(X_train, y_train):
    """Handle class imbalance using SMOTE and optional undersampling."""
    print("\nBalancing Data...")
    print(f"Original class distribution: {np.bincount(y_train)}")
    
    # First apply SMOTE to oversample minority class
    smote = SMOTE(sampling_strategy='auto', random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {np.bincount(y_res)}")
    
    # Only apply undersampling if severe imbalance remains
    class_counts = np.bincount(y_res)
    if class_counts[0] > 1.5 * class_counts[1]:
        rus = RandomUnderSampler(sampling_strategy=0.8, random_state=RANDOM_STATE)
        X_res, y_res = rus.fit_resample(X_res, y_res)
        print(f"After undersampling: {np.bincount(y_res)}")
    
    return X_res, y_res

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with comprehensive metrics."""
    print("\nEvaluating Model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No CKD', 'CKD'],
                yticklabels=['No CKD', 'CKD'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        feat_importances = pd.Series(model.feature_importances_, index=X_test.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.title('Top 10 Feature Importances')
        plt.show()

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train and evaluate the stacking model with hyperparameter tuning."""
    print("\n Training Stacking Model...")

    # Base models
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Meta model
    logreg = LogisticRegression(
        penalty='l2',
        C=0.1,
        solver='liblinear',
        random_state=RANDOM_STATE
    )
    
    # Create stacking ensemble
    stacking_model = StackingClassifier(
        estimators=[('xgb', xgb), ('rf', rf)],
        final_estimator=logreg,
        passthrough=True,
        cv=CV_FOLDS,
        n_jobs=-1
    )
    
    # Hyperparameter tuning
    param_grid = {
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__max_depth': [3, 5],
        'rf__n_estimators': [30, 50, 100]
    }
    
    random_search = RandomizedSearchCV(
        estimator=stacking_model,
        param_distributions=param_grid,
        n_iter=5,
        cv=CV_FOLDS,
        scoring='roc_auc',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    print("\nStarting hyperparameter tuning...")
    random_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best ROC AUC score: {random_search.best_score_:.4f}")
    
    # Get best model
    best_model = random_search.best_estimator_
    
    # Evaluate on test set
    evaluate_model(best_model, X_test, y_test)
    
    return best_model

def save_artifacts(model, scaler):
    """Save model and preprocessing artifacts."""
    print("\nSaving artifacts...")
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Scaler saved to: {SCALER_SAVE_PATH}")

def main():
    print("\nStarting Kidney Disease Prediction Pipeline...")
    
    try:
        # Load and preprocess data
        X, y = load_and_preprocess_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE, 
            stratify=y
        )
        print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Handle class imbalance
        X_train_res, y_train_res = handle_class_imbalance(X_train, y_train)
        
        # Preprocess features
        X_train_proc, X_test_proc, scaler = preprocess_features(X_train_res, X_test)
        
        # Train and evaluate model
        model = train_and_evaluate(X_train_proc, X_test_proc, y_train_res, y_test)
        
        # Save artifacts
        save_artifacts(model, scaler)
        
        print("\n Training completed successfully!")
        
    except Exception as e:
        print(f"\n Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()