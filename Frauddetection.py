# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate synthetic transaction data with enhanced features
n_samples = 10000
start_date = datetime(2025, 1, 1)
timestamps = [start_date + timedelta(minutes=int(x)) for x in np.random.uniform(0, 30*24*60, n_samples)]

data = {
    'timestamp': timestamps,
    'transaction_amount': np.random.normal(100, 50, n_samples),  # Normal transactions ~$100
    'merchant_id': np.random.randint(1, 100, n_samples),         # Merchant identifier
    'customer_id': np.random.randint(1, 500, n_samples),        # Customer identifier
    'location_lat': np.random.normal(40, 0.1, n_samples),       # Simulated latitude
    'location_long': np.random.normal(-74, 0.1, n_samples)      # Simulated longitude
}
df = pd.DataFrame(data)

# Simulate fraud (0.5% of data)
n_fraud = int(0.005 * n_samples)
fraud_idx = np.random.choice(df.index, n_fraud, replace=False)
df.loc[fraud_idx, 'transaction_amount'] = np.random.uniform(500, 2000, n_fraud)  # High amounts
df.loc[fraud_idx, 'timestamp'] = [start_date + timedelta(hours=np.random.uniform(0, 6)) for _ in range(n_fraud)]  # Early hours

# Create labels (1 for normal, -1 for fraud)
df['label'] = 1
df.loc[fraud_idx, 'label'] = -1

# Step 2: Feature engineering
# Extract time-based features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_night'] = df['hour'].apply(lambda x: 1 if 0 <= x <= 6 else 0)

# Statistical features
df['amount_log'] = np.log1p(df['transaction_amount'])  # Log-transform amount
df['customer_avg_amount'] = df.groupby('customer_id')['transaction_amount'].transform('mean')
df['amount_zscore'] = (df['transaction_amount'] - df['customer_avg_amount']) / df.groupby('customer_id')['transaction_amount'].transform('std').fillna(1)

# Time since last transaction (in seconds)
df = df.sort_values(['customer_id', 'timestamp'])
df['time_since_last_txn'] = df.groupby('customer_id')['timestamp'].diff().dt.total_seconds().fillna(0)

# Step 3: Preprocess the data
features = [
    'amount_log', 'hour', 'day_of_week', 'is_night',
    'amount_zscore', 'time_since_last_txn', 'merchant_id', 'location_lat', 'location_long'
]
X = df[features].fillna(0)  # Fill NaNs (e.g., for z-score of single transactions)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Handle class imbalance with SMOTE
# SMOTE for supervised evaluation
smote = SMOTE(sampling_strategy=0.1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, df['label'])

# Step 5: Train Isolation Forest with tuned parameters
model = IsolationForest(
    n_estimators=200,
    max_samples='auto',
    contamination=0.005,  # Expected fraud rate
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)
model.fit(X_scaled)  # Train on original (non-SMOTE) data

# Predict anomalies (-1 for anomalies, 1 for normal)
df['prediction'] = model.predict(X_scaled)
df['anomaly_score'] = model.decision_function(X_scaled)  # Raw anomaly scores

# Step 6: Evaluate the model
# Convert predictions to match label format (1 for fraud, 0 for normal)
y_true = (df['label'] == -1).astype(int)  # 1 for fraud, 0 for normal
y_pred = (df['prediction'] == -1).astype(int)
y_scores = -df['anomaly_score']  # Higher scores for anomalies

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))
print(f"AUC-ROC: {roc_auc_score(y_true, y_scores):.4f}")

# Precision-Recall Curve
precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
plt.figure(figsize=(8, 5))
plt.plot(recalls, precisions, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 7: Explainability with SHAP
explainer = shap.TreeExplainer(model, X_scaled)
shap_values = explainer.shap_values(X_scaled[:1000])  # Limit for speed
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X[:1000], feature_names=features, plot_type="bar")
plt.title('SHAP Feature Importance')
plt.show()

# Step 8: Visualize anomalies
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='transaction_amount', y='hour',
    hue=df['prediction'], style=df['prediction'],
    palette={1: 'blue', -1: 'red'}, data=df, alpha=0.6
)
plt.title('Transaction Amount vs Hour (Anomalies in Red)')
plt.xlabel('Transaction Amount ($)')
plt.ylabel('Hour of Day')
plt.legend(labels=['Normal', 'Anomaly'])
plt.show()

# README
# Overview:
# This script implements an anomaly detection model for identifying fraudulent financial transactions using the Isolation Forest algorithm. It includes feature engineering, handling of class imbalance with SMOTE, hyperparameter tuning, comprehensive evaluation metrics, and explainability via SHAP. The model is designed to run in Google Colab, leveraging synthetic transaction data for demonstration.
#
# Usage Instructions:
# 1. Open Google Colab (colab.research.google.com) and create a new notebook.
# 2. Install the SHAP library by running: `!pip install shap` (other dependencies are pre-installed).
# 3. Copy and paste this code into a cell and run it using Shift + Enter.
# 4. Review outputs: classification report, AUC-ROC score, precision-recall curve, confusion matrix, SHAP feature importance, and anomaly visualization.
#
# Dependencies:
# - Python 3.x
# - Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, imblearn, shap
# - Note: SHAP requires installation in Colab; others are pre-installed.
#
# Adapting to Your Dataset:
# - To use your own data, upload a CSV file using: `from google.colab import files; uploaded = files.upload(); df = pd.read_csv('your_file.csv')`.
# - Ensure a timestamp column is in datetime format: `df['timestamp'] = pd.to_datetime(df['timestamp'])`.
# - Update the `features` list to match your columns (e.g., amount, merchant_category).
# - If fraud labels exist, map them: `df['label'] = df['is_fraud'].map({0: 1, 1: -1})`.
# - Adjust `contamination` in IsolationForest to reflect your dataset’s fraud rate (e.g., 0.01 for 1% fraud).
#
# Notes:
# - The synthetic dataset assumes a 0.5% fraud rate; tune `contamination` for real data.
# - SHAP is limited to 1000 samples for performance; remove the limit for smaller datasets.
# - For large datasets, monitor memory usage in Colab’s free tier.
# - Contact for support or enhancements (e.g., alternative models like DBSCAN).