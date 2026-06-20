import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

print("Loading Dataset....")

df = pd.read_csv("fintech_fraud_logs.csv")
print("Dataset Loaded Successfully")

print("Features Selection & Data spliting")

features = ["tx_hour", "tx_amount", "wallet_drainage_ratio_original",
            "device_changed", "recent_tx_count_1hr"]
X = df[features]

y = df["is_fraud"]

# Split the dataset into 98.5% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2,  random_state=42 )

# Anomaly Detection Layer (Isolation Forest)
# We flag things that deviate from the normal distribution
iso = IsolationForest(contamination= 0.015, random_state= 42)
iso_pred = iso.fit_predict(X_train)

# XGBoost learns to distinguish fraud from legitimate traffic

neg_count = y_train.value_counts()[0]
pos_count = y_train.value_counts()[1]

ratio = neg_count/ pos_count
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight = ratio,
    random_state = 42   
)

model.fit(X_train, y_train)

# Evaluation
preds = model.predict(X_test)
print("Model Classification")
print(classification_report(y_test, preds))
print(f"ROC-AUC Score: {roc_auc_score(y_test, preds):.4f}")

joblib.dump(model, "fraud_detection_model.pkl")
print("Model training Completed")