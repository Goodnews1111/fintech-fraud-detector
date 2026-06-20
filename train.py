import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

# Split the dataset into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2,  random_state=42 )

# Anomaly Detection Layer (Isolation Forest)
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

# Features Importance
fig1, ax1 = plt.subplots(figsize=(11, 7))
plot_importance(model, max_num_features=5, importance_type='weight', ax=ax1)
plt.title("Top Fraud Indicators")
plt.tight_layout()
plt.savefig("feature_importance.png") 
plt.close(fig1)

# Evaluation
preds = model.predict(X_test)

print("Model Classification")
print(classification_report(y_test, preds))
print(f"ROC-AUC Score: {roc_auc_score(y_test, preds):.4f}")

joblib.dump(model, "fraud_detection_model.pkl")
print("Model training Completed")

fig2, ax2 = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legit', 'Fraud'])
disp.plot(cmap=plt.cm.Blues, ax=ax2)
plt.title("Fraud Detection Performance")
plt.savefig("confusion_matrix.png")
plt.close(fig2)
