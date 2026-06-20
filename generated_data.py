import os
import numpy as np
import pandas as pd

np.random.seed(42)
num_transactions = 50000

print(f"Manufacturing {num_transactions} streaming fintech transaction logs...")

# Generate structural baseline features
tx_ids = [
    f"TX_{i:06d}" for i in range(1, num_transactions + 1) 
    ]
user_ids = [
    f"OPay_{np.random.randint(1, 2000):04d}" for k in range(num_transactions)
    ]

tx_amount = np.random.exponential(scale=35000, size=num_transactions) + 100

tx_hour = np.random.choice(
    np.arange(24), size=num_transactions, p=[0.01] * 6 + [0.06] * 12 + [0.03666666666666666666666666666666666666667] * 6
)

wallet_drainage_ratio = np.random.uniform(0.01, 0.85, num_transactions)

device_changed = np.random.choice([0, 1], size=num_transactions, p=[0.94, 0.06])

recent_tx_count_1hr = np.random.negative_binomial(n=2, p=0.4, size=num_transactions)

# Combine into DataFrame

df = pd.DataFrame(
    {
        "tx_id": tx_ids,
        "user_id": user_ids,
        "tx_amount": tx_amount,
        "tx_hour": tx_hour,
        "wallet_drainage_ratio": wallet_drainage_ratio,
        "device_changed": device_changed,
        "recent_tx_count_1hr": recent_tx_count_1hr,
    }
)

# We claculate a hidden fraud score
fraud_score = (
    (df["device_changed"] * 5.0)
    + (df["wallet_drainage_ratio"] * 4.0)
    + (df["recent_tx_count_1hr"] * 0.4)
    + np.where((df["tx_hour"] >= 0) & (df["tx_hour"] <= 5), 2.5, 0)
)

# Market baseline randomness
noise = np.random.normal(0, 1.2, num_transactions)
final_fraud_metric = fraud_score + noise

# Set very strict fraud rate (~1.5% of all transactions are actual malicious fraud blocks)
threshold = np.percentile(final_fraud_metric, 98.5)
df["is_fraud"] = (final_fraud_metric >= threshold).astype(int)

# Force high-fraud rows to match structural profiles perfectly (overriding noise impacts)
# Preserve the original wallet ratio to avoid label leakage when training models.
df["wallet_drainage_ratio_original"] = df["wallet_drainage_ratio"]

# This keeps the true feature (`wallet_drainage_ratio_original`) unchanged so
# labels are derived only from existing features (no post-label overwrites).
df["wallet_drainage_ratio_manipulated"] = df["wallet_drainage_ratio_original"].copy()
df.loc[df["is_fraud"] == 1, "wallet_drainage_ratio_manipulated"] = np.random.uniform(
    0.90, 1.0, df["is_fraud"].sum()
)

print(
    df.groupby("is_fraud")[
        [
            "tx_amount",
            "wallet_drainage_ratio_original",
            "wallet_drainage_ratio_manipulated",
            "recent_tx_count_1hr",
        ]
    ].mean()
)

df.to_csv("fintech_fraud_logs.csv", index=False)