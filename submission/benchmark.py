"""LightGBM fraud-detection benchmark on the Kaggle Credit Card Fraud dataset (Lab 16, step 4.4)."""
import json
import os
import platform
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.expanduser("~/ml-benchmark/creditcard.csv")
RESULT_PATH = "benchmark_result.json"
SEED = 42

# 1. Load data
t0 = time.perf_counter()
df = pd.read_csv(DATA_PATH)
load_time = time.perf_counter() - t0

X = df.drop(columns=["Class"])
y = df["Class"]
# Stratified split keeps the ~0.17% fraud ratio in every split: 64% train / 16% val / 20% test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=SEED
)

# 2. Train with early stopping on the validation set
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
t0 = time.perf_counter()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
train_time = time.perf_counter() - t0

# 3. Evaluate on the held-out test set
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
metrics = {
    "auc_roc": roc_auc_score(y_test, proba),
    "accuracy": accuracy_score(y_test, pred),
    "f1_score": f1_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
}

# 4. Inference speed: latency = median over 1000 single-row calls; throughput = one batch of 1000 rows
one_row = X_test.iloc[[0]]
model.predict_proba(one_row)  # warm-up
latencies = []
for _ in range(1000):
    t0 = time.perf_counter()
    model.predict_proba(one_row)
    latencies.append(time.perf_counter() - t0)
latency_ms = float(np.median(latencies) * 1000)

batch = X_test.iloc[:1000]
t0 = time.perf_counter()
model.predict_proba(batch)
batch_time = time.perf_counter() - t0

result = {
    "environment": {
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
    },
    "dataset": {
        "rows": len(df),
        "features": X.shape[1],
        "fraud_ratio": float(y.mean()),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
    },
    "load_time_s": load_time,
    "train_time_s": train_time,
    "best_iteration": model.best_iteration_,
    **metrics,
    "inference_latency_1row_ms": latency_ms,
    "inference_1000rows_ms": batch_time * 1000,
    "inference_throughput_rows_per_s": 1000 / batch_time,
}

with open(RESULT_PATH, "w") as f:
    json.dump(result, f, indent=2)

print(f"{'Load data time':<34}{load_time:.3f} s")
print(f"{'Training time':<34}{train_time:.3f} s")
print(f"{'Best iteration':<34}{model.best_iteration_}")
for name, value in metrics.items():
    print(f"{name:<34}{value:.4f}")
print(f"{'Inference latency (1 row)':<34}{latency_ms:.3f} ms")
print(f"{'Inference 1000 rows':<34}{batch_time * 1000:.3f} ms ({1000 / batch_time:,.0f} rows/s)")
print(f"\nSaved results to {RESULT_PATH}")
