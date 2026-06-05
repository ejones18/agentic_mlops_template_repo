"""
AutoResearch training script — XGBoost classifier on synthetic tabular data.

The agent modifies this file to improve val_accuracy.
Runs in a fixed 60-second time budget (wall clock).

Metric: val_accuracy (higher is better)

NOTE: Always preserve the mlflow logging section at the end of this file.
"""

import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import mlflow

# ── Time Budget ───────────────────────────────────────────────────
TIME_BUDGET = 60  # seconds

# ── Data Generation ───────────────────────────────────────────────
SEED = 42
N_SAMPLES = 50_000
N_FEATURES = 30
N_INFORMATIVE = 15
N_CLASSES = 5

np.random.seed(SEED)

X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    n_redundant=5,
    n_classes=N_CLASSES,
    n_clusters_per_class=2,
    random_state=SEED,
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# ── Hyperparameters (agent modifies these) ────────────────────────
PARAMS = {
    "objective": "multi:softmax",
    "num_class": N_CLASSES,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
    "random_state": SEED,
}

# ── Training ──────────────────────────────────────────────────────
t_start = time.time()

model = xgb.XGBClassifier(**PARAMS)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

t_train = time.time() - t_start

# Check time budget
if t_train > TIME_BUDGET:
    print(f"FAIL: training exceeded time budget ({t_train:.1f}s > {TIME_BUDGET}s)")
    exit(1)

# ── Evaluation ────────────────────────────────────────────────────
y_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred)
train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)

# ── Summary (parsed by the Foundry agent) ─────────────────────────
print("---")
print(f"val_accuracy:     {val_accuracy:.6f}")
print(f"train_accuracy:   {train_accuracy:.6f}")
print(f"training_seconds: {t_train:.1f}")
print(f"n_estimators:     {PARAMS['n_estimators']}")
print(f"max_depth:        {PARAMS['max_depth']}")
print(f"learning_rate:    {PARAMS['learning_rate']}")
print(f"n_samples:        {N_SAMPLES}")
print(f"n_features:       {N_FEATURES}")

# ── MLflow Logging (DO NOT REMOVE) ────────────────────────────────
mlflow.log_metrics({
    "val_accuracy": val_accuracy,
    "train_accuracy": train_accuracy,
    "training_seconds": t_train,
})
mlflow.log_params({
    "n_estimators": PARAMS["n_estimators"],
    "max_depth": PARAMS["max_depth"],
    "learning_rate": PARAMS["learning_rate"],
    "subsample": PARAMS["subsample"],
    "colsample_bytree": PARAMS["colsample_bytree"],
    "min_child_weight": PARAMS["min_child_weight"],
    "gamma": PARAMS["gamma"],
    "reg_alpha": PARAMS["reg_alpha"],
    "reg_lambda": PARAMS["reg_lambda"],
    "n_samples": N_SAMPLES,
    "n_features": N_FEATURES,
})
