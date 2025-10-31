from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os, time, csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.utils.balance_data import balance_data
from src.utils.k_fold_split import k_fold_split

# =====================================================
# SETUP
# =====================================================
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/plots", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

mean_log_path = "logs/hparam_results_RF_sklearn.csv"
fold_log_path = "logs/fold_results_RF_sklearn.csv"

# CSV headers
if not os.path.exists(fold_log_path):
    with open(fold_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "n_estimators", "max_depth", "fold", "acc", "f1", "time"]
        )

if not os.path.exists(mean_log_path):
    with open(mean_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "n_estimators",
                "max_depth",
                "mean_acc",
                "mean_f1",
                "mean_time",
            ]
        )

# =====================================================
# CONFIG
# =====================================================
N_SPLITS = 10
PARAM_GRID = {"n_estimators": [50, 100, 150], "max_depth": [5, 10, 15, 20]}
np.random.seed(42)

# =====================================================
# LOAD DATA
# =====================================================
X_train = np.loadtxt("data/processed/train/X_train.csv", delimiter=",")
y_train = np.loadtxt("data/processed/train/y_train.csv", delimiter=",")
X_test = np.loadtxt("data/processed/test/X_test.csv", delimiter=",")
y_test = np.loadtxt("data/processed/test/y_test.csv", delimiter=",")

folds = k_fold_split(X_train, y_train, N_SPLITS)

results = []
f1_trend = {}

# =====================================================
# K-FOLD CROSS-VALIDATION
# =====================================================
for n_est in PARAM_GRID["n_estimators"]:
    for depth in PARAM_GRID["max_depth"]:
        fold_accs, fold_f1, fold_times = [], [], []
        key = f"n_estimators={n_est}, max_depth={depth}"
        f1_trend[key] = []

        print(f"\nTesting n_estimators={n_est}, max_depth={depth}")

        for fold_idx in range(N_SPLITS):
            val_idx = folds[fold_idx]
            train_idx = np.hstack([folds[i] for i in range(N_SPLITS) if i != fold_idx])

            X_val, y_val = X_train[val_idx], y_train[val_idx]
            X_tr_raw, y_tr_raw = X_train[train_idx], y_train[train_idx]

            X_tr, y_tr = balance_data(X_tr_raw, y_tr_raw)

            rf = RandomForestClassifier(
                n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1
            )

            start_time = time.time()
            rf.fit(X_tr, y_tr)
            elapsed = time.time() - start_time

            y_pred = rf.predict(X_val)

            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="macro")

            fold_accs.append(acc)
            fold_f1.append(f1)
            fold_times.append(elapsed)
            f1_trend[key].append(f1)

            print(
                f"Fold {fold_idx+1}/{N_SPLITS} -> Acc={acc:.3f}, F1={f1:.3f}, Time={elapsed:.2f}s"
            )

            with open(fold_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        n_est,
                        depth,
                        fold_idx + 1,
                        round(acc, 4),
                        round(f1, 4),
                        round(elapsed, 2),
                    ]
                )

        mean_acc = np.mean(fold_accs)
        mean_f1 = np.mean(fold_f1)
        mean_time = np.mean(fold_times)
        results.append((n_est, depth, mean_acc, mean_f1, mean_time))

        print(
            f"→ Mean Accuracy={mean_acc:.3f}, Mean F1={mean_f1:.3f}, Time={mean_time:.2f}s"
        )

        with open(mean_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    n_est,
                    depth,
                    round(mean_acc, 4),
                    round(mean_f1, 4),
                    round(mean_time, 2),
                ]
            )

# =====================================================
# BEST PARAMETER
# =====================================================
results = np.array(results, dtype=object)
best_idx = np.argmax(results[:, 3])  # index by F1 score
best_params = results[best_idx]
best_n_est, best_depth = best_params[0], best_params[1]

print("\nBest Params:")
print(f"  n_estimators={best_n_est}, max_depth={best_depth}")
print(f"  Mean F1={best_params[3]:.3f}, Mean Time={best_params[4]:.2f}s")

# =====================================================
# TRAIN BEST MODEL ON FULL DATA
# =====================================================
X_train_bal, y_train_bal = balance_data(X_train, y_train)
best_rf = RandomForestClassifier(
    n_estimators=int(best_n_est), max_depth=int(best_depth), random_state=42, n_jobs=-1
)
best_rf.fit(X_train_bal, y_train_bal)

import pickle

with open("checkpoints/best_random_forest_sklearn.pkl", "wb") as f:
    pickle.dump(best_rf, f)

# =====================================================
# TEST EVALUATION
# =====================================================
y_pred_test = best_rf.predict(X_test)
acc_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test, average="macro")

print("\nTest Performance:")
print(f"Accuracy: {acc_test:.3f}")
print(f"F1: {f1_test:.3f}")

# =====================================================
# PLOTTING
# =====================================================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- F1 trend across folds ---
plt.figure(figsize=(10, 5))
for key, values in f1_trend.items():
    plt.plot(range(1, N_SPLITS + 1), values, marker="o", label=key)
plt.xlabel("Fold")
plt.ylabel("F1 Score")
plt.title("Fold-by-Fold F1 Trend for Each Hyperparameter Setting (Random Forest)")
plt.legend()
plt.grid(True)
f1_path = f"logs/plots/f1_trend_RF_sklearn_{timestamp}.png"
plt.savefig(f1_path, bbox_inches="tight")
plt.show()

# =====================================================
# VALIDATION CURVE (Effect of max_depth)
# =====================================================
param_range = np.arange(2, 21, 2)
train_scores, val_scores = validation_curve(
    RandomForestClassifier(n_estimators=int(best_n_est), random_state=42, n_jobs=-1),
    X_train_bal,
    y_train_bal,
    param_name="max_depth",
    param_range=param_range,
    scoring="f1_macro",
    cv=5,
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(param_range, train_mean, label="Training F1", marker="o")
plt.plot(param_range, val_mean, label="Validation F1", marker="s")
plt.xlabel("max_depth")
plt.ylabel("Macro F1 Score")
plt.title("Validation Curve (Effect of max_depth on Random Forest)")
plt.legend()
plt.grid(True)
val_path = f"logs/plots/val_curve_RF_sklearn_{timestamp}.png"
plt.savefig(val_path, bbox_inches="tight")
plt.show()

# =====================================================
# LEARNING CURVE
# =====================================================
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(
        n_estimators=int(best_n_est),
        max_depth=int(best_depth),
        random_state=42,
        n_jobs=-1,
    ),
    X_train_bal,
    y_train_bal,
    cv=5,
    scoring="f1_macro",
    train_sizes=np.linspace(0.1, 1.0, 5),
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, label="Training F1", marker="o")
plt.plot(train_sizes, val_mean, label="Validation F1", marker="s")
plt.xlabel("Training Set Size")
plt.ylabel("Macro F1 Score")
plt.title("Learning Curve (Random Forest)")
plt.legend()
plt.grid(True)
learn_path = f"logs/plots/learn_curve_RF_sklearn_{timestamp}.png"
plt.savefig(learn_path, bbox_inches="tight")
plt.show()
