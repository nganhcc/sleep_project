import numpy as np
import matplotlib.pyplot as plt
from src.models.RandomForest import RandomForest
from datetime import datetime
from sklearn.model_selection import validation_curve, learning_curve
import os, time, csv

from src.utils.balance_data import balance_data
from src.utils.k_fold_split import k_fold_split

# =====================================================
# SETUP LOG PATHS
# =====================================================
os.makedirs("logs", exist_ok=True)
mean_log_path = "logs/hparam_results_RF.csv"
fold_log_path = "logs/fold_results_RF.csv"
os.makedirs("logs/plots", exist_ok=True)

if not os.path.exists(fold_log_path):
    with open(fold_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "fold",
                "mean_acc",
                "mean_f1",
                "mean_time",
            ]
        )
if not os.path.exists(mean_log_path):
    with open(mean_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "mean_acc",
                "mean_f1",
                "mean_time",
            ]
        )

# =====================================================
# CONFIG
# =====================================================
N_SPLITS = 10
PARAM_GRID = {
    "n_estimators": [5, 10, 20, 30],
    "max_depth": [5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
}
np.random.seed(42)

X_train = np.loadtxt("data/processed/train/X_train.csv", delimiter=",")
y_train = np.loadtxt("data/processed/train/y_train.csv", delimiter=",")
X_test = np.loadtxt("data/processed/test/X_test.csv", delimiter=",")
y_test = np.loadtxt("data/processed/test/y_test.csv", delimiter=",")

folds = k_fold_split(X_train, y_train, N_SPLITS)

results = []
f1_trend = {}

# =====================================================
# GRID SEARCH
# =====================================================
for n_estimators in PARAM_GRID["n_estimators"]:
    for max_depth in PARAM_GRID["max_depth"]:
        for min_samples_split in PARAM_GRID["min_samples_split"]:
            fold_accs, fold_f1s, fold_times = [], [], []
            key = f"trees={n_estimators}_depth={max_depth}_split={min_samples_split}"
            f1_trend[key] = []

            print(
                f"\n Testing n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}"
            )

            for fold_idx in range(N_SPLITS):
                val_idx = folds[fold_idx]
                train_idx = np.hstack(
                    [folds[i] for i in range(N_SPLITS) if i != fold_idx]
                )

                X_tr_raw, y_tr_raw = X_train[train_idx], y_train[train_idx]
                X_val, y_val = X_train[val_idx], y_train[val_idx]

                X_tr, y_tr = balance_data(X_tr_raw, y_tr_raw)

                rf = RandomForest(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                )

                start_time = time.time()
                rf.fit(X_tr, y_tr)
                elapsed = time.time() - start_time

                y_pred = rf.predict(X_val)
                report = rf.classification_report(y_val, y_pred)

                acc = report["accuracy"]
                f1 = report["macro_avg"]["f1"]
                fold_accs.append(acc)
                fold_f1s.append(f1)
                fold_times.append(elapsed)
                f1_trend[key].append(f1)

                print(
                    f"  Fold {fold_idx+1}/{N_SPLITS} → Acc={acc:.3f}, F1={f1:.3f}, Time={elapsed:.2f}s"
                )

                with open(fold_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            n_estimators,
                            max_depth,
                            min_samples_split,
                            fold_idx + 1,
                            round(acc, 4),
                            round(f1, 4),
                            round(elapsed, 2),
                        ]
                    )

            mean_acc = np.mean(fold_accs)
            mean_f1 = np.mean(fold_f1s)
            mean_time = np.mean(fold_times)
            results.append(
                (
                    n_estimators,
                    max_depth,
                    min_samples_split,
                    mean_acc,
                    mean_f1,
                    mean_time,
                )
            )
            print(
                f"→ Mean Accuracy={mean_acc:.3f}, Mean F1={mean_f1:.3f}, Time={mean_time:.2f}s"
            )

            with open(mean_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        n_estimators,
                        max_depth,
                        min_samples_split,
                        round(mean_acc, 4),
                        round(mean_f1, 4),
                        round(mean_time, 2),
                    ]
                )

# =====================================================
# BEST PARAMS
# =====================================================
results = np.array(results, dtype=object)
best_idx = np.argmax(results[:, 4])  # sort by F1
best_params = results[best_idx]
best_n_estimators, best_depth, best_min_split = (
    best_params[0],
    best_params[1],
    best_params[2],
)

print("\n Best Params:")
print(f"  n_estimators={best_n_estimators}")
print(f"  max_depth={best_depth}")
print(f"  min_samples_split={best_min_split}")
print(f"  Mean F1={best_params[4]:.3f}, Mean Time={best_params[5]:.2f}s")

# =====================================================
# TRAIN BEST MODEL ON FULL TRAIN SET
# =====================================================
X_train_bal, y_train_bal = balance_data(X_train, y_train)
best_rf = RandomForest(
    n_estimators=best_n_estimators,
    max_depth=best_depth,
    min_samples_split=best_min_split,
)
best_rf.fit(X_train_bal, y_train_bal)
os.makedirs("checkpoints", exist_ok=True)
best_rf.save_checkpoint("checkpoints/best_random_forest.pkl")

# =====================================================
# TEST PERFORMANCE
# =====================================================
y_pred_test = best_rf.predict(X_test)
report_test = best_rf.classification_report(y_test, y_pred_test)

print("\n Test Performance:")
print(f"Accuracy: {report_test['accuracy']:.3f}")
print(f"F1: {report_test['macro_avg']['f1']:.3f}")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# =====================================================
# F1 TREND PLOT
# =====================================================
plt.figure(figsize=(10, 5))
for key, values in f1_trend.items():
    plt.plot(range(1, N_SPLITS + 1), values, marker="o", label=key)
plt.xlabel("Fold")
plt.ylabel("F1 Score")
plt.title("Fold-by-Fold F1 Trend for Each Hyperparameter Setting (Random Forest)")
plt.legend()
plt.grid(True)
f1_path = f"logs/plots/f1_trend_RF_{timestamp}.png"
plt.savefig(f1_path, bbox_inches="tight")
plt.show()

# =====================================================
# VALIDATION CURVE
# =====================================================
param_range = np.arange(2, 21, 2)
train_scores, val_scores = validation_curve(
    RandomForest(n_estimators=best_n_estimators, min_samples_split=best_min_split),
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
plt.xlabel("n_estimators")
plt.ylabel("Macro F1 Score")
plt.title("Validation Curve (Effect of n_estimators)")
plt.legend()
plt.grid(True)
val_path = f"logs/plots/val_curve_RF_{timestamp}.png"
plt.savefig(val_path, bbox_inches="tight")
plt.show()

# =====================================================
# LEARNING CURVE
# =====================================================
train_sizes, train_scores, val_scores = learning_curve(
    best_rf,
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
plt.title("Learning Curve (Effect of Training Data Size)")
plt.legend()
plt.grid(True)
learn_path = f"logs/plots/learn_curve_RF_{timestamp}.png"
plt.savefig(learn_path, bbox_inches="tight")
plt.show()
