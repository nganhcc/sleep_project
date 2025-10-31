import numpy as np
import matplotlib.pyplot as plt
from src.models.DecisionTree import DecisionTree
from datetime import datetime
from sklearn.model_selection import validation_curve, learning_curve
import os, time, csv

from src.utils.balance_data import balance_data
from src.utils.k_fold_split import k_fold_split

os.makedirs("logs", exist_ok=True)
mean_log_path = "logs/hparam_results_DT.csv"
fold_log_path = "logs/fold_results_DT.csv"

if not os.path.exists(fold_log_path):
    with open(fold_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
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
                "max_depth",
                "min_samples_split",
                "mean_acc",
                "mean_f1",
                "mean_time",
            ]
        )
N_SPLITS = 10
PARAM_GRID = {
    "max_depth": [5, 7, 10, 13, 15, 17, 20],
    "min_samples_split": [2, 5, 7, 10],
}
np.random.seed(42)

X_train = np.loadtxt("data/processed/train/X_train.csv", delimiter=",")
y_train = np.loadtxt("data/processed/train/y_train.csv", delimiter=",")
X_test = np.loadtxt("data/processed/test/X_test.csv", delimiter=",")
y_test = np.loadtxt("data/processed/test/y_test.csv", delimiter=",")

folds = k_fold_split(X_train, y_train, N_SPLITS)

results = []
f1_trend = {}  # store fold-wise F1 trend

for max_depth in PARAM_GRID["max_depth"]:
    for min_samples_split in PARAM_GRID["min_samples_split"]:
        fold_accs, fold_f1s, fold_times = [], [], []
        key = f"depth={max_depth}_split={min_samples_split}"
        f1_trend[key] = []

        print(
            f"\n Testing max_depth={max_depth}, min_samples_split={min_samples_split}"
        )

        for fold_idx in range(N_SPLITS):
            val_idx = folds[fold_idx]
            train_idx = np.hstack([folds[i] for i in range(N_SPLITS) if i != fold_idx])

            X_tr_raw, y_tr_raw = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            # Balance ONLY the training part
            X_tr, y_tr = balance_data(X_tr_raw, y_tr_raw)

            tree = DecisionTree(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                feature_idx=list(range(X_tr.shape[1])),
            )

            start_time = time.time()
            tree.fit(X_tr, y_tr)
            elapsed = time.time() - start_time

            y_pred = tree.predict(X_val)
            report = tree.classification_report(y_val, y_pred)

            acc = report["accuracy"]
            f1 = report["macro_avg"]["f1"]
            fold_accs.append(acc)
            fold_f1s.append(f1)
            fold_times.append(elapsed)
            f1_trend[key].append(f1)

            print(
                f"  Fold {fold_idx+1}/{N_SPLITS} → "
                f"Acc={acc:.3f}, F1={f1:.3f}, Time={elapsed:.2f}s"
            )
            with open(fold_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        results.append((max_depth, min_samples_split, mean_acc, mean_f1, mean_time))
        print(
            f"→ Mean Accuracy={mean_acc:.3f}, Mean F1={mean_f1:.3f}, Time={mean_time:.2f}s"
        )

        with open(mean_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    max_depth,
                    min_samples_split,
                    round(mean_acc, 4),
                    round(mean_f1, 4),
                    round(mean_time, 2),
                ]
            )

results = np.array(results, dtype=object)
best_idx = np.argmax(results[:, 3])  # by F1 score
best_params = results[best_idx]
best_depth, best_min_split = best_params[0], best_params[1]

print("\n Best Params:")
print(f"  max_depth={best_depth}")
print(f"  min_samples_split={best_min_split}")
print(f"  Mean F1={best_params[3]:.3f}, Mean Time={best_params[4]:.2f}s")

# TRAIN BEST MODEL ON FULL TRAIN SET (BALANCED)

X_train_bal, y_train_bal = balance_data(X_train, y_train)
best_tree = DecisionTree(
    max_depth=best_depth,
    min_samples_split=best_min_split,
    feature_idx=list(range(X_train.shape[1])),
)
best_tree.fit(X_train_bal, y_train_bal)
os.makedirs("checkpoints", exist_ok=True)
best_tree.save_checkpoint("checkpoints/best_decision_tree.pkl")

# TEST EVALUATION

y_pred_test = best_tree.predict(X_test)
report_test = best_tree.classification_report(y_test, y_pred_test)

print("\n Test Performance:")
print(f"Accuracy: {report_test['accuracy']:.3f}")

tmp = report_test["macro_avg"]["f1"]
print(f"F1:{tmp:.3f}")


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- F1 trend across folds ---
plt.figure(figsize=(10, 5))
for key, values in f1_trend.items():
    plt.plot(range(1, N_SPLITS + 1), values, marker="o", label=key)
plt.xlabel("Fold")
plt.ylabel("F1 Score")
plt.title("Fold-by-Fold F1 Trend for Each Hyperparameter Setting")
plt.legend()
plt.grid(True)
f1_path = f"logs/plots/f1_trend_DT_{timestamp}.png"
plt.savefig(f1_path, bbox_inches="tight")
plt.show()

# =====================================================
# VALIDATION CURVE
# =====================================================
param_range = np.arange(2, 21, 2)
train_scores, val_scores = validation_curve(
    DecisionTree(min_samples_split=best_min_split),
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
plt.title("Validation Curve (Effect of max_depth)")
plt.legend()
plt.grid(True)

val_path = f"logs/plots/val_curve_DT_{timestamp}.png"
plt.savefig(val_path, bbox_inches="tight")
plt.show()

# =====================================================
# LEARNING CURVE
# =====================================================
train_sizes, train_scores, val_scores = learning_curve(
    DecisionTree(max_depth=best_depth, min_samples_split=best_min_split),
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

learn_path = f"logs/plots/learn_curve_DT_{timestamp}.png"
plt.savefig(learn_path, bbox_inches="tight")
plt.show()
