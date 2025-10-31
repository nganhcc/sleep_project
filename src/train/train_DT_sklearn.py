import numpy as np
import matplotlib.pyplot as plt
import os, time, csv
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, validation_curve, learning_curve
from src.utils.balance_data import balance_data  # keep your balancing logic

# =====================================================
# CONFIGURATION
# =====================================================
os.makedirs("logs", exist_ok=True)
mean_log_path = "logs/hparam_results_DT_sklearn.csv"
fold_log_path = "logs/fold_results_DT_sklearn.csv"

# Write headers if files don't exist
if not os.path.exists(fold_log_path):
    with open(fold_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "max_depth", "min_samples_split", "fold", "acc", "f1", "time"]
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

# =====================================================
# LOAD DATA
# =====================================================
X_train = np.loadtxt("data/processed/train/X_train.csv", delimiter=",")
y_train = np.loadtxt("data/processed/train/y_train.csv", delimiter=",")
X_test = np.loadtxt("data/processed/test/X_test.csv", delimiter=",")
y_test = np.loadtxt("data/processed/test/y_test.csv", delimiter=",")

# =====================================================
# CROSS VALIDATION LOOP
# =====================================================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
results = []
f1_trend = {}

for max_depth in PARAM_GRID["max_depth"]:
    for min_samples_split in PARAM_GRID["min_samples_split"]:
        fold_accs, fold_f1s, fold_times = [], [], []
        key = f"depth={max_depth}_split={min_samples_split}"
        f1_trend[key] = []

        print(
            f"\n Testing max_depth={max_depth}, min_samples_split={min_samples_split}"
        )

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr_raw, y_tr_raw = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            # Balance ONLY training data
            X_tr, y_tr = balance_data(X_tr_raw, y_tr_raw)

            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
            )

            start_time = time.time()
            model.fit(X_tr, y_tr)
            elapsed = time.time() - start_time

            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="macro")

            fold_accs.append(acc)
            fold_f1s.append(f1)
            fold_times.append(elapsed)
            f1_trend[key].append(f1)

            print(
                f"  Fold {fold_idx+1}/{N_SPLITS} → Acc={acc:.3f}, F1={f1:.3f}, Time={elapsed:.2f}s"
            )

            # Log per fold
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

# =====================================================
# BEST PARAMETER SELECTION
# =====================================================
results = np.array(results, dtype=object)
best_idx = np.argmax(results[:, 3])  # best by mean F1
best_params = results[best_idx]
best_depth, best_min_split = best_params[0], best_params[1]

print("\n Best Params:")
print(f"  max_depth={best_depth}")
print(f"  min_samples_split={best_min_split}")
print(f"  Mean F1={best_params[3]:.3f}, Mean Time={best_params[4]:.2f}s")

# =====================================================
# TRAIN BEST MODEL ON FULL TRAIN SET (BALANCED)
# =====================================================
X_train_bal, y_train_bal = balance_data(X_train, y_train)
best_model = DecisionTreeClassifier(
    max_depth=best_depth, min_samples_split=best_min_split, random_state=42
)
best_model.fit(X_train_bal, y_train_bal)

# Save model (using joblib)
import joblib

os.makedirs("checkpoints", exist_ok=True)
joblib.dump(best_model, "checkpoints/best_decision_tree_sklearn.pkl")

# =====================================================
# TEST EVALUATION
# =====================================================
y_pred_test = best_model.predict(X_test)
report_test = classification_report(y_test, y_pred_test, output_dict=True)

print("\n Test Performance:")
print(f"Accuracy: {report_test['accuracy']:.3f}")
print(f"Macro F1: {report_test['macro avg']['f1-score']:.3f}")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# =====================================================
# F1 TREND VISUALIZATION
# =====================================================
plt.figure(figsize=(20, 8))
for key, values in f1_trend.items():
    plt.plot(range(1, N_SPLITS + 1), values, marker="o", label=key)
plt.xlabel("Fold")
plt.ylabel("F1 Score")
plt.title("Fold-by-Fold F1 Trend for Each Hyperparameter Setting (sklearn)")
plt.legend()
plt.grid(True)

f1_path = f"logs/plots/f1_trend_DT_sklearn_{timestamp}.png"
plt.savefig(f1_path, bbox_inches="tight")

plt.show()
# =====================================================
# VALIDATION CURVE
# =====================================================
param_range = np.arange(2, 21, 2)
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(min_samples_split=best_min_split, random_state=42),
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

val_path = f"logs/plots/val_curve_DT_sklearn_{timestamp}.png"
plt.savefig(val_path, bbox_inches="tight")
plt.show()

# =====================================================
# LEARNING CURVE
# =====================================================
train_sizes, train_scores, val_scores = learning_curve(
    DecisionTreeClassifier(
        max_depth=best_depth, min_samples_split=best_min_split, random_state=42
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
plt.title("Learning Curve (Effect of Training Data Size)")
plt.legend()
plt.grid(True)

learn_path = f"logs/plots/learn_curve_DT_sklearn_{timestamp}.png"
plt.savefig(learn_path, bbox_inches="tight")

plt.show()
