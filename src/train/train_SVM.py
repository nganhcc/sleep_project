from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os, time, csv, pickle

from sklearn.model_selection import validation_curve, learning_curve, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

from src.models.SVM import SVM_SMO as SVM  # ← your custom SVM class
from src.utils.balance_data import balance_data
from src.utils.k_fold_split import k_fold_split

# =====================================================
# SETUP
# =====================================================
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/plots", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

mean_log_path = "logs/hparam_results_SVM.csv"
fold_log_path = "logs/fold_results_SVM.csv"

if not os.path.exists(fold_log_path):
    with open(fold_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "C", "kernel", "fold", "acc", "f1", "time"])

if not os.path.exists(mean_log_path):
    with open(mean_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "C", "kernel", "mean_acc", "mean_f1", "mean_time"]
        )

# =====================================================
# CONFIG
# =====================================================
N_SPLITS = 5
PARAM_GRID = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf", "poly"]}
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
for C_val in PARAM_GRID["C"]:
    for kernel_type in PARAM_GRID["kernel"]:
        fold_accs, fold_f1, fold_times = [], [], []
        key = f"C={C_val}, kernel={kernel_type}"
        f1_trend[key] = []

        print(f"\nTesting C={C_val}, kernel={kernel_type}")

        for fold_idx in range(N_SPLITS):
            val_idx = folds[fold_idx]
            train_idx = np.hstack([folds[i] for i in range(N_SPLITS) if i != fold_idx])

            X_val, y_val = X_train[val_idx], y_train[val_idx]
            X_tr_raw, y_tr_raw = X_train[train_idx], y_train[train_idx]
            X_tr, y_tr = balance_data(X_tr_raw, y_tr_raw)

            model = SVM(C=C_val, kernel=kernel_type)

            start_time = time.time()
            model.fit(X_tr, y_tr)
            elapsed = time.time() - start_time

            y_pred = model.predict(X_val)

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
                        C_val,
                        kernel_type,
                        fold_idx + 1,
                        round(acc, 4),
                        round(f1, 4),
                        round(elapsed, 2),
                    ]
                )

        mean_acc = np.mean(fold_accs)
        mean_f1 = np.mean(fold_f1)
        mean_time = np.mean(fold_times)
        results.append((C_val, kernel_type, mean_acc, mean_f1, mean_time))

        print(
            f"→ Mean Accuracy={mean_acc:.3f}, Mean F1={mean_f1:.3f}, Time={mean_time:.2f}s"
        )

        with open(mean_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    C_val,
                    kernel_type,
                    round(mean_acc, 4),
                    round(mean_f1, 4),
                    round(mean_time, 2),
                ]
            )

# =====================================================
# BEST PARAMETER
# =====================================================
results = np.array(results, dtype=object)
best_idx = np.argmax(results[:, 3])
best_params = results[best_idx]
best_C, best_kernel = best_params[0], best_params[1]

print("\nBest Params:")
print(f"  C={best_C}, kernel={best_kernel}")
print(f"  Mean F1={best_params[3]:.3f}, Mean Time={best_params[4]:.2f}s")

# =====================================================
# TRAIN BEST MODEL ON FULL DATA
# =====================================================
X_train_bal, y_train_bal = balance_data(X_train, y_train)
best_svm = SVM(C=float(best_C), kernel=best_kernel)
best_svm.fit(X_train_bal, y_train_bal)

with open("checkpoints/best_svm.pkl", "wb") as f:
    pickle.dump(best_svm, f)

# =====================================================
# TEST EVALUATION
# =====================================================
y_pred_test = best_svm.predict(X_test)
acc_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test, average="macro")

print("\nTest Performance:")
print(f"Accuracy: {acc_test:.3f}")
print(f"F1: {f1_test:.3f}")

# =====================================================
# PLOTTING F1 TREND
# =====================================================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

plt.figure(figsize=(10, 5))
for key, values in f1_trend.items():
    plt.plot(range(1, N_SPLITS + 1), values, marker="o", label=key)
plt.xlabel("Fold")
plt.ylabel("F1 Score")
plt.title("Fold-by-Fold F1 Trend for Each Hyperparameter Setting (Custom SVM)")
plt.legend()
plt.grid(True)
f1_path = f"logs/plots/f1_trend_SVM_{timestamp}.png"
plt.savefig(f1_path, bbox_inches="tight")
plt.show()

# =====================================================
# VALIDATION CURVE (Effect of kernel)
# =====================================================
kernel_list = ["linear", "poly", "rbf"]
train_scores, val_scores = [], []

for kernel in kernel_list:
    print(f"Evaluating kernel = {kernel}")

    model = SVM(C=float(best_C), kernel=kernel)
    model.fit(X_train_bal, y_train_bal)
    y_train_pred = model.predict(X_train_bal)
    train_f1 = f1_score(y_train_bal, y_train_pred, average="macro")

    val_f1 = cross_val_score(
        model, X_train_bal, y_train_bal, cv=5, scoring="f1_macro"
    ).mean()
    train_scores.append(train_f1)
    val_scores.append(val_f1)

plt.figure(figsize=(8, 5))
plt.plot(kernel_list, train_scores, label="Training F1", marker="o")
plt.plot(kernel_list, val_scores, label="Validation F1", marker="s")
plt.xlabel("Kernel")
plt.ylabel("Macro F1 Score")
plt.title(f"Validation Curve (Effect of kernel on Custom SVM, C={best_C})")
plt.legend()
plt.grid(True)
plt.tight_layout()
val_kernel_path = f"logs/plots/val_curve_kernel_SVM{timestamp}.png"
plt.savefig(val_kernel_path, bbox_inches="tight")
plt.show()

# =====================================================
# VALIDATION CURVE (Effect of C)
# =====================================================
C_values = np.logspace(-2, 2, 6)  # [0.01, 0.1, 1, 10, 100]
best_kernel = best_params[0] if best_params else "rbf"

model = SVM(kernel=best_kernel)

train_scores, val_scores = validation_curve(
    model,
    X_train_bal,
    y_train_bal,
    param_name="C",
    param_range=C_values,
    cv=5,
    scoring="f1_macro",
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.semilogx(C_values, train_mean, label="Training F1", marker="o")
plt.semilogx(C_values, val_mean, label="Validation F1", marker="s")
plt.xlabel("C (log scale)")
plt.ylabel("Macro F1 Score")
plt.title(f"Validation Curve (Effect of C) - kernel={best_kernel}")
plt.legend()
plt.grid(True)
plt.tight_layout()

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
val_C_path = f"logs/plots/val_curve_C_SVM_{timestamp}.png"
os.makedirs("logs/plots", exist_ok=True)
plt.savefig(val_C_path, bbox_inches="tight")
plt.show()

# =====================================================
# LEARNING CURVE
# =====================================================
train_sizes, train_scores, val_scores = learning_curve(
    best_svm,  # sklearn-compatible model
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
plt.title("Learning Curve (Custom SVM)")
plt.legend()
plt.grid(True)
learn_path = f"logs/plots/learn_curve_SVM_{timestamp}.png"
plt.savefig(learn_path, bbox_inches="tight")
plt.show()
