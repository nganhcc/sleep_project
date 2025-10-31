import numpy as np
import matplotlib.pyplot as plt
import os, time, csv
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, validation_curve, learning_curve
from src.utils.balance_data import balance_data
import joblib

# =====================================================
# CONFIGURATION
# =====================================================
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/plots", exist_ok=True)

mean_log_path = "logs/hparam_results_KNN_sklearn.csv"
fold_log_path = "logs/fold_results_KNN_sklearn.csv"

# CSV headers
if not os.path.exists(fold_log_path):
    with open(fold_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "n_neighbors", "fold", "acc", "f1", "time"])

if not os.path.exists(mean_log_path):
    with open(mean_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "n_neighbors", "mean_acc", "mean_f1", "mean_time"]
        )

N_SPLITS = 10
PARAM_GRID = {"n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15]}
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

for n_neighbors in PARAM_GRID["n_neighbors"]:
    fold_accs, fold_f1s, fold_times = [], [], []
    f1_trend[n_neighbors] = []

    print(f"\n Testing n_neighbors={n_neighbors}")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr_raw, y_tr_raw = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        X_tr, y_tr = balance_data(X_tr_raw, y_tr_raw)

        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)

        start_time = time.time()
        model.fit(X_tr, y_tr)
        elapsed = time.time() - start_time

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")

        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_times.append(elapsed)
        f1_trend[n_neighbors].append(f1)

        print(
            f"  Fold {fold_idx+1}/{N_SPLITS} → Acc={acc:.3f}, F1={f1:.3f}, Time={elapsed:.2f}s"
        )

        # Log per fold
        with open(fold_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    n_neighbors,
                    fold_idx + 1,
                    round(acc, 4),
                    round(f1, 4),
                    round(elapsed, 2),
                ]
            )

    mean_acc = np.mean(fold_accs)
    mean_f1 = np.mean(fold_f1s)
    mean_time = np.mean(fold_times)
    results.append((n_neighbors, mean_acc, mean_f1, mean_time))

    print(
        f"→ Mean Accuracy={mean_acc:.3f}, Mean F1={mean_f1:.3f}, Time={mean_time:.2f}s"
    )

    with open(mean_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                n_neighbors,
                round(mean_acc, 4),
                round(mean_f1, 4),
                round(mean_time, 2),
            ]
        )

# =====================================================
# BEST PARAMETER SELECTION
# =====================================================
results = np.array(results, dtype=object)
best_idx = np.argmax(results[:, 2])  # by mean F1
best_n = results[best_idx][0]
print("\n Best Params:")
print(f"  n_neighbors={best_n}")
print(f"  Mean F1={results[best_idx][2]:.3f}")

# =====================================================
# TRAIN BEST MODEL ON FULL TRAIN SET
# =====================================================
X_train_bal, y_train_bal = balance_data(X_train, y_train)
best_model = KNeighborsClassifier(n_neighbors=int(best_n), n_jobs=-1)
best_model.fit(X_train_bal, y_train_bal)

os.makedirs("checkpoints", exist_ok=True)
joblib.dump(best_model, "checkpoints/best_knn_sklearn.pkl")

# =====================================================
# TEST EVALUATION
# =====================================================
y_pred_test = best_model.predict(X_test)
report_test = classification_report(y_test, y_pred_test, output_dict=True)

print("\n Test Performance:")
print(f"Accuracy: {report_test['accuracy']:.3f}")
print(f"Macro F1: {report_test['macro avg']['f1-score']:.3f}")

# =====================================================
# VISUALIZATIONS (Auto Save)
# =====================================================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- F1 trend ---
plt.figure(figsize=(10, 5))
for key, values in f1_trend.items():
    plt.plot(range(1, N_SPLITS + 1), values, marker="o", label=f"k={key}")
plt.xlabel("Fold")
plt.ylabel("F1 Score")
plt.title("Fold-by-Fold F1 Trend for KNN")
plt.legend()
plt.grid(True)
f1_path = f"logs/plots/f1_trend_knn_sklearn_{timestamp}.png"
plt.savefig(f1_path, bbox_inches="tight")
plt.show()
print(f"✅ Saved F1 trend plot → {f1_path}")

# --- Validation Curve ---
param_range = range(1, 21)
train_scores, val_scores = validation_curve(
    KNeighborsClassifier(),
    X_train,
    y_train,
    param_name="n_neighbors",
    param_range=param_range,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
)
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(param_range, train_mean, label="Train F1", marker="o")
plt.plot(param_range, val_mean, label="Validation F1", marker="o")
plt.title("Validation Curve (KNN)")
plt.xlabel("n_neighbors")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
val_path = f"logs/plots/validation_curve_knn_sklearn_{timestamp}.png"
plt.savefig(val_path, bbox_inches="tight")
plt.show()
print(f"✅ Saved Validation Curve → {val_path}")

# --- Learning Curve ---
train_sizes, train_scores, val_scores = learning_curve(
    KNeighborsClassifier(n_neighbors=int(best_n), n_jobs=-1),
    X_train,
    y_train,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 6),
)
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, "o-", label="Train F1")
plt.plot(train_sizes, val_mean, "o-", label="Validation F1")
plt.title(f"Learning Curve (KNN, best n_neighbors={best_n})")
plt.xlabel("Training Size")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
learn_path = f"logs/plots/learning_curve_knn_sklearn_{timestamp}.png"
plt.savefig(learn_path, bbox_inches="tight")
plt.show()
print(f"✅ Saved Learning Curve → {learn_path}")
