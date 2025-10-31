import numpy as np
import pickle
from datetime import datetime
from itertools import combinations
from collections import Counter


class SVM_SMO:
    def __init__(
        self, C=1.0, tol=1e-3, max_passes=5, kernel="linear", kernel_params=None
    ):
        """
        Soft-Margin SVM using Sequential Minimal Optimization (SMO)
        Supports linear, polynomial, and RBF kernels
        Handles multi-class via One-vs-One strategy
        """
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.kernel = kernel
        self.kernel_params = kernel_params if kernel_params else {}
        self.models = {}  # store binary models for each class pair

    # ---------- Kernel Function ----------
    def kernel_function(self, X1, X2):
        if self.kernel == "linear":
            return np.dot(X1, X2.T)
        elif self.kernel == "poly":
            degree = self.kernel_params.get("degree", 3)
            coef0 = self.kernel_params.get("coef0", 1)
            return (np.dot(X1, X2.T) + coef0) ** degree
        elif self.kernel == "rbf":
            gamma = self.kernel_params.get("gamma", 1 / X1.shape[1])
            # RBF: exp(-gamma * ||x1 - x2||^2)
            X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
            dist = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
            return np.exp(-gamma * dist)
        else:
            raise ValueError("Unsupported kernel type")

    def train_binary(self, X, y):
        """
        Train a binary SVM using simplified SMO algorithm
        y ∈ {+1, -1}
        """
        m, n = X.shape
        alphas = np.zeros(m)
        b = 0
        passes = 0

        K = self.kernel_function(X, X)

        while passes < self.max_passes:
            num_changed = 0
            for i in range(m):
                # Compute Ei = f(x_i) - y_i
                f_i = np.sum(alphas * y * K[:, i]) + b
                E_i = f_i - y[i]

                # Check KKT conditions
                if (y[i] * E_i < -self.tol and alphas[i] < self.C) or (
                    y[i] * E_i > self.tol and alphas[i] > 0
                ):
                    # Select j ≠ i randomly
                    j = np.random.choice([x for x in range(m) if x != i])
                    f_j = np.sum(alphas * y * K[:, j]) + b
                    E_j = f_j - y[j]

                    alpha_i_old, alpha_j_old = alphas[i], alphas[j]

                    # Compute bounds L and H
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])

                    if L == H:
                        continue

                    # Compute eta (the second derivative of objective)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Update α_j
                    alphas[j] -= y[j] * (E_i - E_j) / eta
                    alphas[j] = np.clip(alphas[j], L, H)

                    # If change in α_j is too small, skip
                    if abs(alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update α_i
                    alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                    # Compute new threshold b
                    b1 = (
                        b
                        - E_i
                        - y[i] * (alphas[i] - alpha_i_old) * K[i, i]
                        - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                    )
                    b2 = (
                        b
                        - E_j
                        - y[i] * (alphas[i] - alpha_i_old) * K[i, j]
                        - y[j] * (alphas[j] - alpha_j_old) * K[j, j]
                    )

                    # Choose b based on whether alphas are within margin
                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2

                    num_changed += 1

            passes = passes + 1 if num_changed == 0 else 0

        # Save support vectors
        idx = alphas > 1e-5
        model = {"X": X[idx], "y": y[idx], "alphas": alphas[idx], "b": b}
        return model

    # ---------- Fit Multi-class (One-vs-One) ----------
    def fit(self, X, y):
        """
        Train multi-class SVM using One-vs-One
        """
        classes = np.unique(y)
        for c1, c2 in combinations(classes, 2):
            # Select samples belonging to class c1 or c2
            idx = np.where((y == c1) | (y == c2))[0]
            X_pair, y_pair = X[idx], y[idx]
            y_pair = np.where(y_pair == c1, 1, -1)

            print(f"Training pair ({c1}, {c2}) ...")
            model = self.train_binary(X_pair, y_pair)
            self.models[(c1, c2)] = model
        return self

    def predict(self, X):
        """
        Predict class labels using voting over One-vs-One classifiers
        """
        votes = np.zeros((X.shape[0], len(self.models)))

        # Vote accumulation
        classes = list(set([c for pair in self.models.keys() for c in pair]))
        preds = []

        for idx, ((c1, c2), model) in enumerate(self.models.items()):
            K = self.kernel_function(X, model["X"])
            pred = np.sign(
                np.sum(model["alphas"] * model["y"] * K, axis=1) + model["b"]
            )
            vote = np.where(pred > 0, c1, c2)
            preds.append(vote)

        preds = np.array(preds).T  # shape: (n_samples, n_models)

        # Majority vote
        y_pred = []
        for row in preds:
            y_pred.append(Counter(row).most_common(1)[0][0])
        return np.array(y_pred)

    def confusion_matrix(self, y_true, y_pred):
        classes = np.unique(np.concatenate((y_true, y_pred)))
        n = len(classes)
        matrix = np.zeros((n, n), dtype=int)
        for i, t in enumerate(y_true):
            j = np.where(classes == y_pred[i])[0][0]
            k = np.where(classes == t)[0][0]
            matrix[k, j] += 1
        return matrix, classes

    def classification_report(self, y_true, y_pred):
        matrix, classes = self.confusion_matrix(y_true, y_pred)
        n_classes = len(classes)
        precision, recall, f1 = (
            np.zeros(n_classes),
            np.zeros(n_classes),
            np.zeros(n_classes),
        )
        support = matrix.sum(axis=1)

        for i in range(n_classes):
            tp = matrix[i, i]
            fp = matrix[:, i].sum() - tp
            fn = matrix[i, :].sum() - tp
            precision[i] = tp / (tp + fp) if tp + fp > 0 else 0
            recall[i] = tp / (tp + fn) if tp + fn > 0 else 0
            if precision[i] + recall[i] > 0:
                f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
            else:
                f1[i] = 0

        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)

        weights = support / np.sum(support)
        weighted_precision = np.sum(precision * weights)
        weighted_recall = np.sum(recall * weights)
        weighted_f1 = np.sum(f1 * weights)

        report = {
            "classes": classes.tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
            "macro_avg": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
            },
            "weighted_avg": {
                "precision": weighted_precision,
                "recall": weighted_recall,
                "f1": weighted_f1,
            },
            "accuracy": np.mean(y_true == y_pred),
        }
        return report

    def get_params(self, deep=True):
        return {
            "C": self.C,
            "tol": self.tol,
            "max_passes": self.max_passes,
            "kernel": self.kernel,
            "kernel_params": self.kernel_params,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def save_checkpoint(self, path=None):
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"checkpoints/svm_{ts}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Model saved at: {path}")

    @staticmethod
    def load_checkpoint(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from: {path}")
        return model
