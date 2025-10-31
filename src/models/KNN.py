import numpy as np
from datetime import datetime
import pickle
from collections import Counter


class KNN:
    def __init__(self, k=5, distance_metric="euclidean", weights="uniform"):
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None

    # =====================================================
    # --- Fit & Predict ---
    # =====================================================
    def fit(self, X, y):
        """Store training data"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self  # sklearn compatibility

    def predict(self, X):
        """Predict class labels for given samples"""
        X = np.array(X)
        preds = []
        for x in X:
            distances = self._compute_distances(x)
            k_idx = np.argsort(distances)[: self.k]
            k_labels = self.y_train[k_idx]

            if self.weights == "distance":
                k_distances = distances[k_idx]
                weights = 1 / (k_distances + 1e-8)
                vote_counts = {}
                for label, w in zip(k_labels, weights):
                    vote_counts[label] = vote_counts.get(label, 0) + w
                preds.append(max(vote_counts, key=vote_counts.get))
            else:  # uniform weights
                preds.append(Counter(k_labels).most_common(1)[0][0])

        return np.array(preds)

    # =====================================================
    # --- Distance metrics ---
    # =====================================================
    def _compute_distances(self, x):
        if self.distance_metric == "manhattan":
            return np.sum(np.abs(self.X_train - x), axis=1)
        elif self.distance_metric == "minkowski":
            p = 3  # you can parameterize this if needed
            return np.sum(np.abs(self.X_train - x) ** p, axis=1) ** (1 / p)
        else:  # euclidean
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    # =====================================================
    # --- Evaluation ---
    # =====================================================
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

    def save_checkpoint(self, path=None):
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"checkpoints/knn_{ts}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Model saved at: {path}")

    @staticmethod
    def load_checkpoint(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from: {path}")
        return model

    def get_params(self, deep=True):
        return {
            "k": self.k,
            "distance_metric": self.distance_metric,
            "weights": self.weights,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
