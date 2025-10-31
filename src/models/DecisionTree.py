import numpy as np
from datetime import datetime
import pickle


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_idx = None

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p))

    def _information_gain(self, y, left_y, right_y):
        n = len(y)
        if len(left_y) == 0 or len(right_y) == 0:
            return 0
        p_left, p_right = len(left_y) / n, len(right_y) / n
        return self._entropy(y) - (
            p_left * self._entropy(left_y) + p_right * self._entropy(right_y)
        )

    def _majority_vote(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _build(self, X, y, depth=0):
        if (
            len(np.unique(y)) == 1
            or depth >= self.max_depth
            or len(y) < self.min_samples_split
        ):
            return Node(value=self._majority_vote(y))

        _, n_features = X.shape
        best_gain, best_feature, best_threshold = 0, None, None
        best_left_idx, best_right_idx = None, None

        for feature in range(n_features):
            values = np.unique(X[:, feature])
            if len(values) == 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx
                gain = self._information_gain(y, y[left_idx], y[right_idx])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_left_idx, best_right_idx = left_idx, right_idx

        if best_gain == 0:
            return Node(value=self._majority_vote(y))

        left = self._build(X[best_left_idx], y[best_left_idx], depth + 1)
        right = self._build(X[best_right_idx], y[best_right_idx], depth + 1)
        return Node(
            feature=best_feature, threshold=best_threshold, left=left, right=right
        )

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.root = self._build(X, y)
        return self

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_one(x, self.root) for x in X])

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
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def save_checkpoint(self, path=None):
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"checkpoints/decision_tree_{ts}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Model saved at: {path}")

    @staticmethod
    def load_checkpoint(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from: {path}")
        return model
