import numpy as np
from datetime import datetime
import pickle
from collections import Counter
from src.models.DecisionTree import DecisionTree


class RandomForest:
    def __init__(
        self,
        n_estimators=10,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=None,
        random_state=None,
        ratio=0.7,
    ):
        self.n_estimators = n_estimators
        self.ratio = ratio  # %of dataset we use to boostrap sample
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        if random_state is not None:
            np.random.seed(random_state)

    def _bootstrap_sample(self, X, y, ratio):
        n_samples = X.shape[0]
        n_bootstraps = int(ratio * n_samples)
        indices = np.random.choice(n_samples, size=n_bootstraps, replace=True)
        return X[indices], y[indices]

    def _sample_features(self, n_features):
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                size = int(np.sqrt(n_features))
            elif self.max_features == "log2":
                size = int(np.log2(n_features))
            else:
                size = n_features
        else:
            size = int(self.max_features)
        return np.random.choice(n_features, size=size, replace=False)

    def fit(self, X, y):
        n_features = X.shape[1]
        self.trees = []
        for _ in range(self.n_estimators):
            feature_idx = self._sample_features(n_features)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.feature_idx = feature_idx
            X_sample, y_sample = self._bootstrap_sample(X, y, self.ratio)
            _X, _y = X_sample[:, feature_idx], y_sample
            tree.fit(_X, _y)
            self.trees.append(tree)
        return self

    def predict(self, X):
        all_preds = np.array(
            [tree.predict(X[:, tree.feature_idx]) for tree in self.trees]
        )
        final_preds = []
        for col in all_preds.T:
            final_preds.append(Counter(col).most_common(1)[0][0])
        return np.array(final_preds)

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
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "max_features": self.max_features,
            "random_state": self.random_state,
            "ratio": self.ratio,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def save_checkpoint(self, path=None):
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"checkpoints/random_forest_{ts}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Model saved at: {path}")

    @staticmethod
    def load_checkpoint(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from: {path}")
        return model
