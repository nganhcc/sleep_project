import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, validation_curve, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import StackingClassifier


class StackingEnsemble:
    def __init__(
        self,
        model_dir="checkpoints",
        save_path="checkpoints/best_stacking_ensemble.pkl",
    ):
        self.model_dir = model_dir
        self.save_path = save_path
        self.meta_model = LogisticRegression(max_iter=1000)
        self.base_models = []
        self.model = None

    # ============================================================
    # LOAD BASE MODELS
    # ============================================================
    def load_base_models(self):
        self.base_models = []
        for name in os.listdir(self.model_dir):
            if name.startswith("best_") and name.endswith(".pkl"):
                model_path = os.path.join(self.model_dir, name)
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                self.base_models.append((name.replace(".pkl", ""), model))
        print(f"Loaded base models: {[n for n, _ in self.base_models]}")

    # ============================================================
    # BUILD STACKING MODEL
    # ============================================================
    def build_model(self):
        if not self.base_models:
            self.load_base_models()
        self.model = StackingClassifier(
            estimators=self.base_models,
            final_estimator=self.meta_model,
            cv=5,
            n_jobs=-1,
        )
        print("✅ Stacking ensemble built successfully.")

    # ============================================================
    # TRAIN
    # ============================================================
    def fit(self, X_train, y_train):
        if self.model is None:
            self.build_model()
        print("🚀 Training stacking ensemble...")
        self.model.fit(X_train, y_train)
        print("✅ Training completed.")

    # ============================================================
    # PREDICT
    # ============================================================
    def predict(self, X):
        return self.model.predict(X)

    # ============================================================
    # EVALUATE
    # ============================================================
    def evaluate(self, X_test, y_test, log_path="logs/stacking_results.csv"):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        print("\n=== STACKING MODEL REPORT ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))

        # Log results
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"{timestamp},{acc:.4f},{f1:.4f}\n")

        return acc, f1

    # ============================================================
    # SAVE & LOAD
    # ============================================================
    def save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"💾 Model saved to {self.save_path}")

    def load(self):
        with open(self.save_path, "rb") as f:
            self.model = pickle.load(f)
        print(f"📦 Loaded stacking model from {self.save_path}")

    # ============================================================
    # PLOT LEARNING CURVE
    # ============================================================
    def plot_learning_curve(
        self, X, y, save_path="logs/plots/learn_curve_stacking_ensemble.png"
    ):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        train_sizes, train_scores, valid_scores = learning_curve(
            self.model,
            X,
            y,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 6),
        )
        train_mean = np.mean(train_scores, axis=1)
        valid_mean = np.mean(valid_scores, axis=1)

        plt.figure()
        plt.plot(train_sizes, train_mean, "o-", label="Training score")
        plt.plot(train_sizes, valid_mean, "o-", label="Validation score")
        plt.title("Learning Curve - Stacking Ensemble")
        plt.xlabel("Training examples")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"📊 Learning curve saved to {save_path}")

    # ============================================================
    # PLOT VALIDATION CURVE
    # ============================================================
    def plot_validation_curve(
        self, X, y, save_path="logs/plots/val_curve_stacking_ensemble.png"
    ):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        param_range = np.logspace(-3, 2, 6)
        train_scores, valid_scores = validation_curve(
            self.model,
            X,
            y,
            param_name="final_estimator__C",
            param_range=param_range,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )
        train_mean = np.mean(train_scores, axis=1)
        valid_mean = np.mean(valid_scores, axis=1)

        plt.figure()
        plt.semilogx(param_range, train_mean, label="Training score")
        plt.semilogx(param_range, valid_mean, label="Validation score")
        plt.title("Validation Curve - Logistic Regression (meta model)")
        plt.xlabel("C (Regularization Strength)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"📈 Validation curve saved to {save_path}")
