import numpy as np
from src.models.StackingEnsemble import StackingEnsemble

# Load your data
X_train = np.loadtxt("data/processed/train/X_train.csv", delimiter=",")
y_train = np.loadtxt("data/processed/train/y_train.csv", delimiter=",")
X_test = np.loadtxt("data/processed/test/X_test.csv", delimiter=",")
y_test = np.loadtxt("data/processed/test/y_test.csv", delimiter=",")

# Initialize and train
stacker = StackingEnsemble()
stacker.load_base_models()
stacker.build_model()
stacker.plot_validation_curve(X_train, y_train)
stacker.plot_learning_curve(X_train, y_train)
stacker.fit(X_train, y_train)
stacker.evaluate(X_test, y_test)
stacker.save()
