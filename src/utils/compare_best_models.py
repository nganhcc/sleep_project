import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

results = [
    np.array(
        [0.886, 0.8493, 0.8588, 0.8094, 0.8178, 0.8462, 0.8124, 0.7464, 0.8505, 0.8549]
    ),
    np.array(
        [0.886, 0.8493, 0.8588, 0.8094, 0.8178, 0.8462, 0.8124, 0.7464, 0.8505, 0.8549]
    ),
    np.array(
        [0.8807, 0.8753, 0.8617, 0.8093, 0.8222, 0.816, 0.862, 0.8358, 0.854, 0.8591]
    ),
    np.array(
        [0.8577, 0.8393, 0.8421, 0.8456, 0.7879, 0.8637, 0.8592, 0.8248, 0.857, 0.8602]
    ),
    np.array(
        [0.8966, 0.9043, 0.9014, 0.8105, 0.8833, 0.8856, 0.8644, 0.8241, 0.8729, 0.8382]
    ),
    np.array(
        [0.8973, 0.925, 0.9116, 0.8467, 0.8986, 0.9021, 0.8739, 0.8248, 0.9091, 0.8498]
    ),
    np.array(
        [0.8662, 0.8979, 0.8901, 0.84, 0.8465, 0.8796, 0.8508, 0.8053, 0.8646, 0.8067]
    ),
]
# Labels for the x-axis (your model names)
names = ["dt", "dt_sk", "knn", "knn_sk", "rf", "rf_knn", "svm_sk"]

# 2. Create the Box Plot
fig, ax = plt.subplots(figsize=(10, 6))

# The key function: plt.boxplot
ax.boxplot(
    results,
    labels=names,
    showmeans=True,  # Displays the mean as a green triangle (like your plot)
    meanline=False,  # The mean is a point, not a line
    patch_artist=False,  # Standard box plot appearance
)

# 3. Add labels and title for clarity
ax.set_title("Model Comparison using Cross-Validation Scores")
ax.set_ylabel("Mean F1 Score")
ax.set_ylim(0.70, 1.00)  # Set limits to match your reference plot's y-axis range
ax.grid(False)  # Optional: Remove grid lines if you prefer a clean look
plt.savefig("logs/plots/compare_best_models.png", bbox_inches="tight")

# 4. Show the plot
plt.show()
