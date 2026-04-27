# ============================================================
#  Water Quality Dataset Analysis Using KNN
#  Assignment 1 - Horus University
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ============================================================
# 1. LOAD DATASET
# ============================================================
# Option A: Load from CSV if you have the file
df = pd.read_csv("water_potability.csv")

# Option B: Synthetic water-quality data (used when no file provided)
#np.random.seed(42)
#n_samples = 3276

#data = {
   # "ph":                  np.random.normal(7.0,  1.5,  n_samples),
    #"Hardness":            np.random.normal(196,  32,   n_samples),
    #"Solids":              np.random.normal(22014, 8768, n_samples),
    #"Chloramines":         np.random.normal(7.1,  1.6,  n_samples),
    #"Sulfate":             np.random.normal(333,   41,  n_samples),
    #"Conductivity":        np.random.normal(426,   80,  n_samples),
    #"Organic_carbon":      np.random.normal(14.3,  3.3, n_samples),
    #"Trihalomethanes":     np.random.normal(66.4,  16,  n_samples),
    #"Turbidity":           np.random.normal(3.97,  0.78,n_samples),
    #"Potability":          np.random.randint(0, 2,      n_samples),
#}
#
# 3df = pd.DataFrame(data)

print("=" * 55)
print("         WATER QUALITY - KNN ANALYSIS")
print("=" * 55)

print("\n[1] Dataset Overview")
print(f"    Shape : {df.shape}")
print(f"    Target distribution:\n{df['Potability'].value_counts().to_string()}")

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================
print("\n[2] Data Preprocessing")

# 2a. Check & fill missing values
missing = df.isnull().sum()
if missing.any():
    print(f"    Missing values found – filling with column mean")
    df.fillna(df.mean(), inplace=True)
else:
    print("    No missing values found.")

# 2b. Split features / target
X = df.drop("Potability", axis=1)
y = df["Potability"]

# 2c. Train / test split  (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"    Train samples : {len(X_train)}")
print(f"    Test  samples : {len(X_test)}")

# 2d. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("    Features scaled with StandardScaler.")

# ============================================================
# 3. MODEL TRAINING
# ============================================================
print("\n[3] Model Training")

# Choose best k with a quick loop
k_range = range(1, 21)
cv_scores = []
for k in k_range:
    knn_tmp = KNeighborsClassifier(n_neighbors=k)
    knn_tmp.fit(X_train_scaled, y_train)
    cv_scores.append(knn_tmp.score(X_test_scaled, y_test))

best_k = k_range[np.argmax(cv_scores)]
print(f"    Best k found   : {best_k}")

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)
print("    Model trained successfully.")

# ============================================================
# 4. MODEL EVALUATION
# ============================================================
print("\n[4] Model Evaluation")

y_pred = knn.predict(X_test_scaled)

acc  = accuracy_score (y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score   (y_test, y_pred, zero_division=0)
f1   = f1_score       (y_test, y_pred, zero_division=0)

print(f"\n    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"    Precision : {prec:.4f}")
print(f"    Recall    : {rec:.4f}")
print(f"    F1-Score  : {f1:.4f}")

print("\n    Full Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["Not Potable", "Potable"]))

# ============================================================
# 5. VISUALIZATIONS
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Water Quality – KNN Analysis", fontsize=15, fontweight="bold")

# -- Plot 1: Confusion Matrix --
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Not Potable", "Potable"],
            yticklabels=["Not Potable", "Potable"])
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

# -- Plot 2: Accuracy vs k --
axes[1].plot(list(k_range), cv_scores, marker="o", color="steelblue")
axes[1].axvline(x=best_k, color="red", linestyle="--",
                label=f"Best k={best_k}")
axes[1].set_title("Accuracy vs Number of Neighbors (k)")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# -- Plot 3: Metrics Bar Chart --
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
values  = [acc, prec, rec, f1]
colors  = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
bars = axes[2].bar(metrics, values, color=colors, edgecolor="black", width=0.5)
axes[2].set_ylim(0, 1.1)
axes[2].set_title("Evaluation Metrics")
axes[2].set_ylabel("Score")
for bar, val in zip(bars, values):
    axes[2].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("water_quality_knn_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n    Plot saved as 'water_quality_knn_results.png'")
print("\n" + "=" * 55)
print("  Analysis complete!")
print("=" * 55)