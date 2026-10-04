# WEEK 3 - Unsupervised Learning & Model Evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

# 1. Load Amazon dataset
df = pd.read_csv("amazon.csv")

# 2. Convert columns to numeric
df["discounted_price"] = (
    df["discounted_price"].str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False).astype(float)
)

df["actual_price"] = (
    df["actual_price"].str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False).astype(float)
)

df["discount_percentage"] = (
    df["discount_percentage"].str.replace("%", "", regex=False)
    .astype(float)
)

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

df["rating_count"] = pd.to_numeric(
    df["rating_count"].str.replace(",", "", regex=False),
    errors="coerce"
)

# Remove missing values
df = df.dropna(subset=[
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating",
    "rating_count"
])

# 3. Select features
X = df[[
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating_count"
]]

# 4. Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-MEANS CLUSTERING

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

print("K-Means Silhouette Score:",
      silhouette_score(X_scaled, clusters))


# HIERARCHICAL CLUSTERING

hierarchical = AgglomerativeClustering(n_clusters=3)
h_clusters = hierarchical.fit_predict(X_scaled)

print("Hierarchical Silhouette Score:",
      silhouette_score(X_scaled, h_clusters))


# PCA DIMENSIONALITY REDUCTION

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clusters")
plt.show()


# MODEL EVALUATION
# Create classification target
# 1 = rating >= 4, 0 = rating < 4
y = (df["rating"] >= 4).astype(int)

model = LogisticRegression(max_iter=1000)

# Cross-validation
scores = cross_val_score(model, X_scaled, y, cv=5)

print("\nCross Validation Scores:", scores)
print("Mean CV Score:", scores.mean())


# Train model
model.fit(X_scaled, y)
pred = model.predict(X_scaled)
prob = model.predict_proba(X_scaled)[:, 1]

# Evaluation metrics
print("\nAccuracy :", accuracy_score(y, pred))
print("Precision:", precision_score(y, pred))
print("Recall   :", recall_score(y, pred))
print("F1 Score :", f1_score(y, pred))
print("ROC-AUC  :", roc_auc_score(y, prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y, pred))

# HYPERPARAMETER TUNING

params = {
    "C": [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    params,
    cv=5
)

grid.fit(X_scaled, y)

print("\nBest Parameter:", grid.best_params_)
print("Best CV Score:", grid.best_score_)
