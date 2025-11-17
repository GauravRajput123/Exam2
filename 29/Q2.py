import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------
# Load & preprocess dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\29\employees.csv")  # Must include "income" column
df = df.dropna()   # Remove missing values

X = df[['income']]  # Only income used for clustering

# -----------------------------
# Elbow Method to find best K
# -----------------------------
inertia_list = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia_list.append(km.inertia_)

plt.plot(K_range, inertia_list, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

# -----------------------------
# Silhouette Score to confirm k
# -----------------------------
silhouette_vals = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_vals.append(score)
    print(f"K={k} → Silhouette Score = {round(score, 4)}")

plt.plot(K_range, silhouette_vals, marker='o')
plt.title("Silhouette Scores")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.show()

# -----------------------------
# Choose optimal K (example: 3)
# -----------------------------
k_optimal = silhouette_vals.index(max(silhouette_vals)) + 2
print("Best K =", k_optimal)

# Train final model
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# -----------------------------
# Plot Clusters
# -----------------------------
plt.scatter(df['income'], [0]*len(df), c=df['Cluster'], cmap='viridis')
plt.xlabel("Employee Income")
plt.title("K-Means Clustering on Income")
plt.show()

print(df.head())
