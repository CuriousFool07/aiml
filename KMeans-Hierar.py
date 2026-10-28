
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

# 1. Load dataset
df = pd.read_csv("Mall_Customers.csv")
print(df.head())

# 2. Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# 3. Elbow method
inertia = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertia.append(model.inertia_)

plt.plot(range(1, 11), inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# 4. K-Means with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(X)

plt.scatter(
    X.iloc[:, 0], X.iloc[:, 1],
    c=df["KMeans_Cluster"], cmap="viridis"
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Clustering")
plt.show()

# 5. Hierarchical clustering dendrogram
linked = linkage(X, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

# 6. Agglomerative clustering with 5 clusters
hierarchical = AgglomerativeClustering(n_clusters=5)
df["Hierarchical_Cluster"] = hierarchical.fit_predict(X)

plt.scatter(
    X.iloc[:, 0], X.iloc[:, 1],
    c=df["Hierarchical_Cluster"], cmap="viridis"
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Hierarchical Clustering")
plt.show()

print(df[[
    "Annual Income (k$)", "Spending Score (1-100)",
    "KMeans_Cluster", "Hierarchical_Cluster"
]].head())
