import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Define the data
points_labels = ['a', 'b', 'c', 'd']
points_values = np.array([
    [1, 1],
    [2, 1],
    [4, 3],
    [5, 4]
])

print("Data Points and Values:")
for i, label in enumerate(points_labels):
    print(f"{label}: {points_values[i]}")

# Perform Agglomerative Clustering
# We'll use 'euclidean' distance and 'single' linkage for this example
# n_clusters can be set to None if we want to extract clusters from the dendrogram
agglomerative_model = AgglomerativeClustering(n_clusters=2, linkage='single')

# Fit the model to the data
agglomerative_model.fit(points_values)

# Get the cluster labels for each data point
cluster_labels = agglomerative_model.labels_

print("\n--- Agglomerative Clustering Results ---")
print(f"Cluster labels: {cluster_labels}")

# Print points in each cluster
num_clusters = len(np.unique(cluster_labels))
for i in range(num_clusters):
    cluster_points = [points_labels[j] for j, label in enumerate(cluster_labels) if label == i]
    cluster_values = [points_values[j] for j, label in enumerate(cluster_labels) if label == i]
    print(f"Cluster {i+1}: Points {cluster_points}, Values {cluster_values}")
