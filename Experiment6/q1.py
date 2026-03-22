import numpy as np
import matplotlib.pyplot as plt

# Data points
data_points = np.array([
    [2, 3],  # a
    [3, 4],  # b
    [6, 6],  # c
    [7, 7]   # d
])

# Number of clusters
k = 2     

# Step 1: Initialize centroids (e.g., randomly pick 2 data points as initial centroids)
# For reproducibility and clear demonstration, let's pick point 'a' and point 'd' as initial centroids.
centroids = np.array([data_points[0], data_points[3]]) # Initial centroids: [2,3] and [7,7]

print("K-Means Clustering Iterations:")
print("-" * 40)
print(f"Initial Centroids: C1={centroids[0]}, C2={centroids[1]}\n")

max_iterations = 10 # Set a maximum number of iterations to prevent infinite loops
for iteration in range(max_iterations):
    print(f"Iteration {iteration + 1}:")

    # Step 2: Assignment Step - Assign each data point to the closest centroid
    distances = np.array([np.linalg.norm(data_points - centroid, axis=1) for centroid in centroids])
    cluster_assignments = np.argmin(distances, axis=0)

    print(f"  Data Point Assignments: {cluster_assignments} (0 for C1, 1 for C2)")

    # Step 3: Update Step - Recalculate new centroids
    new_centroids = np.array([data_points[cluster_assignments == i].mean(axis=0) if np.any(cluster_assignments == i) else centroids[i] for i in range(k)])

    print(f"  New Centroids: C1={new_centroids[0]}, C2={new_centroids[1]}")

    # Check for convergence
    if np.allclose(centroids, new_centroids):
        print("  Centroids converged. Stopping iterations.")
        break
    centroids = new_centroids
    print("\n" + "-" * 40 + "\n")

print("\nFinal Clustering Result:")
print(f"Final Centroids: C1={centroids[0]}, C2={centroids[1]}")
print(f"Final Cluster Assignments: {cluster_assignments}")

# Visualization
plt.figure(figsize=(8, 6))
colors = ['r', 'g']
markers = ['o', 's']
point_names = ['a', 'b', 'c', 'd']

for i in range(k):
    cluster_points_indices = np.where(cluster_assignments == i)[0]
    cluster_points = data_points[cluster_points_indices]
    for j, idx in enumerate(cluster_points_indices):
        plt.scatter(data_points[idx, 0], data_points[idx, 1], color=colors[i], marker=markers[i], s=100)
        plt.text(data_points[idx, 0] + 0.1, data_points[idx, 1] + 0.1, point_names[idx], fontsize=12)

plt.scatter(centroids[:, 0], centroids[:, 1], color='blue', marker='X', s=200, label='Centroids', edgecolor='black')

plt.title('K-Means Clustering of Data Points')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.grid(True)
plt.legend(['Cluster 1 Points', 'Cluster 2 Points', 'Centroids'], loc='upper left')
plt.show()
'''
K-Means Clustering Analysis
Aim:
Question: How can we group the given set of 2D data points into two distinct clusters using the K-Means algorithm?
Objective: To demonstrate the K-Means clustering algorithm step-by-step, including centroid initialization, iterative assignment, and update phases, and visualize the final clusters and their centroids.
Method Used:
We employed the K-Means clustering algorithm, an unsupervised machine learning technique, to partition the data points into k=2 clusters. The steps involved are:

Initialization: Two initial centroids (C1 and C2) were chosen from the data points ('a' and 'd') for reproducible demonstration.
Assignment Step: Each data point was assigned to the cluster whose centroid was closest to it, based on Euclidean distance.
Update Step: The centroids of each cluster were recalculated by taking the mean of all data points assigned to that cluster.
Convergence: Steps 2 and 3 were repeated until the centroids no longer changed significantly (i.e., they converged).
The process was implemented in Python using numpy for numerical operations and matplotlib for visualization.

Interpretation of Results:
Initial State: We started with C1 = [2, 3] (point 'a') and C2 = [7, 7] (point 'd') as initial centroids.
Iteration 1:
Data points 'a'([2,3]) and 'b'([3,4]) were assigned to Cluster 0 (C1) as they were closer to [2,3].
Data points 'c'([6,6]) and 'd'([7,7]) were assigned to Cluster 1 (C2) as they were closer to [7,7].
The new centroids were calculated: C1 = [2.5, 3.5] (mean of 'a' and 'b') and C2 = [6.5, 6.5] (mean of 'c' and 'd').
Iteration 2:
With the new centroids, the assignment of data points remained the same: 'a' and 'b' to Cluster 0; 'c' and 'd' to Cluster 1.
The recalculation of centroids resulted in the same values: C1 = [2.5, 3.5] and C2 = [6.5, 6.5].
Convergence: Since the centroids did not change between Iteration 1 and Iteration 2, the algorithm converged. The K-Means process successfully identified two distinct groups of data points.
Visualization:
The plot below illustrates the final clustering result. Data points assigned to Cluster 0 are shown in red, and points assigned to Cluster 1 are shown in green. The final positions of the centroids are marked with large blue 'X's. The plot clearly shows the separation of the data points into two distinct groups, with centroids representing the center of each cluster.

Conclusion:
Through the K-Means clustering algorithm, the four given data points a(2,3), b(3,4), c(6,6), and d(7,7) were effectively partitioned into two clusters. Cluster 0 contains points 'a' and 'b', centered around [2.5, 3.5]. Cluster 1 contains points 'c' and 'd', centered around [6.5, 6.5]. The algorithm converged quickly in two iterations, indicating a clear separation of the data into these two groups. This demonstrates the K-Means algorithm's ability to identify underlying structures in data.


[ ]
'''