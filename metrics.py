import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Coefficient of all samples.

    The Silhouette Coefficient for a sample is:
        s(i) = (b(i) - a(i)) / max{ a(i), b(i) }

    where
    - a(i) is the mean intra-cluster distance (average distance within the same cluster).
    - b(i) is the mean nearest-cluster distance (the smallest average distance to any other cluster).

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_samples, n_features).
    labels : np.ndarray
        1D array of cluster labels (same length as X). Label = -1 for noise.

    Returns
    -------
    float
        Mean Silhouette Score. The best value is 1, and the worst value is -1.
        Values near 0 indicate overlapping clusters.
        Returns 0 if the metric cannot be computed (e.g., all points in one cluster).
    """
    # Get the set of unique clusters (excluding noise = -1)
    unique_labels = [c for c in set(labels) if c != -1]
    n_clusters = len(unique_labels)
    # Silhouette is not defined for 1 or 0 clusters
    if n_clusters <= 1:
        return 0.0

    # Precompute distance matrix (O(n^2) approach)
    n_samples = len(X)
    dist_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # For each cluster, gather indices
    cluster_indices = {}
    for c in unique_labels:
        cluster_indices[c] = np.where(labels == c)[0]

    # Compute the silhouette for each sample that isn't noise
    sil_values = []
    for i in range(n_samples):
        c_i = labels[i]
        if c_i == -1:
            # Noise point, skip or treat as its own cluster
            continue

        # All points in the same cluster (excluding itself)
        same_cluster_idx = cluster_indices[c_i]
        if len(same_cluster_idx) == 1:
            # This means the point i is alone in its cluster => silhouette = 0 by convention
            sil_values.append(0)
            continue

        # --- a(i): mean distance to points in the same cluster (excluding itself) ---
        a_i = np.mean([dist_matrix[i, x] for x in same_cluster_idx if x != i])

        # --- b(i): smallest mean distance to any other cluster ---
        b_i = np.inf
        for c in unique_labels:
            if c == c_i:
                continue
            other_cluster_idx = cluster_indices[c]
            dist_to_cluster = np.mean([dist_matrix[i, x] for x in other_cluster_idx])
            if dist_to_cluster < b_i:
                b_i = dist_to_cluster

        # Silhouette for this point
        s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) != 0 else 0
        sil_values.append(s_i)

    # Mean of all silhouette values
    if len(sil_values) == 0:
        return 0.0
    return np.mean(sil_values)


def davies_bouldin_score(X, labels):
    """
    Compute the Davies-Bouldin Index (DBI) for clustering results.

    The lower the DBI, the better the clustering.

    For each cluster i, define:
        S_i = average distance of members of cluster i to its centroid.
    For each pair of clusters (i, j):
        R_{i,j} = (S_i + S_j) / d(C_i, C_j),
    where C_i is the centroid of cluster i, and d(C_i, C_j) is the distance
    between the two centroids.

    The DBI is then the average over i of the maximum R_{i,j}.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_samples, n_features).
    labels : np.ndarray
        1D array of cluster labels (same length as X). Label = -1 for noise.

    Returns
    -------
    float
        The Davies-Bouldin score. The lower the score, the better the clusters.
        Returns 0 if it cannot be computed (e.g., fewer than 2 clusters).
    """
    # Collect unique clusters (excluding noise)
    unique_labels = [c for c in set(labels) if c != -1]
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        return 0.0  # DBI is undefined or trivial with fewer than 2 clusters

    # For each cluster, compute centroid and average distance (S_i)
    cluster_centroids = {}
    cluster_scatters = {}

    for c in unique_labels:
        members = np.where(labels == c)[0]
        cluster_points = X[members]
        centroid = np.mean(cluster_points, axis=0)
        cluster_centroids[c] = centroid
        # Compute average distance of members to centroid
        scatter = np.mean([np.sqrt(np.sum((p - centroid) ** 2)) for p in cluster_points])
        cluster_scatters[c] = scatter

    # Compute DBI
    dbi_list = []
    for i in unique_labels:
        max_ratio = float('-inf')
        for j in unique_labels:
            if i == j:
                continue
            dist_centroids = np.sqrt(np.sum((cluster_centroids[i] - cluster_centroids[j]) ** 2))
            if dist_centroids == 0:
                # Avoid division by zero -> clusters overlap or identical
                ratio = float('inf')
            else:
                ratio = (cluster_scatters[i] + cluster_scatters[j]) / dist_centroids
            if ratio > max_ratio:
                max_ratio = ratio
        dbi_list.append(max_ratio)

    # Davies-Bouldin is the average of these max_ratios
    return np.mean(dbi_list)


if __name__ == "__main__":

    # Dummy data
    np.random.seed(42)
    cluster1 = np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2))
    cluster2 = np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2))
    X = np.vstack((cluster1, cluster2))

    # Fake labels (e.g., suppose we had 2 clusters)
    labels = np.array([0]*50 + [1]*50)

    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)

    print("Silhouette Score:", sil)
    print("Davies-Bouldin Index:", dbi)
