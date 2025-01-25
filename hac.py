import numpy as np
import scipy.spatial.distance as dist

class HAC:
    """
    A simplified HAC implementation (single or complete linkage) that uses
    a precomputed distance matrix and updates it at each iteration.
    """

    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def fit(self, X):
        """
        Fit the hierarchical clustering on X and return the final cluster merges.
        """
        n_samples = X.shape[0]
        # Initially, each data point is its own cluster
        clusters = {i: [i] for i in range(n_samples)}  # cluster_id -> list of point indices

        # Precompute the full distance matrix (square form)
        # This is an O(N^2) preprocessing, but done only once.
        dist_matrix = dist.squareform(dist.pdist(X, metric='euclidean'))

        # Keep track of "active" clusters; each point starts in its own cluster (ID = i)
        # We'll say cluster i is "active" if it hasn't been merged into another cluster
        active = set(clusters.keys())  # set of cluster IDs still in play

        # We will do exactly (n_samples - n_clusters) merges
        for _ in range(n_samples - self.n_clusters):
            # 1. Find the two closest active clusters
            c1, c2 = None, None
            min_dist = np.inf
            active_list = list(active)
            for i in range(len(active_list)):
                for j in range(i+1, len(active_list)):
                    cid_i = active_list[i]
                    cid_j = active_list[j]
                    # Distances between these two "clusters" is stored 
                    # as dist_matrix of some representative indices. 
                    # We can pick the first index in each cluster for reference
                    # if we properly keep dist_matrix updated.
                    d = dist_matrix[cid_i, cid_j]
                    if d < min_dist:
                        min_dist = d
                        c1, c2 = cid_i, cid_j

            # 2. Merge cluster c2 into cluster c1, remove c2 from active
            clusters[c1].extend(clusters[c2])
            del clusters[c2]
            active.remove(c2)

            # 3. Update the distance matrix for the newly formed cluster c1
            for other_cid in active:
                if other_cid == c1:
                    continue

                if self.linkage == 'single':
                    # single linkage distance = min(dist(c1, other), dist(c2, other))
                    new_dist = min(dist_matrix[c1, other_cid],
                                   dist_matrix[c2, other_cid])
                elif self.linkage == 'complete':
                    # complete linkage distance = max(dist(c1, other), dist(c2, other))
                    new_dist = max(dist_matrix[c1, other_cid],
                                   dist_matrix[c2, other_cid])
                else:
                    raise ValueError('Invalid linkage')

                # Symmetrically update distance matrix for c1
                dist_matrix[c1, other_cid] = new_dist
                dist_matrix[other_cid, c1] = new_dist

            # Optionally set the distances for c2 to some large number (inf)
            # so we never pick it again
            dist_matrix[c2, :] = np.inf
            dist_matrix[:, c2] = np.inf

        # Build final labels
        # Now we have exactly self.n_clusters "active" clusters
        self.labels_ = np.zeros(n_samples, dtype=int)
        for label_idx, c_id in enumerate(clusters):
            for sample_idx in clusters[c_id]:
                self.labels_[sample_idx] = label_idx

        return self

    def predict(self, X):
        # This is a stub. Typically hierarchical clustering doesn't do 'predict' 
        # without re-fitting. We'll just return self.labels_.
        return self.labels_
