import numpy as np
import scipy as sp


class HAC():
    """
    Implementation of Hierarchical Agglomerative Clustering (HAC).
    """
    def __init__(self, n_clusters, linkage='single', metric='euclidean'):
        """
        Args:
            n_clusters (int): target number of clusters
            linkage (str, optional): The linkage criterion for merging clusters. Defaults to 'single'.
                - 'single': Minimum distance between points in clusters.
                - 'average': Average linkage.
                - 'complete': Maximum distance between points in clusters.
                - 'ward': Minimizes the "increase" in within-cluster SSE. 
                Calculated according to the Lance–Williams formula

         Raises:
            ValueError: If an invalid linkage criterion is provided.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.labels_ = None

        if self.linkage == 'ward' and self.metric != 'euclidean':
            raise ValueError("Ward linkage can be used only with euclidean")

    def fit(self, X):
        """
        Performs HAC for the given data.
        Merges clusters iteratively based on the chosen linkage criterion until
        only n_clusters remain. Each datapoint is assigned a cluster label, and is
        stored as '_labels' atrtibute.

        To speed up the computation, it stores the cluster distances as a distance
        matrix, which is updated after forming a new cluster.

        Args:
            X (np.ndarray of shape (n_samples, n_features)): data to cluster

        Raises:
            ValueError: if an invalid linkage criterion is provided

        Returns:
            HAC: class instance
        """
        n_clusters = X.shape[0]
        clusters = {i: [i] for i in range(n_clusters)}
        remaining_clusters = list(clusters.keys())
        distance_matrix = self._distance_matrix(X)
        c_sizes = {cid: 1 for cid in clusters.keys()}

        while len(remaining_clusters) > self.n_clusters:
            min_distance = np.inf
            clusters_to_merge = (0, 0)
            
            for i in range(0, len(remaining_clusters), 1):
                for j in range(i+1, len(remaining_clusters), 1):
                    cid_i = remaining_clusters[i]
                    cid_j = remaining_clusters[j]
                    dist_ij = distance_matrix[cid_i, cid_j]
                    if dist_ij < min_distance:
                        min_distance = dist_ij
                        clusters_to_merge = (cid_i, cid_j)

            cluster1, cluster2 = clusters_to_merge
            clusters[cluster1].extend(clusters[cluster2])

            for cluster in remaining_clusters:
                if cluster == cluster1:
                    continue
                
                if self.linkage == 'single':
                    new_dist = min(distance_matrix[cluster1, cluster],
                                   distance_matrix[cluster2, cluster])
                elif self.linkage == 'complete':
                    new_dist = max(distance_matrix[cluster1, cluster],
                                   distance_matrix[cluster2, cluster])
                elif self.linkage == 'ward':
                    nA = c_sizes[cluster1]
                    nB = c_sizes[cluster2]
                    nC = c_sizes[cluster]
                    dAC = distance_matrix[cluster1, cluster]
                    dBC = distance_matrix[cluster2, cluster]
                    dAB = distance_matrix[cluster1, cluster2]
                    new_dist = ((nA + nC) * dAC + 
                                (nB + nC) * dBC - nC * dAB) / (nA + nB + nC)
                elif self.linkage == 'average':
                    new_dist = (c_sizes[cluster1] * distance_matrix[cluster1, cluster] +
                        c_sizes[cluster2] * distance_matrix[cluster2, cluster]) / (c_sizes[cluster1] + c_sizes[cluster2])
                else:
                    raise ValueError("invalid linkage type passed as an argument")
                
                distance_matrix[cluster1, cluster] = new_dist
                distance_matrix[cluster, cluster1] = new_dist

            c_sizes[cluster1] += c_sizes[cluster2]

            del c_sizes[cluster2]
            del clusters[cluster2]
            remaining_clusters.remove(cluster2)

            self.labels_ = np.zeros(n_clusters, dtype=int)
            label_id = 0
            for cid in clusters:
                for idx in clusters[cid]:
                    self.labels_[idx] = label_id
                label_id += 1

        return self


    def _distance_matrix(self, X):
        """
        Returns a distance matrix

        Args:
            X (np.ndarray of shape (n_samples, n_features)): datapoints

        Returns:
            np: _description_
        """
        n_init_clusters = X.shape[0]
        distance_matrix = np.zeros((n_init_clusters, n_init_clusters))
        for i in range(n_init_clusters):
            for j in range(n_init_clusters):
                if self.linkage == 'ward':
                    distance_matrix[i][j] = sp.spatial.distance.euclidean(X[i], X[j])**2
                else:
                    if self.metric == 'euclidean':
                        distance_matrix[i][j] = sp.spatial.distance.euclidean(X[i], X[j])
                    elif self.metric == 'manhattan':
                        distance_matrix[i][j] = sp.spatial.distance.cityblock(X[i], X[j])
                    elif self.metric == 'cosine':
                        distance_matrix[i][j] = sp.spatial.distance.cosine(X[i], X[j])
                    else:
                        raise ValueError("invalide distance metric")

        return distance_matrix