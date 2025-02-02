import numpy as np
import scipy as sp
import pandas as pd


class DBSCAN:
    """
    eps : float, optional (default=0.5)
        The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples : int, optional (default=5)
        The number of samples in a neighborhood for a point to be considered as a core point.
        This includes the point itself.
    """
    
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.n_clusters_ = 0

    def fit(self, X):
        """
        Parameters
        ----------
        X : np.ndarray
            2D numpy array of shape (n_samples, n_features) containing the data.
        
        Returns
        -------
        self : DBSCAN
            Fitted DBSCAN instance (with labels_ and n_clusters_ attributes).
        """
        # Initialize all labels as 0 (unvisited)
        labels = [0] * len(X)
        cluster_id = 0

        # Iterate over each point
        for i in range(len(X)):
            if labels[i] != 0:
                # Already visited
                continue

            # Find neighbors of point i
            neighbors = self._region_query(X, i)

            # Check if it is a core point (check if the number of neighbors is greater than min_samples)
            if len(neighbors) < self.min_samples:
                # Label as noise
                labels[i] = -1
            else:
                # We have a new cluster
                cluster_id += 1
                labels[i] = cluster_id
                # Expand the cluster
                self._expand_cluster(X, labels, i, neighbors, cluster_id)

        self.labels_ = np.array(labels)
        # The number of clusters is the max label (excluding noise = -1)
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)

        return self

    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id):
        """
        Expand the cluster to include density-reachable points.
        
        Parameters
        ----------
        X : np.ndarray
            Dataset of shape (n_samples, n_features).
        labels : list or np.ndarray
            Current labels of the data points.
        point_idx : int
            The index of the current core point from which to expand the cluster.
        neighbors : list
            Indices of the neighbor points for point_idx.
        cluster_id : int
            The current cluster ID we are assigning.
        """

        i = 0

        # Iterate over the immediate neighbors
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            # If the neighbor hasn't been visited yet
            if labels[neighbor_idx] == 0:
                # Assign the cluster ID to the neighbor
                labels[neighbor_idx] = cluster_id
                # Get neighbors of this neighbor
                neighbor_neighbors = self._region_query(X, neighbor_idx)
                # If it's a core point, append its neighbors
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors += neighbor_neighbors
            
            # If the neighbor was labeled as noise, update its label
            elif labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id

            i += 1


    def _region_query(self, X, point_idx):
        """
        Find all points within distance `eps` of the point at index `point_idx`.
        
        Parameters
        ----------
        X : np.ndarray
            The dataset of shape (n_samples, n_features).
        point_idx : int
            Index of the point to query around.
        
        Returns
        -------
        neighbors : list of int
            Indices of all the points within eps (inclusive).
        """
        neighbors = []
        for i in range(len(X)):
            # Euclidean distance
            dist = np.sqrt(np.sum((X[point_idx] - X[i]) ** 2))
            if dist <= self.eps:
                neighbors.append(i)
        return neighbors


class HAC():
    def __init__(self, n_clusters, linkage='single', metric='euclidean'):
        """
        Parameters:
            n_clusters (int): target number of clusters
            linkage (str, optional): The linkage criterion for merging clusters. Defaults to 'single'.
                - 'single': Minimum distance between points in clusters.
                - 'complete': Maximum distance between points in clusters.
                - 'ward': Minimizes the "increase" in within-cluster SSE. 
                Calculated according to the Lance–Williams formula
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.labels_ = None

        if self.linkage == 'ward' and self.metric != 'euclidean':
            raise ValueError("Ward linkage can be used only with euclidean")

    def fit(self, X):
        """
        Merges clusters iteratively based on the chosen linkage criterion until
        only n_clusters remain. Each datapoint is assigned a cluster label, and is
        stored as '_labels' atrtibute.

        To speed up the computation, it stores the cluster distances as a distance
        matrix, which is updated after forming a new cluster.

        Parameters:
            X (np.ndarray of shape (n_samples, n_features)): data to cluster
        """

        # Initialize number of clusters
        n_clusters = X.shape[0]
        # Initialize clusters
        clusters = {i: [i] for i in range(n_clusters)}
        # Initialize remaining clusters
        remaining_clusters = list(clusters.keys())
        # Initialize distance matrix
        distance_matrix = self._distance_matrix(X)
        # Initialize cluster sizes
        c_sizes = {cid: 1 for cid in clusters.keys()}

        # Iterate till the target number of clusters is reached
        while len(remaining_clusters) > self.n_clusters:

            # Initialize minimum distance as the largest possible distance
            min_distance = np.inf

            # Initialize clusters to merge
            clusters_to_merge = (0, 0)
            
            # Iterate over the remaining clusters to find closest clusters
            for i in range(0, len(remaining_clusters), 1):
                for j in range(i+1, len(remaining_clusters), 1):
                    cid_i = remaining_clusters[i]
                    cid_j = remaining_clusters[j]
                    dist_ij = distance_matrix[cid_i, cid_j]
                    if dist_ij < min_distance:
                        min_distance = dist_ij
                        clusters_to_merge = (cid_i, cid_j)

            # Merge the two closest clusters
            cluster1, cluster2 = clusters_to_merge
            clusters[cluster1].extend(clusters[cluster2])

            # Update the distance matrix for the new cluster
            for cluster in remaining_clusters:
                if cluster == cluster1:
                    continue
                
                # If the linkage is single, find the minimum distance between the new cluster and the remaining clusters
                if self.linkage == 'single':
                    new_dist = min(distance_matrix[cluster1, cluster],
                                   distance_matrix[cluster2, cluster])

                # If the linkage is complete, find the maximum distance between the new cluster and the remaining clusters
                elif self.linkage == 'complete':
                    new_dist = max(distance_matrix[cluster1, cluster],
                                   distance_matrix[cluster2, cluster])

                # If the linkage is ward, find the weighted average distance between the new cluster and the remaining clusters
                elif self.linkage == 'ward':
                    nA = c_sizes[cluster1]
                    nB = c_sizes[cluster2]
                    nC = c_sizes[cluster]
                    dAC = distance_matrix[cluster1, cluster]
                    dBC = distance_matrix[cluster2, cluster]
                    dAB = distance_matrix[cluster1, cluster2]
                    new_dist = ((nA + nC) * dAC + 
                                (nB + nC) * dBC - nC * dAB) / (nA + nB + nC)
                else:
                    raise ValueError("invalid linkage type passed as an argument")
                
                # Update the distance matrix for the new cluster
                distance_matrix[cluster1, cluster] = new_dist
                distance_matrix[cluster, cluster1] = new_dist

            # Update the cluster sizes
            c_sizes[cluster1] += c_sizes[cluster2]

            # Remove the absorbed cluster
            del c_sizes[cluster2]
            del clusters[cluster2]
            remaining_clusters.remove(cluster2)

            # Update the labels
            self.labels_ = np.zeros(n_clusters, dtype=int)
            label_id = 0
            for cid in clusters:
                for idx in clusters[cid]:
                    self.labels_[idx] = label_id
                label_id += 1

        return self


    def _distance_matrix(self, X):
        """
        Parameters:
            X (np.ndarray of shape (n_samples, n_features)): datapoints
        """
        # Initialize the distance matrix
        n_init_clusters = X.shape[0]
        distance_matrix = np.zeros((n_init_clusters, n_init_clusters))

        # Iterate over the initial clusters
        for i in range(n_init_clusters):
            for j in range(n_init_clusters):
                # If the linkage is ward, calculate the squared euclidean distance
                if self.linkage == 'ward':
                    distance_matrix[i][j] = sp.spatial.distance.euclidean(X[i], X[j])**2
                else:
                    # If the linkage is not ward, calculate the distance based on the metric
                    if self.metric == 'euclidean':
                        distance_matrix[i][j] = sp.spatial.distance.euclidean(X[i], X[j])
                    elif self.metric == 'manhattan':
                        distance_matrix[i][j] = sp.spatial.distance.cityblock(X[i], X[j])
                    elif self.metric == 'cosine':
                        distance_matrix[i][j] = sp.spatial.distance.cosine(X[i], X[j])
                    else:
                        raise ValueError("invalide distance metric")

        return distance_matrix


class KMeans:
    """
    Parameters
    ----------
    n_clusters : int, optional (default=3)
        The number of clusters to form.
    max_iter : int, optional (default=300)
        The maximum number of iterations for the algorithm.
    tol : float, optional (default=1e-4)
        The tolerance for convergence. If the change in the centroids is less than this value, the algorithm stops.
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids_ = None
        self.labels_ = None

    def fit(self, X):
        """
        Parameters
        ----------
        X : np.ndarray
            2D numpy array of shape (n_samples, n_features) containing the data.

        Returns
        -------
        self : KMeans
            Fitted KMeans instance (with centroids_ and labels_ attributes).
        """
        # Randomly initialize centroids by selecting random data points
        random_idx = np.random.choice(len(X), size=self.n_clusters, replace=False)
        self.centroids_ = X[random_idx]

        for _ in range(self.max_iter):
            # Assign each data point to the nearest centroid
            self.labels_ = self._assign_labels(X)

            # Calculate new centroids
            new_centroids = self._calculate_centroids(X)

            # Check for convergence (if centroids do not change)
            centroid_shift = np.linalg.norm(new_centroids - self.centroids_)
            if centroid_shift < self.tol:
                break

            self.centroids_ = new_centroids

        return self

    def _assign_labels(self, X):
        """
        Parameters
        ----------
        X : np.ndarray
            2D numpy array of shape (n_samples, n_features) containing the data.

        Returns
        -------
        labels : np.ndarray
            1D array of labels indicating the assigned cluster for each point.
        """
        # Assign each point to the nearest centroid
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids_, axis=2), axis=1)
        return labels

    def _calculate_centroids(self, X):
        """
        Parameters
        ----------
        X : np.ndarray
            2D numpy array of shape (n_samples, n_features) containing the data.

        Returns
        -------
        centroids : np.ndarray
            2D numpy array of shape (n_clusters, n_features) containing the new centroids.
        """
        # Calculate the new centroids by taking the mean of the points in each cluster
        centroids = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])
        return centroids


class PCA:
    def __init__(self, n_components):
        """
        Parameters
        ----------
        n_components : int
            The number of principal components to retain.
        """
        self.n_components_ = n_components
        self.mean_ = None
        self.std_ = None
        self.pcs_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.loadings_ = None


    def fit(self, X):
        """
        Parameters
        ----------
        X : np.ndarray
            2D numpy array of shape (n_samples, n_features) containing the data.

        Returns
        -------
        self : PCA
            Fitted PCA instance (with centroids_ and labels_ attributes).
        """
        # Calculate the mean of the data
        self.mean_ = X.mean(axis=0)
        # Calculate the covariance matrix
        cov_matrix = np.cov(X - self.mean_, rowvar=False)
        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = self._eigen_decomp(cov_matrix)

        # Store the principal components
        self.pcs_ = eigenvectors[:, :self.n_components_]
        # Store the explained variance
        self.explained_variance_ = eigenvalues[:self.n_components_]
        # Compute the explained variance ratio
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvalues)
        # Compute the loadings
        self.loadings_ = eigenvectors[:, :self.n_components_] * np.sqrt(eigenvalues[:self.n_components_])

        return self


    def transform(self, X):
        """
        Projects data onto principal components obtained when calling the 
        'fit' method, resulting in dimensionality rediced representation
        of the input data

        Args:
            X (np.ndarray of shape (n_samples, n_features)): data to be transformed.

        Returns:
            np.ndarray of shape (n_samples, n_components): data projected onto n principal components
        """
        X = X - self.mean_
        return np.dot(X, self.pcs_)


    def _eigen_decomp(self, cov_matrix):
        """
        Eigenvalue decomposition of the covariance matrix.
        Eigenvalues and eigenvectors are sorted in descending order of
        respective eigenvalues.

        Args:
            cov_matrix (np.ndarray of shape (n_features, n_features)): covariance matrix

        Returns:
            tuple:
                - eigenvalues (np.ndarray of shape (n_features,)): 
                eigenvalues sorted in descending order
                - eigenvectors (np.ndarray of shape (n_features, n_features)): 
                eigenvectors sorted to match the order of eigenvalues
        """
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        return eigenvalues, eigenvectors

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
