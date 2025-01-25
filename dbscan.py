import numpy as np

class DBSCAN:
    """
    A simple DBSCAN clustering implementation from scratch.
    
    Parameters
    ----------
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
        Perform DBSCAN clustering on the given dataset.
        
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

            # Check if it is a core point
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
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            # If the neighbor hasn't been visited yet
            if labels[neighbor_idx] == 0:
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


# Example usage
if __name__ == "__main__":
    # Generate some synthetic 2D data
    np.random.seed(42)
    cluster1 = np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2))
    cluster2 = np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2))
    data = np.vstack((cluster1, cluster2))

    # Create and fit DBSCAN model
    eps = 0.8
    min_samples = 5
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(data)

    # Print labels and number of clusters found
    print("Labels assigned to each point:")
    print(model.labels_)
    print(f"Number of clusters found: {model.n_clusters_}")
