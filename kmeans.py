import numpy as np

class KMeans:
    """
    A simple K-means clustering implementation from scratch.

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
        Perform K-means clustering on the given dataset.

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
            # Step 1: Assign each data point to the nearest centroid
            self.labels_ = self._assign_labels(X)

            # Step 2: Calculate new centroids
            new_centroids = self._calculate_centroids(X)

            # Step 3: Check for convergence (if centroids do not change)
            centroid_shift = np.linalg.norm(new_centroids - self.centroids_)
            if centroid_shift < self.tol:
                break

            self.centroids_ = new_centroids

        return self

    def _assign_labels(self, X):
        """
        Assign each point in the dataset to the closest centroid.

        Parameters
        ----------
        X : np.ndarray
            2D numpy array of shape (n_samples, n_features) containing the data.

        Returns
        -------
        labels : np.ndarray
            1D array of labels indicating the assigned cluster for each point.
        """
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids_, axis=2), axis=1)
        return labels

    def _calculate_centroids(self, X):
        """
        Calculate the new centroids by taking the mean of the points in each cluster.

        Parameters
        ----------
        X : np.ndarray
            2D numpy array of shape (n_samples, n_features) containing the data.

        Returns
        -------
        centroids : np.ndarray
            2D numpy array of shape (n_clusters, n_features) containing the new centroids.
        """
        centroids = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])
        return centroids

# Example usage
if __name__ == "__main__":
    # Generate some synthetic 2D data
    np.random.seed(42)
    cluster1 = np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2))
    cluster2 = np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2))
    cluster3 = np.random.normal(loc=[10, 0], scale=0.5, size=(50, 2))
    data = np.vstack((cluster1, cluster2, cluster3))

    # Create and fit KMeans model
    n_clusters = 3
    model = KMeans(n_clusters=n_clusters)
    model.fit(data)

    # Print the labels and centroids
    print("Labels assigned to each point:")
    print(model.labels_)
    print(f"Centroids found: {model.centroids_}")
