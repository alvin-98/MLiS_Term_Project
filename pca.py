import numpy as np
import scipy as sp
import pandas as pd

class PCA:
    def __init__(self, n_components):
        """
        Principal Component Analysis

        Args:
            n_components (int): the number of principal components
        """

        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("invalid component number. it must be an integer value more than 0")
        
        self.n_components = n_components
        self.pcs = None
        self.mean = None
        self.feature_loadings = None
        self.explained_variance = None


    def fit(self, X):
        """
        Calculate Principal Components and Feature Loadings

        Args:
            X (numpy.ndarray or pandas.DataFrame): input data (n_samples, n_features)
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            X = X.values
            X = X.astype(np.float64)
        else: 
            self.feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    
        # covariance matrix with centered data, eigenvalue decomposition, extraction of PCs
        cov_matrix = self._compute_cov(X)
        eigenvalues, eigenvectors = self._eigen_decomp(cov_matrix)
        eigenvectors_sorted, eigvalues_sorted = self._sort_eigenvps(eigenvalues, eigenvectors)
        self.pcs = self._select_n_pcs(eigenvectors_sorted, self.n_components)

        # calculation of feature loadings on each of selected components
        feature_loadings = self.pcs * np.sqrt(eigenvalues[:self.n_components])
        loadings_df = pd.DataFrame(
            feature_loadings, 
            index=self.feature_names, 
            columns=[f"PC{i+1}" for i in range(len(eigenvalues))]
        )
        self.feature_loadings = loadings_df

        # explained variance
        total_variance = np.sum(eigvalues_sorted)
        self.explained_variance = eigvalues_sorted[:self.n_components] / total_variance


    def transform(self, X):
        """
        Project data into its principal components

        Args:
            X (numpy.ndarray or pandas.DataFrame): data to be transformed

        Returns:
            numpy.ndarray: The transformed data with shape (n_samples, n_components)
        """

        if self.pcs is None or self.mean is None:
            raise RuntimeError("Call 'fit' before using 'transform'")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            X = X.astype(np.float64)

        return self._transform_data(X-self.mean, self.pcs)
    

    def get_loadings(self):
        """
        Loading of each feature for the selected Principal Components

        Returns:
        pandas.DataFrame: Feature loadings with features as rows and principal components as columns.
        """
        if self.feature_loadings is None:
            raise AttributeError("Feature loadings are not yet set. call 'fit' on your data")
        return self.feature_loadings
    
    def explained_variance_(self):
        """
        Explained variance ratio for each of the compenents

        Returns:
        pandas.Series: Explained variance ratio for each principal component.
        """
        if self.explained_variance is None:
            raise AttributeError("Explained variance is not yet set. call 'fit' on your data")
        return pd.Series(
            self.explained_variance,
            index=[f"PC{i+1}" for i in range(len(self.explained_variance))]
        )
    

    def _compute_cov(self, X):
        """
        Covariance matrix

        Args:
            X (np.ndarray): data (n_samples, n_features)

        Returns:
            np.ndarray: covariance matrix (n_features, n_features)
        """
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        return np.cov(X.T)
        

    def _eigen_decomp(self, cov_matrix):
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            
        return eigenvalues, eigenvectors
        

    def _sort_eigenvps(self, eigenvalues, eigenvectors):
        idx = np.arange(0, len(eigenvalues))
        idx = ([x for _,x in sorted(zip(eigenvalues, idx))])[::-1]
        
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = eigenvalues[:, idx]

        return eigenvectors, eigenvalues
        

    def _select_n_pcs(self, eigenvectors, n_components):
        return eigenvectors[:, :n_components]
        

    def _transform_data(self, X, eigenvectors):
        return np.dot(X, eigenvectors)
