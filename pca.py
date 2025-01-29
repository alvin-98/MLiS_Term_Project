import pandas as pd
import numpy as np
import scipy as sp

class PCA:
    def __init__(self, n_components):
        """
        Parameters

        Args:
            n_components (int): number of principal components to retain.
        """
        self.n_components_ = n_components
        self.mean_ = None
        self.std_ = None
        self.pcs_ = None
        self.explained_variance_ = None
        self.loadings_ = None


    def fit(self, X):
        """
        Computes the principal components, feature loadings 
        and explained variance by PCA for the given dataset

        Args:
            X (np.ndarray of shape (n_samples, n_features)): input dataset for PCA

        Returns:
            PCA: an instance of the object
        """
        self.mean_ = X.mean(axis=0)
        cov_matrix = np.cov(X - self.mean_, rowvar=False)
        eigenvalues, eigenvectors = self._eigen_decomp(cov_matrix)

        self.pcs_ = eigenvectors[:, :self.n_components_]
        self.explained_variance_ = np.sum(eigenvalues[:self.n_components_]) / np.sum(eigenvalues)
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
    
