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
        self.loadings = None


    def fit(self, X):
        """
        Compute parameters and principal components for the given dataset

        Args:
            X (np.ndarray): dataset

        Returns:
            self: an instance of the object
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
        Projects data into principal components

        Args:
            X (np.ndarray): data to transform.

        Returns:
            np.ndarray: data projected onto n principal components
        """
        X = X - self.mean_
        return np.dot(X, self.pcs_)


    def _eigen_decomp(self, cov_matrix):
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        return eigenvalues, eigenvectors
