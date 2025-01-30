import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self


    def transform(self, X):
        return (X-self.mean_) / self.std_
    

    def inverse_transform(self, X_scaled):
        """Convert standardized data back to the original scale."""
        return (X_scaled * self.std_) + self.mean_
    