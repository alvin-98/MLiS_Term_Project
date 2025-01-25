import numpy as np
import scipy as sp


# def distance_matrix(X):
#     n_clusters = X.shape[0]

#     distance_matrix = np.zeros((n_clusters, n_clusters))

#     for i in range(n_clusters):
#          for j in range(n_clusters):
#             distance_matrix[i][j] = sp.spatial.distance.euclidean(X[i], X[j])

#     return distance_matrix


class HAC:
    def __init__(self, n_clusters, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X):
        n_clusters = X.shape[0]
        clusters = [[i] for i in range(n_clusters)]

        while len(clusters) > self.n_clusters:
            min_distance = np.inf
            clusters_to_merge = (0, 0)
            
            for i in range(0, len(clusters), 1):
                for j in range(i+1, len(clusters), 1):
                    dist_ij = self._calc_linkage(X, clusters[i], clusters[j])
                    if (dist_ij < min_distance):
                        min_distance = dist_ij
                        clusters_to_merge = (i, j)

            i, j = clusters_to_merge
            clusters[i].extend(clusters[j])
            del clusters[j]

        self.labels_ = np.zeros(n_clusters, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels_[idx] = cluster_id

        return clusters
    
    def _calc_linkage(self, X, cluster_i, cluster_j):
        if self.linkage == 'single':
            return min(
                sp.spatial.distance.euclidean(X[k], X[l]) 
                for k in cluster_i 
                for l in cluster_j)
        elif self.linkage == 'complete':
            return max(
                sp.spatial.distance.euclidean(X[k], X[l]) 
                for k in cluster_i 
                for l in cluster_j)
        else:
            raise ValueError('Invalid linkage type')
            

    
    
    

hac = HAC(n_clusters=4, linkage='complete')

# np.random.seed(42)
# data = np.random.rand(10, 2)
# print("Data:\n", data)

# import pandas as pd

# file_path = "data/breast-cancer-wisconsin.data"

# df = pd.read_csv(file_path, names=['Sample_code_number', 
#                                     'Clump_thickness', 
#                                     'Uniformity_of_cell_size', 
#                                     'Uniformity_of_cell_shape', 
#                                     'Marginal_adhesion',
#                                     'Single_epithelial_cell_size',
#                                     'Bare_nuclei',
#                                     'Bland_chromatin',
#                                     'Normal_nucleoli',
#                                     'Mitoses',
#                                     'Class'])

# df['Bare_nuclei'] = pd.to_numeric(df['Bare_nuclei'], errors='coerce')

# x = df.drop(["Sample_code_number"], axis=1).dropna()
# classes = x["Class"]
# X = x.drop(["Class"], axis=1)

# df.dropna()

# X = df.values  # Convert DataFrame to numpy array
# X = X.astype(np.float64)

# final_clusters = hac.fit(X)
# print("Final Clusters:", final_clusters)

# custom_labels = hac.labels_
# print(custom_labels)


# from sklearn.cluster import AgglomerativeClustering

# sklearn_model = AgglomerativeClustering(
#     n_clusters=4,
#     linkage='complete',  # Ensure linkage matches
#     metric='euclidean'  # Ensure metric matches
# )
# sklearn_labels = sklearn_model.fit_predict(X)
# print(sklearn_labels)