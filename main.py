from dbscan import DBSCAN
import numpy as np
from metrics import silhouette_score, davies_bouldin_score


# Generate or load your data
np.random.seed(42)
cluster1 = np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2))
cluster2 = np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2))
data = np.vstack((cluster1, cluster2))

# Fit DBSCAN
dbscan_model = DBSCAN(eps=0.8, min_samples=5)
dbscan_model.fit(data)

print("Cluster labels:", dbscan_model.labels_)
print("Number of clusters:", dbscan_model.n_clusters_)

labels = dbscan_model.labels_

sil_score = silhouette_score(data, labels)
dbi_score = davies_bouldin_score(data, labels)

print("Silhouette Score:", sil_score)
print("Davies-Bouldin Index:", dbi_score)
