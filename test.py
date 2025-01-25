# test_dbscan.py

import pytest
import numpy as np
from dbscan import DBSCAN as MyDBSCAN
from sklearn.cluster import DBSCAN as SkDBSCAN

def test_dbscan_equivalence():
    """
    Compare results of our from-scratch DBSCAN implementation (MyDBSCAN)
    with scikit-learn's DBSCAN (SkDBSCAN) on a synthetic dataset.
    """

    # 1. Generate synthetic data
    np.random.seed(42)
    cluster1 = np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2))
    cluster2 = np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2))
    data = np.vstack((cluster1, cluster2))

    eps = 0.8
    min_samples = 5

    # 2. My from-scratch DBSCAN
    my_dbscan = MyDBSCAN(eps=eps, min_samples=min_samples)
    my_dbscan.fit(data)
    my_labels = my_dbscan.labels_
    my_num_clusters = my_dbscan.n_clusters_           # excluding noise
    my_noise_count = np.sum(my_labels == -1)

    # 3. Scikit-learn DBSCAN
    sk_dbscan = SkDBSCAN(eps=eps, min_samples=min_samples)
    sk_dbscan.fit(data)
    sk_labels = sk_dbscan.labels_
    unique_sk_labels = set(sk_labels)
    sk_num_clusters = len(unique_sk_labels - {-1})
    sk_noise_count = np.sum(sk_labels == -1)

    # --- Compare the basic metrics ---
    # (1) Number of clusters found
    assert my_num_clusters == sk_num_clusters, (
        f"Different number of clusters: MyDBSCAN={my_num_clusters}, "
        f"SkDBSCAN={sk_num_clusters}"
    )

    # (2) Number of noise points
    assert my_noise_count == sk_noise_count, (
        f"Different number of noise points: MyDBSCAN={my_noise_count}, "
        f"SkDBSCAN={sk_noise_count}"
    )

    # --- Compare cluster size distribution (excluding noise) ---
    def get_cluster_size_distribution(labels):
        cluster_labels = set(labels) - {-1}
        sizes = []
        for c in cluster_labels:
            sizes.append(np.sum(labels == c))
        return sorted(sizes)

    my_cluster_sizes = get_cluster_size_distribution(my_labels)
    sk_cluster_sizes = get_cluster_size_distribution(sk_labels)

    # We only compare the sorted list of cluster sizes, since labeling
    # (e.g., 1 vs. 0) can differ but the distribution should match
    # if both algorithms cluster the data similarly.
    assert my_cluster_sizes == sk_cluster_sizes, (
        f"Cluster size distributions differ:\n"
        f"MyDBSCAN={my_cluster_sizes}\n"
        f"SkDBSCAN={sk_cluster_sizes}"
    )

