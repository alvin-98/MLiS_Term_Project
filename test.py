# test_dbscan.py
import pytest
import numpy as np
from pca import PCA as ourPCA
from dbscan import DBSCAN as MyDBSCAN
from hac import HAC as ourHAC
from sklearn.cluster import DBSCAN as SkDBSCAN
from sklearn.decomposition import PCA as SkPCA
from sklearn.cluster import AgglomerativeClustering as skAgglomerativeClustering


def test_myhac_equivalence():
    """
    Compare results of our from-scratch HAC implementation (MyHAC)
    with scikit-learn's AgglomerativeClustering on a synthetic dataset.
    We compare:
      - The number of clusters formed
      - The distribution of cluster sizes
    """
    # -----------------------------
    # 1. Generate synthetic data
    # -----------------------------
    np.random.seed(42)
    cluster1 = np.random.normal(loc=[0, 0], scale=0.4, size=(30, 2))
    cluster2 = np.random.normal(loc=[5, 5], scale=0.4, size=(30, 2))
    cluster3 = np.random.normal(loc=[10, 0], scale=0.4, size=(30, 2))
    data = np.vstack((cluster1, cluster2, cluster3))
    
    # We'll request 3 clusters
    n_clusters = 3
    
    # -----------------------------
    # 2. Our from-scratch HAC
    # -----------------------------
    # Assuming you have a class MyHAC(n_clusters, linkage='single' or 'complete')
    # that exposes `labels_` as in your code snippet.
    myhac = ourHAC(n_clusters=n_clusters, linkage='single')
    myhac.fit(data)
    my_labels = myhac.labels_
    unique_my_labels = set(my_labels)
    my_num_clusters = len(unique_my_labels)
    
    # -----------------------------
    # 3. scikit-learn AgglomerativeClustering
    # -----------------------------
    # Match the linkage method with your custom HAC ("single" or "complete").
    skhac = skAgglomerativeClustering(n_clusters=n_clusters, linkage='single')
    skhac.fit(data)
    sk_labels = skhac.labels_
    unique_sk_labels = set(sk_labels)
    sk_num_clusters = len(unique_sk_labels)
    
    # --- Compare number of clusters ---
    assert my_num_clusters == sk_num_clusters, (
        f"Different number of clusters: "
        f"MyHAC={my_num_clusters}, "
        f"SkHAC={sk_num_clusters}"
    )
    
    # --- Compare cluster size distributions ---
    # Because cluster label IDs can be permuted (e.g., your cluster "0" might 
    # be scikit-learn's cluster "2"), we only compare the sorted list of 
    # cluster sizes rather than exact label assignments.
    def get_cluster_size_distribution(labels):
        cluster_labels = set(labels)
        sizes = []
        for c in cluster_labels:
            sizes.append(np.sum(labels == c))
        return sorted(sizes)
    
    my_cluster_sizes = get_cluster_size_distribution(my_labels)
    sk_cluster_sizes = get_cluster_size_distribution(sk_labels)
    
    assert my_cluster_sizes == sk_cluster_sizes, (
        f"Cluster size distributions differ:\n"
        f"MyHAC={my_cluster_sizes}\n"
        f"SkHAC={sk_cluster_sizes}"
    )

    print("test_myhac_equivalence passed.")


def test_pca_equivalence():
    """
    Compare results of our from-scratch PCA (MyPCA)
    with scikit-learn's PCA on a synthetic dataset.
    """
    # 1. Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100 samples, 5 features

    # 2. Our PCA
    my_pca = ourPCA(n_components=2)   # use your PCA class
    my_pca.fit(X)
    my_transformed = my_pca.transform(X)

    # 3. scikit-learn PCA
    sk_pca = SkPCA(n_components=2, random_state=42)
    sk_pca.fit(X)
    sk_transformed = sk_pca.transform(X)

    # 4. Compare the explained variance ratio
    # -- In your code, explained_variance_ is set to a ratio or maybe an inverse ratio.
    #    We'll assume you update it to the typical fraction-of-variance form:
    #    sum(eigenvalues[:n_components]) / sum(eigenvalues).
    if my_pca.explained_variance_ > 1.0:
        # Possibly you stored the inverse ratio. Let's invert it to compare apples-to-apples:
        my_explained_var_ratio = 1.0 / my_pca.explained_variance_
    else:
        my_explained_var_ratio = my_pca.explained_variance_

    # sum of the top 2 components' explained variance ratio in sklearn
    sk_explained_var_ratio = sk_pca.explained_variance_ratio_[:2].sum()

    assert np.isclose(
        my_explained_var_ratio, sk_explained_var_ratio, atol=1e-5
    ), f"Explained variance ratio mismatch: {my_explained_var_ratio} vs. {sk_explained_var_ratio}"

    # 5. Compare transformations
    # -- Because PCA components can differ by a sign or rotation (especially in 2D),
    #    we typically compare correlation or simply check that they explain the same distances.
    #    For simplicity, let's just compare correlation of each principal component dimension.

    for i in range(2):
        corr = np.corrcoef(my_transformed[:, i], sk_transformed[:, i])[0, 1]
        # Usually we'd expect correlation to be ±1 if they differ only by sign.
        assert abs(corr) > 0.99, (
            f"Component {i} mismatch. Possibly a sign/flip or numerical difference. corr={corr}"
        )

    print("test_pca_equivalence passed successfully!")


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


test_myhac_equivalence()