import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from hac import HAC
from pca import PCA
from kmeans import KMeans
from dbscan import DBSCAN
from metrics import silhouette_score, davies_bouldin_score

file_path = "data/breast-cancer-wisconsin.data"

df = pd.read_csv(file_path, names=['Sample_code_number', 
                                    'Clump_thickness', 
                                    'Uniformity_of_cell_size', 
                                    'Uniformity_of_cell_shape', 
                                    'Marginal_adhesion',
                                    'Single_epithelial_cell_size',
                                    'Bare_nuclei',
                                    'Bland_chromatin',
                                    'Normal_nucleoli',
                                    'Mitoses',
                                    'Class'])


x = df.drop(["Sample_code_number"], axis=1).dropna()

# Handle missing values
x = x.replace('?', np.nan)

# Count rows with missing values
missing_count = x.isna().sum().sum()
print(f"Number of rows with missing values: {missing_count}")

# Drop rows with missing values
x = x.dropna()

classes = x["Class"]
x = x.drop(["Class"], axis=1)

# Automatically determine continuous features based on number of unique values
UNIQUE_THRESHOLD = 10  # Features with more unique values than this are considered continuous
continuous_columns = []
categorical_columns = []

for column in x.columns:
    n_unique = x[column].nunique()
    if n_unique > UNIQUE_THRESHOLD:
        continuous_columns.append(column)
    else:
        categorical_columns.append(column)

print("\nContinuous features:", continuous_columns)
print("Categorical features (to be one-hot encoded):", categorical_columns)

# Create copy of data
X = x.copy()

# Normalize continuous columns
X[continuous_columns] = (X[continuous_columns] - X[continuous_columns].mean()) / X[continuous_columns].std()


# One-hot encode categorical columns (if any)
if categorical_columns:
    # Get one-hot encoded columns
    X_encoded = pd.get_dummies(X.loc[:, categorical_columns], prefix=categorical_columns, prefix_sep='_', columns=categorical_columns, drop_first=True, dtype=float)
    
    # Drop original categorical columns and join encoded ones
    X = X.drop(columns=categorical_columns)
    X = pd.concat([X, X_encoded], axis=1)


# Convert to NumPy array
X_values = X.values  # Use X_values when numpy array is needed

# Plotting function
def plot_clusters(X_values, labels, title):
    # First perform PCA on the data for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_values)
    print(f"Explained variance ratio for {title}: {pca.explained_variance_:.3f}")
    
    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)
    
    # Plot clusters
    for label in unique_labels:
        mask = (labels == label)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   label=f'Cluster {label}',
                   alpha=0.6)
    
    plt.title(title)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.show()

# Compute class proportions in each cluster
def compute_class_proportions(labels, classes):
    df = pd.DataFrame({'Cluster': labels, 'Class': classes})
    
    # Get total samples per cluster
    cluster_sizes = df.groupby('Cluster').size()
    print("\nSamples per cluster:")
    print(cluster_sizes)
    
    # Get proportions
    print("\nClass proportions per cluster:")
    proportions = df.groupby('Cluster')['Class'].value_counts(normalize=True).unstack().fillna(0)
    print(proportions)

# Hierarchical Clustering
hac = HAC(n_clusters=2).fit(X_values)
plot_clusters(X_values, hac.labels_, "HAC Clustering")
compute_class_proportions(hac.labels_, classes)

# KMeans
kmeans = KMeans(n_clusters=2).fit(X_values)
plot_clusters(X_values, kmeans.labels_, "KMeans Clustering")
compute_class_proportions(kmeans.labels_, classes)

# DBSCAN
eps = 1.8
min_samples = 15
dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X_values)
plot_clusters(X_values, dbscan.labels_, "DBSCAN Clustering")
compute_class_proportions(dbscan.labels_, classes)

# Metrics
silhouette_hac = silhouette_score(X_values, hac.labels_)
silhouette_kmeans = silhouette_score(X_values, kmeans.labels_)
silhouette_dbscan = silhouette_score(X_values, dbscan.labels_)

davies_bouldin_hac = davies_bouldin_score(X_values, hac.labels_)
davies_bouldin_kmeans = davies_bouldin_score(X_values, kmeans.labels_)
davies_bouldin_dbscan = davies_bouldin_score(X_values, dbscan.labels_)

print(f"Silhouette score for HAC: {silhouette_hac:.3f}")
print(f"Silhouette score for KMeans: {silhouette_kmeans:.3f}")
print(f"Silhouette score for DBSCAN: {silhouette_dbscan:.3f}")

print(f"Davies-Bouldin score for HAC: {davies_bouldin_hac:.3f}")
print(f"Davies-Bouldin score for KMeans: {davies_bouldin_kmeans:.3f}")
print(f"Davies-Bouldin score for DBSCAN: {davies_bouldin_dbscan:.3f}")

# Statistical test for cluster-class relationship
def test_cluster_class_relationship(labels, classes):
    print("\nStatistical Test of Class Proportions Between Clusters")
    print("-" * 50)
    
    # Create contingency table
    contingency = pd.crosstab(labels, classes)
    
    # Perform chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    print(f"Contingency Table:")
    print(contingency)
    print(f"\nChi-square statistic: {chi2:.3f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Degrees of freedom: {dof}")
    
    # Interpret results
    alpha = 0.05
    print("\nInterpretation:")
    if p_value < alpha:
        print(f"The relationship between clusters and classes is statistically significant (p < {alpha})")
    else:
        print(f"No statistically significant relationship between clusters and classes (p >= {alpha})")

# Perform statistical tests for each clustering method
print("\nHAC Clustering:")
test_cluster_class_relationship(hac.labels_, classes)

print("\nKMeans Clustering:")
test_cluster_class_relationship(kmeans.labels_, classes)

print("\nDBSCAN Clustering:")
test_cluster_class_relationship(dbscan.labels_, classes)

# Statistical test for feature distributions between clusters
def test_feature_distributions(X, labels, feature_names):
    print("\nFeature Distribution Tests Between Clusters")
    print("-" * 50)
    
    # No need to convert X back to DataFrame since it already is one
    X_df = X.copy()
    X_df['Cluster'] = labels
    
    results = []
    for feature in feature_names:
        # Get values for each cluster
        clusters = X_df['Cluster'].unique()
        if len(clusters) != 2:  # Skip if not binary clustering
            print(f"Skipping {feature} - requires exactly 2 clusters")
            continue
            
        values_1 = X_df[X_df['Cluster'] == clusters[0]][feature]
        values_2 = X_df[X_df['Cluster'] == clusters[1]][feature]
        
        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(values_1, values_2, alternative='two-sided')
        
        results.append({
            'Feature': feature,
            'Statistic': statistic,
            'P-value': p_value
        })
    
    # Create and display results DataFrame
    results_df = pd.DataFrame(results)
    results_df['Significant'] = results_df['P-value'] < 0.05
    print("\nMann-Whitney U Test Results:")
    print(results_df)
    
    return results_df

# Get original feature names
feature_names = X.columns.tolist()

# Test feature distributions for each clustering method
print("\nHAC Clustering Feature Tests:")
test_feature_distributions(X, hac.labels_, feature_names)

print("\nKMeans Clustering Feature Tests:")
test_feature_distributions(X, kmeans.labels_, feature_names)

print("\nDBSCAN Clustering Feature Tests:")
test_feature_distributions(X, dbscan.labels_, feature_names)



