"""
Performs clustering on tracks based on a pre-computed affinity matrix.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering

def cluster_from_affinity(affinity_matrix: np.ndarray, n_clusters: int):
    """
    Clusters tracks by converting an affinity matrix to a distance matrix
    and using Agglomerative Clustering.

    Args:
        affinity_matrix (np.ndarray): An N x N matrix where higher values indicate
                                      greater similarity between tracks.
        n_clusters (int): The target number of clusters (conversations).

    Returns:
        np.ndarray: An array of cluster labels of length N, where N is the
                    number of tracks. Returns an empty array if the input
                    matrix is empty.
    """
    if affinity_matrix.size == 0:
        print("Warning: Affinity matrix is empty. Returning no clusters.")
        return np.array([])

    # Convert affinity to distance. A common method is 1 - affinity.
    # Higher affinity should correspond to smaller distance.
    distance_matrix = 1 - affinity_matrix

    # Ensure the distance matrix is valid for clustering algorithms:
    # - Non-negative: Distances cannot be negative.
    # - Symmetric: The distance from i to j is the same as j to i.
    # The geometric affinity function already produces a symmetric matrix.
    distance_matrix = np.maximum(distance_matrix, 0)

    # Initialize the clustering model.
    # 'precomputed' is used because we provide a distance matrix, not feature vectors.
    # 'average' linkage uses the average of the distances between all observations
    # of the two sets of observations.
    clustering_model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',  # Corrected from 'affinity' to 'metric'
        linkage='average'
    )

    # Fit the model to the distance matrix and predict the cluster for each track.
    labels = clustering_model.fit_predict(distance_matrix)

    print(f"Clustering complete. Assigned {len(labels)} tracks to {n_clusters} clusters.")
    return labels