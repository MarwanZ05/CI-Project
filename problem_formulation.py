import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

class ClusteringProblem:
    """
    Formalizes the clustering problem as an optimization problem.
    Objective: Optimize cluster centroids to minimize intra-cluster distance (Sum of Squared Errors - SSE).
    Problem Type: Unconstrained / Box-Constrained continuous optimization.
    """
    def __init__(self, data, n_clusters):
        self.data = data
        self.n_clusters = n_clusters
        self.n_features = data.shape[1]
        self.bounds = np.vstack((np.min(data, axis=0), np.max(data, axis=0))).T # shape: (n_features, 2)
        
    def evaluate_fitness(self, centroids):
        """
        Calculates SSE (Sum of Squared Errors) for a given set of centroids.
        centroids: flattened array of shape (n_clusters * n_features) or reshaped to (n_clusters, n_features)
        Returns a float fitness value (lower is better, SSE)
        """
        centroids = centroids.reshape(self.n_clusters, self.n_features)
        
        # Calculate distances from each data point to each centroid
        distances = cdist(self.data, centroids, metric='euclidean')
        
        # Find the closest centroid for each point
        min_distances = np.min(distances, axis=1)
        
        # SSE is sum of squared distances to the nearest centroid
        sse = np.sum(min_distances ** 2)
        return sse
        
    def get_labels(self, centroids):
        """
        Returns cluster assignments for the dataset given centroids.
        """
        centroids = centroids.reshape(self.n_clusters, self.n_features)
        distances = cdist(self.data, centroids, metric='euclidean')
        return np.argmin(distances, axis=1)
        
    def silhouette(self, centroids):
        """
        Calculates the silhouette score for the given centroids.
        Requires > 1 cluster and at least one point in each cluster for numeric stability.
        """
        labels = self.get_labels(centroids)
        if len(np.unique(labels)) <= 1 or len(np.unique(labels)) != self.n_clusters:
            return -1.0 # Invalid clustering (collapsing to single cluster)
        return silhouette_score(self.data, labels)

    def enforce_bounds(self, centroids):
        """
        Constrains centroids to remain within the feature bounding box of the dataset.
        (Constraint-handling via clipping to feasible region)
        """
        centroids = centroids.reshape(self.n_clusters, self.n_features)
        for i in range(self.n_features):
            centroids[:, i] = np.clip(centroids[:, i], self.bounds[i, 0], self.bounds[i, 1])
        return centroids.flatten()
