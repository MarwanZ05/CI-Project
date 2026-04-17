import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

from problem_formulation import ClusteringProblem

def determine_optimal_k(data, max_k=10, output_dir='plots'):
    """
    Uses the Elbow Method and Silhouette Score to determine the optimal number of clusters.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    sse = []
    silhouettes = []
    K = range(2, max_k + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(data, kmeans.labels_))
        
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('SSE (Inertia)', color=color)
    ax1.plot(K, sse, color=color, marker='o', label='SSE')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(K, silhouettes, color=color, marker='s', label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title('Elbow Method and Silhouette Score for Optimal k')
    plt.savefig(f'{output_dir}/kmeans_optimal_k.png', dpi=300)
    plt.close()
    
    optimal_k = K[np.argmax(silhouettes)]
    return optimal_k, sse, silhouettes

def run_kmeans_baseline(data, n_clusters):
    """
    Runs the K-Means baseline and evaluates it using our custom ClusteringProblem formulation
    to ensure fairness in comparison.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(data)
    
    problem = ClusteringProblem(data, n_clusters)
    
    # Extract centroids
    centroids = kmeans.cluster_centers_.flatten()
    
    # Calculate metrics
    sse = problem.evaluate_fitness(centroids)
    silhouette = problem.silhouette(centroids)
    labels = problem.get_labels(centroids)
    
    return {
        'centroids': centroids,
        'sse': sse,
        'silhouette': silhouette,
        'labels': labels
    }

if __name__ == "__main__":
    from data_generation import generate_customer_data, preprocess_data
    df = generate_customer_data(n_samples=500, n_clusters=4)
    data, _ = preprocess_data(df)
    
    # 1. Determine optimal k
    optimal_k, _, _ = determine_optimal_k(data.values)
    print(f"Optimal number of clusters based on Silhouette: {optimal_k}")
    
    # 2. Run baseline
    results = run_kmeans_baseline(data.values, n_clusters=optimal_k)
    print(f"K-Means Baseline - SSE: {results['sse']:.4f}, Silhouette: {results['silhouette']:.4f}")
