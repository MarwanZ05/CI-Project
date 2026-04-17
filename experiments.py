import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm

from data_generation import generate_customer_data, preprocess_data
from baseline_kmeans import run_kmeans_baseline
from algorithms import GA, PSO, DE, ABC, ES

def run_systematic_experiments(data, n_clusters, num_runs=30, output_dir='results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    results = []
    seeds = np.random.randint(0, 100000, size=num_runs)
    
    # Define configurations to test
    # This covers multiple variation, selection, mutation, and initialization strategies essentially by using the different algorithms.
    # GA uses tournament+gaussian, DE uses differential mutation+binomial crossover, PSO uses swarm mechanics, ABC uses bee phases, ES uses mu+lambda
    algorithms_config = {
        'GA': lambda s: GA(data, n_clusters, pop_size=50, max_iter=100).run(seed=s),
        'PSO': lambda s: PSO(data, n_clusters, pop_size=50, max_iter=100).run(seed=s),
        'DE': lambda s: DE(data, n_clusters, pop_size=50, max_iter=100).run(seed=s),
        'ABC': lambda s: ABC(data, n_clusters, pop_size=50, max_iter=100).run(seed=s),
        'ES': lambda s: ES(data, n_clusters, mu=20, lambda_=100, max_iter=100).run(seed=s),
    }

    print("Running baseline K-Means...")
    kmeans_sse = []
    kmeans_sil = []
    for s in tqdm(seeds):
        res = run_kmeans_baseline(data, n_clusters) # K-Means is deterministic if well separated, but we can just run it. We'll use n_init=10 inherently inside.
        kmeans_sse.append(res['sse'])
        kmeans_sil.append(res['silhouette'])
        
    results.append({
        'Algorithm': 'K-Means',
        'Mean_SSE': np.mean(kmeans_sse),
        'Std_SSE': np.std(kmeans_sse),
        'Min_SSE': np.min(kmeans_sse),
        'Mean_Silhouette': np.mean(kmeans_sil)
    })

    for algo_name, algo_func in algorithms_config.items():
        print(f"Running experiments for {algo_name}...")
        sse_list = []
        sil_list = []
        histories = []
        for s in tqdm(seeds):
            res = algo_func(s)
            sse_list.append(res['sse'])
            sil_list.append(res['silhouette'])
            histories.append(res['history'])
            
        # Save history for convergence plotting later
        np.save(f'{output_dir}/{algo_name}_history.npy', np.mean(histories, axis=0))
            
        results.append({
            'Algorithm': algo_name,
            'Mean_SSE': np.mean(sse_list),
            'Std_SSE': np.std(sse_list),
            'Min_SSE': np.min(sse_list),
            'Mean_Silhouette': np.mean(sil_list)
        })
        
    # Save parameters and seeds for reproducibility
    with open(f'{output_dir}/experiment_metadata.json', 'w') as f:
        json.dump({'seeds': seeds.tolist(), 'num_runs': num_runs, 'n_clusters': n_clusters}, f)
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(f'{output_dir}/summary_results.csv', index=False)
    print("\nExperiment Summary:")
    print(df_results)
    
if __name__ == "__main__":
    df = generate_customer_data(n_samples=500, n_clusters=4)
    data, scaler = preprocess_data(df)
    
    run_systematic_experiments(data.values, n_clusters=4)
