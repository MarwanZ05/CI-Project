import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data_generation import generate_customer_data, preprocess_data
from baseline_kmeans import run_kmeans_baseline
from algorithms import GA, PSO, DE, ABC, ES

def run_k_experiments(max_k=8, num_runs=3, output_dir='results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    results = []
    algorithms = ['K-Means', 'GA', 'PSO', 'DE', 'ABC', 'ES']
    
    # We will loop through k from 2 to max_k
    for k in range(2, max_k + 1):
        print(f"\n--- Running Experiments for k={k} ---")
        
        # Generate the dataset for this specific k
        df = generate_customer_data(n_samples=500, n_clusters=k)
        data, scaler = preprocess_data(df)
        data_np = data.values
        
        # Generate unique seeds for this k
        seeds = np.random.randint(0, 100000, size=num_runs)
        
        for algo in algorithms:
            print(f"Running {algo}...")
            
            for run_idx, s in enumerate(tqdm(seeds, desc=f"{algo} (k={k})", leave=False)):
                if algo == "K-Means":
                    # K-Means handles its own randomness with n_init
                    res = run_kmeans_baseline(data_np, n_clusters=k)
                elif algo == "GA":
                    res = GA(data_np, n_clusters=k, pop_size=50, max_iter=100).run(seed=s)
                elif algo == "PSO":
                    res = PSO(data_np, n_clusters=k, pop_size=50, max_iter=100).run(seed=s)
                elif algo == "DE":
                    res = DE(data_np, n_clusters=k, pop_size=50, max_iter=100).run(seed=s)
                elif algo == "ABC":
                    res = ABC(data_np, n_clusters=k, pop_size=50, max_iter=100).run(seed=s)
                elif algo == "ES":
                    res = ES(data_np, n_clusters=k, mu=10, lambda_=50, max_iter=100).run(seed=s)
                
                results.append({
                    'k': k,
                    'Algorithm': algo,
                    'Run': run_idx,
                    'SSE': res['sse'],
                    'Silhouette': res['silhouette']
                })
                
    # Save the raw results
    df_results = pd.DataFrame(results)
    df_results.to_csv(f'{output_dir}/k_experiments.csv', index=False)
    
    # Generate the plots
    print("\nGenerating Visualizations...")
    generate_k_visualizations(df_results, output_dir)
    print("Done!")

def generate_k_visualizations(df, output_dir):
    sns.set_theme(style="whitegrid")
    
    # Calculate the mean per algorithm per k
    mean_df = df.groupby(['k', 'Algorithm']).mean().reset_index()
    
    # Plot 1: SSE vs k
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=mean_df, x='k', y='SSE', hue='Algorithm', marker='o', linewidth=2)
    plt.title('Average SSE vs. Number of Clusters (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Sum of Squared Errors (SSE)')
    plt.xticks(range(2, mean_df['k'].max() + 1))
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plot_k_sse.png', dpi=300)
    plt.close()
    
    # Plot 2: Silhouette vs k
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=mean_df, x='k', y='Silhouette', hue='Algorithm', marker='s', linewidth=2)
    plt.title('Average Silhouette Score vs. Number of Clusters (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.xticks(range(2, mean_df['k'].max() + 1))
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plot_k_silhouette.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    run_k_experiments(max_k=8, num_runs=3)
