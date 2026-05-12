import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data_generation import generate_customer_data, preprocess_data
from baseline_kmeans import run_kmeans_baseline
from algorithms import GA, PSO, DE, ABC, ES

def run_systematic_experiments(data, n_clusters, num_runs=10, output_dir='results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    seeds = np.random.randint(0, 100000, size=num_runs)
    
    # Define experimental configurations
    experiments = [
        # 1. Population Size Sweep (max_iter=100)
        {'Algorithm': 'GA', 'Group': 'PopSize', 'Config': 'pop=20', 'Func': lambda s, d, k: GA(d, k, pop_size=20, max_iter=100).run(seed=s)},
        {'Algorithm': 'GA', 'Group': 'PopSize', 'Config': 'pop=50', 'Func': lambda s, d, k: GA(d, k, pop_size=50, max_iter=100).run(seed=s)},
        {'Algorithm': 'GA', 'Group': 'PopSize', 'Config': 'pop=100', 'Func': lambda s, d, k: GA(d, k, pop_size=100, max_iter=100).run(seed=s)},
        
        {'Algorithm': 'PSO', 'Group': 'PopSize', 'Config': 'pop=20', 'Func': lambda s, d, k: PSO(d, k, pop_size=20, max_iter=100).run(seed=s)},
        {'Algorithm': 'PSO', 'Group': 'PopSize', 'Config': 'pop=50', 'Func': lambda s, d, k: PSO(d, k, pop_size=50, max_iter=100).run(seed=s)},
        {'Algorithm': 'PSO', 'Group': 'PopSize', 'Config': 'pop=100', 'Func': lambda s, d, k: PSO(d, k, pop_size=100, max_iter=100).run(seed=s)},
        
        {'Algorithm': 'DE', 'Group': 'PopSize', 'Config': 'pop=20', 'Func': lambda s, d, k: DE(d, k, pop_size=20, max_iter=100).run(seed=s)},
        {'Algorithm': 'DE', 'Group': 'PopSize', 'Config': 'pop=50', 'Func': lambda s, d, k: DE(d, k, pop_size=50, max_iter=100).run(seed=s)},
        {'Algorithm': 'DE', 'Group': 'PopSize', 'Config': 'pop=100', 'Func': lambda s, d, k: DE(d, k, pop_size=100, max_iter=100).run(seed=s)},

        {'Algorithm': 'ABC', 'Group': 'PopSize', 'Config': 'pop=20', 'Func': lambda s, d, k: ABC(d, k, pop_size=20, max_iter=100).run(seed=s)},
        {'Algorithm': 'ABC', 'Group': 'PopSize', 'Config': 'pop=50', 'Func': lambda s, d, k: ABC(d, k, pop_size=50, max_iter=100).run(seed=s)},
        {'Algorithm': 'ABC', 'Group': 'PopSize', 'Config': 'pop=100', 'Func': lambda s, d, k: ABC(d, k, pop_size=100, max_iter=100).run(seed=s)},
        
        {'Algorithm': 'ES', 'Group': 'PopSize', 'Config': 'pop=20', 'Func': lambda s, d, k: ES(d, k, mu=4, lambda_=20, max_iter=100).run(seed=s)},
        {'Algorithm': 'ES', 'Group': 'PopSize', 'Config': 'pop=50', 'Func': lambda s, d, k: ES(d, k, mu=10, lambda_=50, max_iter=100).run(seed=s)},
        {'Algorithm': 'ES', 'Group': 'PopSize', 'Config': 'pop=100', 'Func': lambda s, d, k: ES(d, k, mu=20, lambda_=100, max_iter=100).run(seed=s)},

        # 2. Iteration Sweep (pop_size=50)
        {'Algorithm': 'GA', 'Group': 'Iter', 'Config': 'iter=50', 'Func': lambda s, d, k: GA(d, k, pop_size=50, max_iter=50).run(seed=s)},
        {'Algorithm': 'GA', 'Group': 'Iter', 'Config': 'iter=100', 'Func': lambda s, d, k: GA(d, k, pop_size=50, max_iter=100).run(seed=s)},
        {'Algorithm': 'GA', 'Group': 'Iter', 'Config': 'iter=200', 'Func': lambda s, d, k: GA(d, k, pop_size=50, max_iter=200).run(seed=s)},
        
        {'Algorithm': 'PSO', 'Group': 'Iter', 'Config': 'iter=50', 'Func': lambda s, d, k: PSO(d, k, pop_size=50, max_iter=50).run(seed=s)},
        {'Algorithm': 'PSO', 'Group': 'Iter', 'Config': 'iter=100', 'Func': lambda s, d, k: PSO(d, k, pop_size=50, max_iter=100).run(seed=s)},
        {'Algorithm': 'PSO', 'Group': 'Iter', 'Config': 'iter=200', 'Func': lambda s, d, k: PSO(d, k, pop_size=50, max_iter=200).run(seed=s)},
        
        {'Algorithm': 'DE', 'Group': 'Iter', 'Config': 'iter=50', 'Func': lambda s, d, k: DE(d, k, pop_size=50, max_iter=50).run(seed=s)},
        {'Algorithm': 'DE', 'Group': 'Iter', 'Config': 'iter=100', 'Func': lambda s, d, k: DE(d, k, pop_size=50, max_iter=100).run(seed=s)},
        {'Algorithm': 'DE', 'Group': 'Iter', 'Config': 'iter=200', 'Func': lambda s, d, k: DE(d, k, pop_size=50, max_iter=200).run(seed=s)},

        {'Algorithm': 'ABC', 'Group': 'Iter', 'Config': 'iter=50', 'Func': lambda s, d, k: ABC(d, k, pop_size=50, max_iter=50).run(seed=s)},
        {'Algorithm': 'ABC', 'Group': 'Iter', 'Config': 'iter=100', 'Func': lambda s, d, k: ABC(d, k, pop_size=50, max_iter=100).run(seed=s)},
        {'Algorithm': 'ABC', 'Group': 'Iter', 'Config': 'iter=200', 'Func': lambda s, d, k: ABC(d, k, pop_size=50, max_iter=200).run(seed=s)},
        
        {'Algorithm': 'ES', 'Group': 'Iter', 'Config': 'iter=50', 'Func': lambda s, d, k: ES(d, k, mu=10, lambda_=50, max_iter=50).run(seed=s)},
        {'Algorithm': 'ES', 'Group': 'Iter', 'Config': 'iter=100', 'Func': lambda s, d, k: ES(d, k, mu=10, lambda_=50, max_iter=100).run(seed=s)},
        {'Algorithm': 'ES', 'Group': 'Iter', 'Config': 'iter=200', 'Func': lambda s, d, k: ES(d, k, mu=10, lambda_=50, max_iter=200).run(seed=s)},
        
        # 3. Algorithm-Specific Sweeps (pop=50, max_iter=100)
        {'Algorithm': 'GA', 'Group': 'Specific', 'Config': 'mut=0.01', 'Func': lambda s, d, k: GA(d, k, pop_size=50, max_iter=100, mutation_rate=0.01).run(seed=s)},
        {'Algorithm': 'GA', 'Group': 'Specific', 'Config': 'mut=0.2', 'Func': lambda s, d, k: GA(d, k, pop_size=50, max_iter=100, mutation_rate=0.2).run(seed=s)},
        
        {'Algorithm': 'PSO', 'Group': 'Specific', 'Config': 'c1=1.0,c2=2.0', 'Func': lambda s, d, k: PSO(d, k, pop_size=50, max_iter=100, c1=1.0, c2=2.0).run(seed=s)},
        {'Algorithm': 'PSO', 'Group': 'Specific', 'Config': 'c1=2.0,c2=1.0', 'Func': lambda s, d, k: PSO(d, k, pop_size=50, max_iter=100, c1=2.0, c2=1.0).run(seed=s)},
        {'Algorithm': 'PSO', 'Group': 'Specific', 'Config': 'w=0.5', 'Func': lambda s, d, k: PSO(d, k, pop_size=50, max_iter=100, w=0.5).run(seed=s)},
        {'Algorithm': 'PSO', 'Group': 'Specific', 'Config': 'w=0.9', 'Func': lambda s, d, k: PSO(d, k, pop_size=50, max_iter=100, w=0.9).run(seed=s)},
        
        {'Algorithm': 'DE', 'Group': 'Specific', 'Config': 'F=0.5,CR=0.5', 'Func': lambda s, d, k: DE(d, k, pop_size=50, max_iter=100, F=0.5, CR=0.5).run(seed=s)},
        {'Algorithm': 'DE', 'Group': 'Specific', 'Config': 'F=0.9,CR=0.9', 'Func': lambda s, d, k: DE(d, k, pop_size=50, max_iter=100, F=0.9, CR=0.9).run(seed=s)},
    ]
    
    detailed_results = []
    
    # Baseline K-Means
    print("Running baseline K-Means...")
    for run_idx, s in enumerate(tqdm(seeds, desc="K-Means")):
        res = run_kmeans_baseline(data, n_clusters) # n_init handles randomness
        detailed_results.append({
            'Algorithm': 'K-Means',
            'Group': 'Baseline',
            'Config': 'default',
            'Run': run_idx,
            'SSE': res['sse'],
            'Silhouette': res['silhouette']
        })

    # Run EC Configurations
    for exp in experiments:
        algo = exp['Algorithm']
        group = exp['Group']
        config = exp['Config']
        func = exp['Func']
        
        print(f"Running {algo} | Group: {group} | Config: {config}")
        
        for run_idx, s in enumerate(tqdm(seeds, desc=f"{algo} ({config})", leave=False)):
            res = func(s, data, n_clusters)
            detailed_results.append({
                'Algorithm': algo,
                'Group': group,
                'Config': config,
                'Run': run_idx,
                'SSE': res['sse'],
                'Silhouette': res['silhouette']
            })
            
    # Save detailed results
    df_detailed = pd.DataFrame(detailed_results)
    df_detailed.to_csv(f'{output_dir}/detailed_results.csv', index=False)
    
    # Generate summary results
    summary = df_detailed.groupby(['Algorithm', 'Group', 'Config']).agg({
        'SSE': ['mean', 'std', 'min'],
        'Silhouette': ['mean', 'std']
    }).reset_index()
    
    # Flatten columns
    summary.columns = ['Algorithm', 'Group', 'Config', 'Mean_SSE', 'Std_SSE', 'Min_SSE', 'Mean_Silhouette', 'Std_Silhouette']
    summary.to_csv(f'{output_dir}/summary_results.csv', index=False)
    
    print("\nExperiment Summary:")
    print(summary)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(df_detailed, output_dir)
    print("Experiments complete.")

def generate_visualizations(df, output_dir):
    sns.set_theme(style="whitegrid")
    
    for group in df['Group'].unique():
        if group == 'Baseline':
            continue
            
        group_df = df[df['Group'] == group]
        
        # Plot SSE
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=group_df, x='Algorithm', y='SSE', hue='Config')
        plt.title(f'SSE Comparison by {group}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/plot_sse_{group}.png')
        plt.close()
        
        # Plot Silhouette
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=group_df, x='Algorithm', y='Silhouette', hue='Config')
        plt.title(f'Silhouette Score Comparison by {group}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/plot_silhouette_{group}.png')
        plt.close()

if __name__ == "__main__":
    df = generate_customer_data(n_samples=500, n_clusters=4)
    data, scaler = preprocess_data(df)
    
    run_systematic_experiments(data.values, n_clusters=4, num_runs=10)
