import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from data_generation import generate_customer_data, preprocess_data
from baseline_kmeans import run_kmeans_baseline, determine_optimal_k
from algorithms import GA, PSO, DE, ABC, ES

st.set_page_config(page_title="Customer Segmentation with EC", layout="wide")

st.title("Clustering-Based Customer Segmentation")
st.markdown("### Comparing K-Means with Evolutionary Computation Algorithms")

# Sidebar - Data Settings
st.sidebar.header("Dataset Configuration")
n_samples = st.sidebar.slider("Number of Samples", 300, 1000, 500)
n_clusters_true = st.sidebar.slider("True Clusters (Simulation)", 2, 8, 4)

@st.cache_data
def get_data(samples, clusters):
    df = generate_customer_data(n_samples=samples, n_clusters=clusters)
    scaled_df, scaler = preprocess_data(df)
    return df, scaled_df, scaler

df, scaled_df, scaler = get_data(n_samples, n_clusters_true)

st.sidebar.header("Algorithm Configuration")
algorithm = st.sidebar.selectbox("Select Algorithm", ["K-Means", "GA", "PSO", "DE", "ABC", "ES"])
k_target = st.sidebar.slider("Target k (Clusters)", 2, 8, 4)

pop_size = st.sidebar.slider("Population Size", 10, 100, 50)
max_iter = st.sidebar.slider("Max Iterations", 10, 500, 100)

if st.sidebar.button("Run Clustering"):
    data_np = scaled_df.values
    
    # Run Algorithm
    with st.spinner(f'Running {algorithm} algorithm...'):
        if algorithm == "K-Means":
            res = run_kmeans_baseline(data_np, k_target)
        elif algorithm == "GA":
            res = GA(data_np, k_target, pop_size=pop_size, max_iter=max_iter).run()
        elif algorithm == "PSO":
            res = PSO(data_np, k_target, pop_size=pop_size, max_iter=max_iter).run()
        elif algorithm == "DE":
            res = DE(data_np, k_target, pop_size=pop_size, max_iter=max_iter).run()
        elif algorithm == "ABC":
            res = ABC(data_np, k_target, pop_size=pop_size, max_iter=max_iter).run()
        elif algorithm == "ES":
            res = ES(data_np, k_target, mu=max(2, pop_size//5), lambda_=pop_size, max_iter=max_iter).run()

    st.session_state['res'] = res
    st.session_state['algo_used'] = algorithm
    st.session_state['k_used'] = k_target
    st.session_state['data_np'] = data_np

if 'res' in st.session_state:
    res = st.session_state['res']
    algo_used = st.session_state['algo_used']
    k_used = st.session_state['k_used']
    data_np = st.session_state['data_np']

    # Display Metrics
    st.subheader(f"Clustering Results - {algo_used}")
    st.info(f"Results below were generated using the **{algo_used}** algorithm with k={k_used}.")
    col1, col2 = st.columns(2)
    col1.metric("Sum of Squared Errors (SSE)", f"{res['sse']:.4f}")
    col2.metric("Silhouette Score", f"{res['silhouette']:.4f}")
    
    # Plotting
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.write("**Cluster Subject Overlay (PCA - 2D)**")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data_np)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], c=res['labels'], cmap='viridis', s=30, alpha=0.7)
        
        # Centroids
        centroids_reshaped = np.array(res['centroids']).reshape(k_used, -1)
        centroids_2d = pca.transform(centroids_reshaped)
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='X', s=200, label='Centroids')
        
        plt.colorbar(scatter)
        plt.legend()
        st.pyplot(fig)
        
    with col_plot2:
        if algo_used != "K-Means":
            st.write("**Convergence Curve (Fitness history)**")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.plot(res['history'], marker='', color='blue', linewidth=2)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Best Fitness (SSE)")
            ax2.grid(True)
            st.pyplot(fig2)
        else:
            st.write("**Optimal K (Elbow Method)**")
            with st.spinner('Calculating elbow curve...'):
                optimal_k, sse_list, sil_list = determine_optimal_k(data_np, max_k=8)
            
            fig3, ax1 = plt.subplots(figsize=(8, 6))
            K = range(2, 9)
            color = 'tab:blue'
            ax1.set_xlabel('Number of clusters (k)')
            ax1.set_ylabel('SSE (Inertia)', color=color)
            ax1.plot(K, sse_list, color=color, marker='o', label='SSE')
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Silhouette Score', color=color)
            ax2.plot(K, sil_list, color=color, marker='s', label='Silhouette Score')
            ax2.tick_params(axis='y', labelcolor=color)
            
            fig3.tight_layout()
            st.pyplot(fig3)
