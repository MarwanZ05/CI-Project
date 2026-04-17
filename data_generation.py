import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_customer_data(n_samples=500, n_features=4, n_clusters=4, random_state=42):
    """
    Simulates a realistic customer dataset with numerical features:
    0: Age
    1: Income (1000s)
    2: Spending Score (1-100)
    3: Purchase Frequency
    """
    X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, 
                           cluster_std=1.2, random_state=random_state)
    
    # Scale and shift to make them look like real features
    # Age: 18 - 70
    X[:, 0] = np.interp(X[:, 0], (X[:, 0].min(), X[:, 0].max()), (18, 70))
    # Income: 20k - 150k
    X[:, 1] = np.interp(X[:, 1], (X[:, 1].min(), X[:, 1].max()), (20, 150))
    # Spending Score: 1 - 100
    X[:, 2] = np.interp(X[:, 2], (X[:, 2].min(), X[:, 2].max()), (1, 100))
    # Purchase frequency: 1 - 50 times a year
    X[:, 3] = np.interp(X[:, 3], (X[:, 3].min(), X[:, 3].max()), (1, 50))
    
    df = pd.DataFrame(X, columns=['Age', 'Annual_Income_k', 'Spending_Score', 'Purchase_Frequency'])
    df['True_Cluster'] = y_true
    
    return df

def preprocess_data(df):
    """
    Scales the data using StandardScaler. Clustering relies on distance metrics,
    meaning features must be on the same scale.
    """
    scaler = StandardScaler()
    features = ['Age', 'Annual_Income_k', 'Spending_Score', 'Purchase_Frequency']
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    return df_scaled, scaler

def perform_eda(df, output_dir='plots'):
    """
    Performs Exploratory Data Analysis, generating summary statistics and plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    features = ['Age', 'Annual_Income_k', 'Spending_Score', 'Purchase_Frequency']
    
    # Summary stats
    stats = df[features].describe()
    
    # Pairplot to see distributions and correlations
    plt.figure(figsize=(10, 8))
    sns.pairplot(df, vars=features, hue='True_Cluster', palette='viridis', corner=True)
    plt.savefig(f'{output_dir}/eda_pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap")
    plt.savefig(f'{output_dir}/eda_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats

if __name__ == "__main__":
    df = generate_customer_data(n_samples=500, n_clusters=4)
    print("Dataset generated:")
    print(df.head())
    
    stats = perform_eda(df)
    print("\nSummary Statistics:")
    print(stats)
    
    df_scaled, scaler = preprocess_data(df)
    print("\nScaled Data Head:")
    print(df_scaled.head())
