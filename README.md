# Clustering-Based Customer Segmentation with CI/EC Algorithms

This repository contains a comprehensive, submission-ready academic project designed to benchmark Computational Intelligence (CI) and Evolutionary Computation (EC) algorithms against a traditional K-Means baseline for the problem of customer segmentation.

The project models clustering as an optimization problem where the objective is to minimize the Sum of Squared Errors (SSE) by iteratively optimizing the positions of cluster centroids.

## Key Features
- **Synthetic Data Generation**: Simulates realistic customer features (Age, Income, Spending Score, Purchase Frequency) with known true clusters for reliable evaluation.
- **Interactive UI**: A Streamlit frontend allows for real-time adjustments of algorithmic hyperparameters and visualizes the resulting clusters.
- **Evolutionary Algorithms**: Implements 5 distinct EC algorithms from scratch:
  - Genetic Algorithm (GA)
  - Particle Swarm Optimization (PSO)
  - Differential Evolution (DE)
  - Artificial Bee Colony (ABC)
  - Evolution Strategies (ES)
- **Extensive Benchmarking**: Automated scripts for running extensive hyperparameter sweeps and variable `k` tests, outputting detailed CSV results and comparative plots.

## Repository Structure

### Core Components
* **`app.py`**: The Streamlit frontend application. Run this to interactively explore algorithms, tweak their parameters, and view 2D PCA visual overlays and convergence curves.
* **`problem_formulation.py`**: Contains the `ClusteringProblem` class. This defines the search space, enforces feature boundaries, and calculates the fitness function (SSE) and validation metric (Silhouette Score).
* **`data_generation.py`**: Generates the synthetic dataset and provides standard scaling preprocessing. It also includes an EDA function to generate data distribution histograms and correlation heatmaps.
* **`baseline_kmeans.py`**: Implements the traditional K-Means approach and contains logic to generate the optimal `k` Elbow Curve.

### Evolutionary Algorithms (`algorithms/`)
* **`base_ec.py`**: Provides foundational classes (`ParentSelection`, `VariationOperators`, `SurvivorSelection`) containing standard mechanisms (e.g., tournament selection, gaussian mutation) shared across algorithms.
* **`ga.py`, `pso.py`, `de.py`, `abc.py`, `es.py`**: The specific implementations of each evolutionary algorithm, tailored to the continuous optimization problem of centroid placement.

### Experiments & Analysis
* **`experiments.py`**: A systematic benchmarking script. It runs exhaustive parameter sweeps (e.g., varying population sizes, maximum iterations, and algorithm-specific parameters) across all algorithms, outputting raw results and visual boxplots.
* **`k_experiments.py`**: An experimental script specifically designed to test how well the different algorithms scale and maintain performance as the target number of clusters (`k`) increases from 2 through 8.

## Setup Instructions

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Interactive Visualization (Frontend)
Launch the Streamlit app to interactively run the clustering algorithms:
```bash
streamlit run app.py
```

### 2. Systematic Hyperparameter Sweeps
Run the primary experiments script to benchmark different parameter configurations:
```bash
python experiments.py
```
*Results, including `detailed_results.csv`, `summary_results.csv`, and boxplot visualizations, will be saved to the `results/` directory.*

### 3. Target Cluster (k) Variability Testing
Evaluate algorithm stability across different values of `k` (2 through 8):
```bash
python k_experiments.py
```
*Results and line plots will be saved to the `results/` directory.*
