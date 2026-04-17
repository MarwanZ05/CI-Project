import numpy as np
from problem_formulation import ClusteringProblem

class ES:
    """Evolution Strategy: (mu + lambda)-ES"""
    def __init__(self, data, n_clusters, mu=20, lambda_=100, max_iter=100, mutation_step=0.1):
        self.problem = ClusteringProblem(data, n_clusters)
        self.mu = mu
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.mutation_step = mutation_step
        
        self.problem_bounds = np.tile(self.problem.bounds, (self.problem.n_clusters, 1))
        self.range_span = self.problem_bounds[:, 1] - self.problem_bounds[:, 0]

    def _mutate(self, individual):
        # Global Gaussian mutation
        mutated = individual.copy()
        for i in range(len(mutated)):
            std_dev = self.mutation_step * self.range_span[i]
            mutated[i] += np.random.normal(0, std_dev)
        return self.problem.enforce_bounds(mutated)

    def _recombine(self, parent1, parent2):
        # Discrete recombination
        child = np.zeros_like(parent1)
        for i in range(len(parent1)):
            if np.random.rand() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child

    def run(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize population (mu)
        population = np.random.uniform(self.problem_bounds[:, 0], self.problem_bounds[:, 1], 
                                       size=(self.mu, len(self.problem_bounds)))
        
        fitnesses = np.array([self.problem.evaluate_fitness(ind) for ind in population])
        
        best_idx = np.argmin(fitnesses)
        gbest_position = population[best_idx].copy()
        gbest_fitness = fitnesses[best_idx]
        
        best_fitness_history = [gbest_fitness]
        
        for iteration in range(self.max_iter):
            offspring = []
            
            # Generate lambda offspring
            for _ in range(self.lambda_):
                p1_idx, p2_idx = np.random.choice(self.mu, size=2, replace=False)
                child = self._recombine(population[p1_idx], population[p2_idx])
                child = self._mutate(child)
                offspring.append(child)
                
            offspring = np.array(offspring)
            offspring_fitnesses = np.array([self.problem.evaluate_fitness(ind) for ind in offspring])
            
            # (mu + lambda) survivor selection
            pool = np.vstack((population, offspring))
            pool_fitnesses = np.concatenate((fitnesses, offspring_fitnesses))
            
            # Select top mu
            best_indices = np.argsort(pool_fitnesses)[:self.mu]
            population = pool[best_indices]
            fitnesses = pool_fitnesses[best_indices]
            
            if fitnesses[0] < gbest_fitness:
                gbest_fitness = fitnesses[0]
                gbest_position = population[0].copy()
                
            best_fitness_history.append(gbest_fitness)
            
        silhouette = self.problem.silhouette(gbest_position)
        labels = self.problem.get_labels(gbest_position)
        
        return {
            'centroids': gbest_position.reshape(self.problem.n_clusters, self.problem.n_features),
            'sse': gbest_fitness,
            'silhouette': silhouette,
            'labels': labels,
            'history': best_fitness_history
        }
