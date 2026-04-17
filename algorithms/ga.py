import numpy as np
from problem_formulation import ClusteringProblem
from .base_ec import ParentSelection, VariationOperators, SurvivorSelection

class GA:
    def __init__(self, data, n_clusters, pop_size=50, max_iter=100, 
                 crossover_rate=0.8, mutation_rate=0.1, 
                 selection_method='tournament',
                 crossover_method='single_point',
                 survivor_method='elitism'):
        self.problem = ClusteringProblem(data, n_clusters)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.survivor_method = survivor_method
        
        # Determine lengths
        self.feature_bounds = self.problem.bounds # shape (n_features, 2)
        self.dim = self.problem.n_clusters * self.problem.n_features
        # create flattened bounds array
        self.flat_bounds = np.tile(self.feature_bounds, (self.problem.n_clusters, 1))
        self.bounds_range = self.flat_bounds[:, 1] - self.flat_bounds[:, 0]

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            # random init within bounds
            ind = np.random.uniform(self.flat_bounds[:, 0], self.flat_bounds[:, 1])
            population.append(ind)
        return np.array(population)

    def run(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        population = self.initialize_population()
        fitnesses = np.array([self.problem.evaluate_fitness(ind) for ind in population])
        
        best_fitness_history = []
        best_solution = None
        global_best_fit = float('inf')
        
        # Pre-bind methods
        select_parent = getattr(ParentSelection, self.selection_method)
        crossover = getattr(VariationOperators, self.crossover_method + '_crossover')
        
        for iteration in range(self.max_iter):
            new_population = []
            
            while len(new_population) < self.pop_size:
                p1 = select_parent(population, fitnesses)
                p2 = select_parent(population, fitnesses)
                
                if np.random.rand() < self.crossover_rate:
                    c1, c2 = crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                    
                c1 = VariationOperators.gaussian_mutation(c1, self.bounds_range, self.mutation_rate)
                c2 = VariationOperators.gaussian_mutation(c2, self.bounds_range, self.mutation_rate)
                
                # Enforce boundaries
                c1 = self.problem.enforce_bounds(c1)
                c2 = self.problem.enforce_bounds(c2)
                
                new_population.extend([c1, c2])
                
            new_population = np.array(new_population[:self.pop_size])
            new_fitnesses = np.array([self.problem.evaluate_fitness(ind) for ind in new_population])
            
            if self.survivor_method == 'elitism':
                population, fitnesses = SurvivorSelection.elitism(population, fitnesses, new_population, new_fitnesses)
            else:
                population, fitnesses = new_population, new_fitnesses
                
            current_best_idx = np.argmin(fitnesses)
            current_best_fit = fitnesses[current_best_idx]
            
            if current_best_fit < global_best_fit:
                global_best_fit = current_best_fit
                best_solution = population[current_best_idx].copy()
                
            best_fitness_history.append(global_best_fit)
            
        # Final evaluation
        silhouette = self.problem.silhouette(best_solution)
        labels = self.problem.get_labels(best_solution)
        
        return {
            'centroids': best_solution.reshape(self.problem.n_clusters, self.problem.n_features),
            'sse': global_best_fit,
            'silhouette': silhouette,
            'labels': labels,
            'history': best_fitness_history
        }
