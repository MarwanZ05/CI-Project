import numpy as np
from problem_formulation import ClusteringProblem

class DE:
    """Differential Evolution (DE/rand/1/bin)"""
    def __init__(self, data, n_clusters, pop_size=50, max_iter=100, 
                 F=0.8, CR=0.9):
        self.problem = ClusteringProblem(data, n_clusters)
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        self.F = F     # Mutation factor
        self.CR = CR   # Crossover rate
        
        self.flat_bounds = np.tile(self.problem.bounds, (self.problem.n_clusters, 1))

    def run(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        population = np.random.uniform(self.flat_bounds[:, 0], self.flat_bounds[:, 1], 
                                      size=(self.pop_size, len(self.flat_bounds)))
        fitnesses = np.array([self.problem.evaluate_fitness(ind) for ind in population])
        
        best_idx = np.argmin(fitnesses)
        gbest_position = population[best_idx].copy()
        gbest_fitness = fitnesses[best_idx]
        
        best_fitness_history = [gbest_fitness]
        
        for iteration in range(self.max_iter):
            for i in range(self.pop_size):
                # Mutation (Select 3 distinct parents != i)
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                idx1, idx2, idx3 = np.random.choice(candidates, size=3, replace=False)
                
                # Mutant Vector: target + F * (diff)
                mutant = population[idx1] + self.F * (population[idx2] - population[idx3])
                
                # Crossover (Binomial)
                trial = np.copy(population[i])
                j_rand = np.random.randint(0, len(self.flat_bounds)) # Ensure at least one component comes from mutant
                
                for j in range(len(self.flat_bounds)):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                        
                trial = self.problem.enforce_bounds(trial)
                
                # Selection
                trial_fit = self.problem.evaluate_fitness(trial)
                if trial_fit <= fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = trial_fit
                    
                    if trial_fit < gbest_fitness:
                        gbest_fitness = trial_fit
                        gbest_position = trial.copy()
                        
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
