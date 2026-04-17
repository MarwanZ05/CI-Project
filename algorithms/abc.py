import numpy as np
from problem_formulation import ClusteringProblem

class ABC:
    """Artificial Bee Colony Algorithm"""
    def __init__(self, data, n_clusters, pop_size=50, max_iter=100, 
                 limit=20):
        self.problem = ClusteringProblem(data, n_clusters)
        self.pop_size = pop_size // 2  # Number of food sources (SN) is half of colony size
        self.max_iter = max_iter
        self.limit = limit
        
        self.flat_bounds = np.tile(self.problem.bounds, (self.problem.n_clusters, 1))

    def _calculate_fitness(self, sse):
        # In ABC, fitness needs to be strictly positive and to be maximized
        if sse >= 0:
            return 1.0 / (1.0 + sse)
        else:
            return 1.0 + abs(sse)

    def run(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        # Initialization
        foods = np.random.uniform(self.flat_bounds[:, 0], self.flat_bounds[:, 1], 
                                  size=(self.pop_size, len(self.flat_bounds)))
        
        sses = np.array([self.problem.evaluate_fitness(food) for food in foods])
        fitnesses = np.array([self._calculate_fitness(sse) for sse in sses])
        trials = np.zeros(self.pop_size)
        
        gbest_idx = np.argmin(sses)
        gbest_position = foods[gbest_idx].copy()
        gbest_sse = sses[gbest_idx]
        
        best_fitness_history = [gbest_sse]
        
        for iteration in range(self.max_iter):
            # Employed Bees Phase
            for i in range(self.pop_size):
                k = np.random.choice([idx for idx in range(self.pop_size) if idx != i])
                j = np.random.randint(0, len(self.flat_bounds))
                phi = np.random.uniform(-1, 1)
                
                v = foods[i].copy()
                v[j] = foods[i][j] + phi * (foods[i][j] - foods[k][j])
                v = self.problem.enforce_bounds(v)
                
                v_sse = self.problem.evaluate_fitness(v)
                v_fit = self._calculate_fitness(v_sse)
                
                if v_fit > fitnesses[i]: # Better fitness
                    foods[i] = v
                    fitnesses[i] = v_fit
                    sses[i] = v_sse
                    trials[i] = 0
                else:
                    trials[i] += 1
                    
            # Onlooker Bees Phase
            probs = fitnesses / np.sum(fitnesses)
            t = 0
            i = 0
            while t < self.pop_size:
                if np.random.rand() < probs[i]:
                    t += 1
                    k = np.random.choice([idx for idx in range(self.pop_size) if idx != i])
                    j = np.random.randint(0, len(self.flat_bounds))
                    phi = np.random.uniform(-1, 1)
                    
                    v = foods[i].copy()
                    v[j] = foods[i][j] + phi * (foods[i][j] - foods[k][j])
                    v = self.problem.enforce_bounds(v)
                    
                    v_sse = self.problem.evaluate_fitness(v)
                    v_fit = self._calculate_fitness(v_sse)
                    
                    if v_fit > fitnesses[i]:
                        foods[i] = v
                        fitnesses[i] = v_fit
                        sses[i] = v_sse
                        trials[i] = 0
                    else:
                        trials[i] += 1
                i = (i + 1) % self.pop_size
                
            # Memorize Best Source
            current_best_idx = np.argmin(sses)
            if sses[current_best_idx] < gbest_sse:
                gbest_sse = sses[current_best_idx]
                gbest_position = foods[current_best_idx].copy()
                
            # Scout Bees Phase
            for i in range(self.pop_size):
                if trials[i] > self.limit:
                    foods[i] = np.random.uniform(self.flat_bounds[:, 0], self.flat_bounds[:, 1])
                    sses[i] = self.problem.evaluate_fitness(foods[i])
                    fitnesses[i] = self._calculate_fitness(sses[i])
                    trials[i] = 0
                    
            best_fitness_history.append(gbest_sse)
            
        silhouette = self.problem.silhouette(gbest_position)
        labels = self.problem.get_labels(gbest_position)
        
        return {
            'centroids': gbest_position.reshape(self.problem.n_clusters, self.problem.n_features),
            'sse': gbest_sse,
            'silhouette': silhouette,
            'labels': labels,
            'history': best_fitness_history
        }
