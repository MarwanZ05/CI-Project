import numpy as np
from problem_formulation import ClusteringProblem

class PSO:
    def __init__(self, data, n_clusters, pop_size=50, max_iter=100, 
                 w=0.729, c1=1.49445, c2=1.49445):
        self.problem = ClusteringProblem(data, n_clusters)
        self.num_particles = pop_size
        self.max_iter = max_iter
        
        self.w = w     # Inertia
        self.c1 = c1   # Cognitive
        self.c2 = c2   # Social
        
        self.feature_bounds = self.problem.bounds
        self.flat_bounds = np.tile(self.feature_bounds, (self.problem.n_clusters, 1))

    def run(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize particles and velocities
        particles = np.random.uniform(self.flat_bounds[:, 0], self.flat_bounds[:, 1], 
                                      size=(self.num_particles, len(self.flat_bounds)))
        velocities = np.zeros_like(particles)
        
        pbest_positions = particles.copy()
        pbest_fitnesses = np.array([self.problem.evaluate_fitness(p) for p in particles])
        
        gbest_idx = np.argmin(pbest_fitnesses)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_fitness = pbest_fitnesses[gbest_idx]
        
        best_fitness_history = [gbest_fitness]
        
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                # Update velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] + 
                                 self.c1 * r1 * (pbest_positions[i] - particles[i]) + 
                                 self.c2 * r2 * (gbest_position - particles[i]))
                                 
                # Update position
                particles[i] = particles[i] + velocities[i]
                
                # Enforce bounds
                particles[i] = self.problem.enforce_bounds(particles[i])
                
                # Evaluate
                fitness = self.problem.evaluate_fitness(particles[i])
                
                # Update pbest
                if fitness < pbest_fitnesses[i]:
                    pbest_fitnesses[i] = fitness
                    pbest_positions[i] = particles[i].copy()
                    
                    # Update gbest
                    if fitness < gbest_fitness:
                        gbest_fitness = fitness
                        gbest_position = particles[i].copy()
                        
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
