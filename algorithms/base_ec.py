import numpy as np

class ParentSelection:
    @staticmethod
    def roulette_wheel(population, fitnesses):
        # We want to MINIMIZE fitness (SSE). Thus, lower fitness -> higher probability.
        # We transform fitness to a maximization context: f_max - f_i + epsilon
        max_fit = np.max(fitnesses)
        inverted_fitness = max_fit - fitnesses + 1e-6
        probs = inverted_fitness / np.sum(inverted_fitness)
        idx = np.random.choice(len(population), p=probs)
        return population[idx]

    @staticmethod
    def tournament(population, fitnesses, tournament_size=3):
        indices = np.random.choice(len(population), size=tournament_size, replace=False)
        best_idx = indices[np.argmin(fitnesses[indices])]
        return population[best_idx]

    @staticmethod
    def rank_selection(population, fitnesses):
        # Sort indices by fitness ascending (lower is better)
        ranks = np.argsort(np.argsort(fitnesses)) 
        # Rank 0 is best, Rank N-1 is worst.
        N = len(population)
        weights = N - ranks # best gets weight N, worst gets weight 1
        probs = weights / np.sum(weights)
        idx = np.random.choice(len(population), p=probs)
        return population[idx]

class VariationOperators:
    @staticmethod
    def blend_crossover(parent1, parent2, alpha=0.5):
        """BLX-alpha crossover for continuous representations."""
        gamma = (1 + 2 * alpha) * np.random.rand(len(parent1)) - alpha
        child1 = (1 - gamma) * parent1 + gamma * parent2
        child2 = gamma * parent1 + (1 - gamma) * parent2
        return child1, child2

    @staticmethod
    def single_point_crossover(parent1, parent2):
        pt = np.random.randint(1, len(parent1)-1) if len(parent1) > 2 else 1
        child1 = np.concatenate((parent1[:pt], parent2[pt:]))
        child2 = np.concatenate((parent2[:pt], parent1[pt:]))
        return child1, child2

    @staticmethod
    def gaussian_mutation(individual, bounds_range, mutation_rate=0.1, sigma=0.1):
        """
        sigma is a fraction of the domain range.
        bounds_range is expected to be array of shape (n_features*n_clusters, ) or similar representing spans.
        """
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.rand() < mutation_rate:
                std_dev = sigma * bounds_range[i]
                mutated[i] += np.random.normal(0, std_dev)
        return mutated

    @staticmethod
    def random_resetting(individual, bounds, mutation_rate=0.1):
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.rand() < mutation_rate:
                # bounds is shape (dim, 2)
                mutated[i] = np.random.uniform(bounds[i, 0], bounds[i, 1])
        return mutated

class SurvivorSelection:
    @staticmethod
    def elitism(population, fitnesses, new_population, new_fitnesses, num_elites=1):
        """Replaces worst in new_population with best from population"""
        best_indices = np.argsort(fitnesses)[:num_elites]
        worst_indices = np.argsort(new_fitnesses)[-num_elites:]
        
        for i in range(num_elites):
            new_population[worst_indices[i]] = population[best_indices[i]]
            new_fitnesses[worst_indices[i]] = fitnesses[best_indices[i]]
            
        return new_population, new_fitnesses

    @staticmethod
    def generational(new_population, new_fitnesses):
        # Full replacement
        return new_population, new_fitnesses
