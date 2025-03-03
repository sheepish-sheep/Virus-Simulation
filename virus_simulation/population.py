import random
import numpy as np
from typing import List
from .virus import VirusGenome

class VirusPopulation:
    """Manage a population of viruses with evolutionary algorithms"""
    
    def __init__(self, pop_size=100, genome_size=10):
        """
        Initialize the virus population.
        
        Args:
            pop_size: Number of viruses in the population
            genome_size: Size of the virus genome vector
        """
        self.pop_size = pop_size
        self.genome_size = genome_size
        self.population = [VirusGenome(genome_size) for _ in range(pop_size)]
        self.generation = 0
        self.fitness_history = []
        
    def evaluate_fitness(self, environment_factors):
        """
        Evaluate fitness for all viruses in the population.
        
        Args:
            environment_factors: Vector representing environmental conditions
            
        Returns:
            List of fitness scores
        """
        return [virus.get_fitness(environment_factors) for virus in self.population]
    
    def select_parents(self, fitness_scores, num_parents):
        """
        Select parents for reproduction based on fitness scores.
        
        Args:
            fitness_scores: List of fitness scores
            num_parents: Number of parents to select
            
        Returns:
            List of selected parent indices
        """
        # Convert to numpy array
        fitness_array = np.array(fitness_scores)
        
        # Tournament selection
        selected_indices = []
        for _ in range(num_parents):
            # Select random individuals for tournament
            tournament_size = 3
            competitors = random.sample(range(self.pop_size), tournament_size)
            # Select the best one
            winner = max(competitors, key=lambda idx: fitness_array[idx])
            selected_indices.append(winner)
            
        return selected_indices
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to create offspring.
        
        Args:
            parent1, parent2: Two VirusGenome objects
            
        Returns:
            New VirusGenome object
        """
        # One-point crossover
        child = VirusGenome(self.genome_size)
        crossover_point = random.randint(1, self.genome_size - 1)
        child.genome = np.concatenate([
            parent1.genome[:crossover_point],
            parent2.genome[crossover_point:]
        ])
        
        # Add completely random genetic material occasionally
        if random.random() < 0.15:  # 15% chance for random genetic material
            # Decide how many genes to randomize (1-3 genes)
            num_random_genes = random.randint(1, min(3, self.genome_size))
            # Choose which positions to randomize
            positions = random.sample(range(self.genome_size), num_random_genes)
            # Set those positions to completely random values
            for pos in positions:
                child.genome[pos] = random.random()
        
        # Very rarely (1% chance), introduce a completely novel genetic sequence
        if random.random() < 0.01:
            # Create an entirely new segment (10-30% of the genome)
            segment_size = max(1, int(self.genome_size * random.uniform(0.1, 0.3)))
            start_pos = random.randint(0, self.genome_size - segment_size)
            # Fill with random values
            child.genome[start_pos:start_pos+segment_size] = np.random.random(segment_size)
        
        return child

    def evolve_generation(self, environment_factors, mutation_rate=0.1, elitism=2):
        """
        Evolve the population by one generation.
        
        Args:
            environment_factors: Vector representing environmental conditions
            mutation_rate: Probability of mutation for each gene
            elitism: Number of best individuals to preserve unchanged
            
        Returns:
            Average fitness of the new generation
        """
        # Evaluate fitness
        fitness_scores = self.evaluate_fitness(environment_factors)
        
        # Store statistics
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        self.fitness_history.append(avg_fitness)
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order
        
        # Create new population
        new_population = []
        
        # Elitism: keep the best individuals
        for i in range(elitism):
            new_population.append(self.population[sorted_indices[i]])
        
        # Fill the rest of the population through selection, crossover, and mutation
        while len(new_population) < self.pop_size:
            # Select parents
            parent_indices = self.select_parents(fitness_scores, 2)
            parent1 = self.population[parent_indices[0]]
            parent2 = self.population[parent_indices[1]]
            
            # Create offspring through crossover
            child = self.crossover(parent1, parent2)
            
            # Mutate
            child.mutate(mutation_rate)
            
            # Add to new population
            new_population.append(child)
        
        # Replace old population
        self.population = new_population
        self.generation += 1
        
        return avg_fitness
