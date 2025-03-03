import numpy as np

class VirusGenome:
    """
    Represents a virus genome using a simple vector representation.
    """
    def __init__(self, genome_size=10):
        # Initialize with random values between 0 and 1
        self.genome = np.random.random(genome_size)
        
    def mutate(self, mutation_rate=0.1, mutation_strength=0.2):
        """
        Mutate the genome with given rate and strength.
        
        Args:
            mutation_rate: Probability of each gene mutating
            mutation_strength: Standard deviation of the mutation
        """
        # Create a mask for which genes will mutate
        mask = np.random.random(self.genome.shape) < mutation_rate
        
        # Generate mutations
        mutations = np.random.normal(0, mutation_strength, self.genome.shape)
        
        # Apply mutations only to selected genes
        self.genome[mask] += mutations[mask]
        
        # Clip values to stay in valid range (0-1)
        self.genome = np.clip(self.genome, 0, 1)
    
    def get_fitness(self, environment_factors):
        """
        Calculate fitness based on how well the genome is adapted to environment.
        
        Args:
            environment_factors: Vector representing environmental conditions
            
        Returns:
            Fitness score (higher is better)
        """
        # Simple fitness function: dot product with environment
        # This means viruses with genes matching the environment have higher fitness
        return np.dot(self.genome, environment_factors)
    
    def __str__(self):
        return f"Virus Genome: {self.genome}"