import numpy as np

class Environment:
    """Simulates the environment in which viruses evolve"""
    
    def __init__(self, factors_size=10, change_rate=0.05):
        """
        Initialize the environment.
        
        Args:
            factors_size: Size of the environment factors vector
            change_rate: Rate at which environment changes each generation
        """
        self.factors_size = factors_size
        self.change_rate = change_rate
        # Initialize with random values between 0 and 1
        self.factors = np.random.random(factors_size)
        # Normalize to sum to 1
        self.factors = self.factors / np.sum(self.factors)
        
    def get_factors(self):
        """
        Get the current environment factors.
        
        Returns:
            Array of environment factors
        """
        return self.factors
    
    def update(self):
        """
        Update the environment for the next generation.
        
        This simulates changing conditions such as host immune response,
        drug treatments, or other selective pressures.
        """
        # Random drift in environment
        perturbation = np.random.normal(0, self.change_rate, self.factors_size)
        self.factors += perturbation
        
        # Ensure values stay within valid range
        self.factors = np.clip(self.factors, 0, 1)
        # Renormalize
        self.factors = self.factors / np.sum(self.factors)
        
    def apply_drug_treatment(self, drug_vector, strength=0.5):
        """
        Apply a drug treatment to the environment.
        
        Args:
            drug_vector: Vector representing drug effects on different factors
            strength: Strength of the drug effect (0-1)
        """
        # Drug alters the environment to be hostile to certain virus characteristics
        self.factors = (1 - strength) * self.factors + strength * drug_vector
        # Renormalize
        self.factors = self.factors / np.sum(self.factors)
    
    def __str__(self):
        return f"Environment Factors: {self.factors}"
