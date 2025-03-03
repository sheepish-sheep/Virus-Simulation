import numpy as np
from .virus import VirusGenome
from .population import VirusPopulation
from .environment import Environment

class VirusRLEnvironment:
    """
    Reinforcement Learning environment for virus evolution simulation.
    This environment allows an RL agent to learn optimal drug administration strategies.
    """
    def __init__(self, pop_size=100, genome_size=3, episode_length=50):
        """
        Initialize the RL environment.
        
        Args:
            pop_size: Number of viruses in the population
            genome_size: Size of the virus genome
            episode_length: Number of steps (generations) in each episode
        """
        self.pop_size = pop_size
        self.genome_size = genome_size
        self.episode_length = episode_length
        self.current_step = 0
        # Define a drug vector that targets bright viruses
        self.drug_vector = np.array([0.1] * genome_size)  # Low values target bright viruses
        self.reset()
        
    def reset(self):
        """
        Reset the environment for a new episode.
        
        Returns:
            Initial state observation (average fitness, drug active status, generation number)
        """
        # Create new virus population
        self.population = VirusPopulation(self.pop_size, self.genome_size)
        
        # Create environment with random factors
        self.environment = Environment(self.genome_size)
        
        # Reset step counter
        self.current_step = 0
        
        # Drug is initially inactive
        self.drug_active = False
        
        # Evaluate initial fitness
        fitness_scores = self.population.evaluate_fitness(self.environment.get_factors())
        self.avg_fitness = sum(fitness_scores) / len(fitness_scores)
        
        # Return initial state
        return self._get_state()
    
    def step(self, action):
        """
        Take a step in the environment based on the agent's action.
        
        Args:
            action: 0 = do nothing, 1 = toggle drug (apply if not active, remove if active)
        
        Returns:
            next_state: New state after action
            reward: Reward for the action
            done: Whether the episode is complete
            info: Additional information
        """
        # Process action (toggle drug if action is 1)
        if action == 1:
            self.drug_active = not self.drug_active
            if self.drug_active:
                self.environment.apply_drug_treatment(self.drug_vector)
            else:
                # When removing drug, reset to random environment
                # Initialize with random values between 0 and 1
                default_env = np.random.random(self.genome_size)
                # Normalize to sum to 1
                default_env = default_env / np.sum(default_env)
                # Apply this as a 100% strength treatment to reset
                self.environment.apply_drug_treatment(default_env, strength=1.0)
        
        # Evolve virus population for one generation
        current_fitness = self.population.evaluate_fitness(self.environment.get_factors())
        avg_fitness_before = sum(current_fitness) / len(current_fitness)
        
        # Evolve population
        self.population.evolve_generation(self.environment.get_factors())
        self.current_step += 1
        
        # Evaluate new fitness
        new_fitness = self.population.evaluate_fitness(self.environment.get_factors())
        self.avg_fitness = sum(new_fitness) / len(new_fitness)
        
        # Calculate reward (negative fitness is the reward - we want to minimize virus fitness)
        reward = -self.avg_fitness
        
        # Add penalty for toggling drug too frequently (to encourage strategic usage)
        if action == 1:
            reward -= 0.1  # Small penalty for using the drug
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        
        # Return SARSA tuple
        return self._get_state(), reward, done, {'fitness_change': self.avg_fitness - avg_fitness_before}
    
    def _get_state(self):
        """
        Get the current state observation for the RL agent.
        
        Returns:
            Tuple of (normalized average fitness, drug active status, normalized step number)
        """
        # Create state vector
        # State includes: average fitness, drug active status, generation number
        normalized_fitness = self.avg_fitness / 10.0  # Normalize fitness 
        drug_status = 1.0 if self.drug_active else 0.0
        progress = self.current_step / self.episode_length
        
        return np.array([normalized_fitness, drug_status, progress])
