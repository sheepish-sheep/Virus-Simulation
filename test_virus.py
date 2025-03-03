from virus_simulation.virus import VirusGenome
import numpy as np

# Create a virus with a genome of size 5
virus = VirusGenome(genome_size=5)
print("Initial virus genome:")
print(virus.genome)

# Create a simple environment (also size 5)
environment = np.array([0.2, 0.4, 0.1, 0.7, 0.3])

# Calculate fitness
fitness = virus.get_fitness(environment)
print(f"Fitness in environment: {fitness:.4f}")

# Mutate the virus
print("\nMutating virus...")
virus.mutate(mutation_rate=0.3, mutation_strength=0.2)
print("Virus genome after mutation:")
print(virus.genome)

# Calculate new fitness
new_fitness = virus.get_fitness(environment)
print(f"New fitness in environment: {new_fitness:.4f}")
print(f"Fitness change: {new_fitness - fitness:.4f}")
