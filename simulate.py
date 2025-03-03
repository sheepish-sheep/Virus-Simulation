import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from virus_simulation.virus import VirusGenome
from virus_simulation.population import VirusPopulation
from virus_simulation.environment import Environment
from virus_simulation.visualization import plot_fitness_over_generations, plot_virus_genome_heatmap, save_all_plots

def parse_args():
    parser = argparse.ArgumentParser(description='Virus Evolution Simulation')
    parser.add_argument('--pop_size', type=int, default=100, 
                        help='Population size')
    parser.add_argument('--genome_size', type=int, default=10, 
                        help='Size of virus genome')
    parser.add_argument('--generations', type=int, default=50, 
                        help='Number of generations to simulate')
    parser.add_argument('--mutation_rate', type=float, default=0.1, 
                        help='Mutation rate')
    parser.add_argument('--env_change_rate', type=float, default=0.05, 
                        help='Rate at which environment changes')
    parser.add_argument('--drug_treatment_gen', type=int, default=None, 
                        help='Generation to apply drug treatment (default: no treatment)')
    parser.add_argument('--output', type=str, default='virus_sim', 
                        help='Base filename for output files')
    parser.add_argument('--seed', type=int, default=None, 
                        help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    print(f"Starting virus simulation with population size {args.pop_size}")
    print(f"Running for {args.generations} generations")
    
    # Initialize environment and population
    environment = Environment(factors_size=args.genome_size, 
                              change_rate=args.env_change_rate)
    population = VirusPopulation(pop_size=args.pop_size, 
                                genome_size=args.genome_size)
    
    # Track data for visualization
    fitness_history = []
    environment_history = []
    
    # Run simulation
    for gen in range(args.generations):
        # Store environment state
        environment_history.append(environment.get_factors().copy())
        
        # Apply drug treatment if specified
        if args.drug_treatment_gen is not None and gen == args.drug_treatment_gen:
            print(f"Generation {gen}: Applying drug treatment")
            # Create a drug that targets specific virus characteristics
            drug_vector = np.zeros(args.genome_size)
            drug_vector[0:3] = 1.0  # Target the first 3 genes
            drug_vector = drug_vector / np.sum(drug_vector)
            environment.apply_drug_treatment(drug_vector, strength=0.7)
        
        # Evolve one generation
        avg_fitness = population.evolve_generation(
            environment.get_factors(),
            mutation_rate=args.mutation_rate
        )
        
        fitness_history.append(avg_fitness)
        
        # Update environment for next generation
        environment.update()
        
        # Print progress
        if gen % 5 == 0 or gen == args.generations - 1:
            print(f"Generation {gen}: Average fitness = {avg_fitness:.4f}")
    
    # Generate and save visualizations
    save_all_plots(
        fitness_history,
        population.population,
        environment_history,
        base_filename=args.output
    )
    
    # Display results
    plot_fitness_over_generations(fitness_history)
    plt.show()
    
    plot_virus_genome_heatmap(population.population)
    plt.show()
    
    print("Simulation complete!")

if __name__ == "__main__":
    main()
