import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_fitness_over_generations(fitness_history, title="Virus Evolution"):
    """
    Plot the average fitness over generations.
    
    Args:
        fitness_history: List of average fitness values per generation
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, 'b-', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add trend line
    if len(fitness_history) > 1:
        z = np.polyfit(range(len(fitness_history)), fitness_history, 1)
        p = np.poly1d(z)
        plt.plot(range(len(fitness_history)), p(range(len(fitness_history))), 
                 "r--", alpha=0.8, linewidth=1)
    
    plt.tight_layout()
    return plt

def plot_virus_genome_heatmap(population, sample_size=10, title="Virus Genome Comparison"):
    """
    Create a heatmap visualization of virus genomes.
    
    Args:
        population: List of VirusGenome objects
        sample_size: Number of viruses to sample for visualization
        title: Title for the plot
    """
    # Sample viruses
    if len(population) > sample_size:
        sample_indices = np.random.choice(len(population), sample_size, replace=False)
        sampled_viruses = [population[i] for i in sample_indices]
    else:
        sampled_viruses = population
    
    # Create data matrix
    data = np.array([virus.genome for virus in sampled_viruses])
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(data, cmap="viridis", annot=False, 
                     xticklabels=[f"Gene {i+1}" for i in range(data.shape[1])],
                     yticklabels=[f"Virus {i+1}" for i in range(data.shape[0])])
    plt.title(title)
    plt.tight_layout()
    return plt

def plot_environmental_factors(environment_history, title="Environmental Changes"):
    """
    Plot how environmental factors change over time.
    
    Args:
        environment_history: List of environment factors over generations
        title: Title for the plot
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(environment_history)
    
    plt.figure(figsize=(12, 6))
    plt.stackplot(range(len(environment_history)), 
                 [df[col] for col in df.columns],
                 labels=[f"Factor {i+1}" for i in range(df.shape[1])],
                 alpha=0.7)
    
    plt.xlabel('Generation')
    plt.ylabel('Factor Weight')
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    return plt

def save_all_plots(fitness_history, population, environment_history, base_filename="virus_simulation"):
    """
    Save all plots to files.
    
    Args:
        fitness_history: List of average fitness values per generation
        population: List of VirusGenome objects
        environment_history: List of environment factors over generations
        base_filename: Base name for the output files
    """
    # Fitness plot
    plt1 = plot_fitness_over_generations(fitness_history)
    plt1.savefig(f"{base_filename}_fitness.png")
    plt1.close()
    
    # Genome heatmap
    plt2 = plot_virus_genome_heatmap(population)
    plt2.savefig(f"{base_filename}_genomes.png")
    plt2.close()
    
    # Environment plot
    plt3 = plot_environmental_factors(environment_history)
    plt3.savefig(f"{base_filename}_environment.png")
    plt3.close()
    
    print(f"All plots saved with base filename: {base_filename}")
