import pygame
import sys
import numpy as np
import random
import time
from virus_simulation.virus import VirusGenome
from virus_simulation.environment import Environment

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1000, 700
VIRUS_SIZE = 15
MAX_SPEED = 1.5
BACKGROUND_COLOR = (0, 0, 0)
ENV_DISPLAY_HEIGHT = 50
INFO_HEIGHT = 100
SIMULATION_HEIGHT = HEIGHT - ENV_DISPLAY_HEIGHT - INFO_HEIGHT

# Color mapping
def genome_to_color(genome):
    # Use first 3 genes to determine RGB values
    r = int(255 * genome[0])
    g = int(255 * genome[1])
    b = int(255 * genome[2])
    return (r, g, b)

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Virus Evolution Simulation")
font = pygame.font.SysFont('Arial', 16)
title_font = pygame.font.SysFont('Arial', 24, bold=True)

class VisualVirus:
    def __init__(self, virus, pos=None):
        self.virus = virus
        if pos is None:
            self.x = random.randint(VIRUS_SIZE, WIDTH - VIRUS_SIZE)
            self.y = random.randint(VIRUS_SIZE, SIMULATION_HEIGHT - VIRUS_SIZE)
        else:
            self.x, self.y = pos
        
        # Random movement
        self.vx = random.uniform(-MAX_SPEED, MAX_SPEED)
        self.vy = random.uniform(-MAX_SPEED, MAX_SPEED)
        
        # Calculate color based on genome
        self.update_color()
        
        # Size can be based on fitness
        self.size = VIRUS_SIZE
        self.dying = False
        self.alpha = 255  # For fade out effect when dying
        
    def update_color(self):
        self.color = genome_to_color(self.virus.genome)
    
    def update_position(self):
        if self.dying:
            # Fade out and shrink when dying
            self.alpha = max(0, self.alpha - 20)
            self.size = max(1, self.size - 1)
            return
            
        # Random changes to velocity
        self.vx += random.uniform(-0.2, 0.2)
        self.vy += random.uniform(-0.2, 0.2)
        
        # Limit speed
        speed = (self.vx**2 + self.vy**2)**0.5
        if speed > MAX_SPEED:
            self.vx = (self.vx / speed) * MAX_SPEED
            self.vy = (self.vy / speed) * MAX_SPEED
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Bounce off edges
        if self.x < VIRUS_SIZE or self.x > WIDTH - VIRUS_SIZE:
            self.vx = -self.vx
            self.x = max(VIRUS_SIZE, min(WIDTH - VIRUS_SIZE, self.x))
        
        if self.y < VIRUS_SIZE or self.y > SIMULATION_HEIGHT - VIRUS_SIZE:
            self.vy = -self.vy
            self.y = max(VIRUS_SIZE, min(SIMULATION_HEIGHT - VIRUS_SIZE, self.y))
    
    def draw(self, screen):
        if self.dying:
            # Create a surface with per-pixel alpha
            s = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
            color_with_alpha = (*self.color, self.alpha)
            pygame.draw.circle(s, color_with_alpha, (self.size, self.size), self.size)
            screen.blit(s, (int(self.x - self.size), int(self.y - self.size)))
        else:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)
            # Draw a small black outline
            pygame.draw.circle(screen, (0, 0, 0), (int(self.x), int(self.y)), self.size, 1)

def draw_environment(screen, environment, width, height, y_offset):
    # Draw environment bar
    bar_height = height
    
    for i, factor in enumerate(environment.factors):
        # Calculate width of this section
        section_width = int(width / len(environment.factors))
        x_pos = i * section_width
        
        # Calculate color intensity based on factor value
        intensity = int(255 * factor * len(environment.factors))
        # Ensure color values are within valid range (0-255)
        intensity = max(0, min(255, intensity))
        color = (intensity, intensity, intensity)
        
        # Draw rectangle
        pygame.draw.rect(screen, color, (x_pos, y_offset, section_width, bar_height))
        
        # Add factor number
        text = font.render(f"{i+1}", True, (255, 0, 0) if intensity < 128 else (0, 0, 0))
        screen.blit(text, (x_pos + section_width // 2 - 5, y_offset + bar_height // 2 - 8))

def draw_info(screen, generation, avg_fitness, apply_drug, viruses_killed, width, height, y_offset):
    # Clear info area
    pygame.draw.rect(screen, (200, 200, 200), (0, y_offset, width, height))
    
    # Draw generation and fitness info
    gen_text = title_font.render(f"Generation: {generation}", True, (0, 0, 0))
    fitness_text = title_font.render(f"Average Fitness: {avg_fitness:.4f}", True, (0, 0, 0))
    
    screen.blit(gen_text, (20, y_offset + 10))
    screen.blit(fitness_text, (20, y_offset + 50))
    
    # Draw drug info
    if apply_drug:
        drug_text = title_font.render("DRUG APPLIED", True, (255, 0, 0))
        screen.blit(drug_text, (width - 200, y_offset + 25))
    
    # Draw viruses killed info
    if viruses_killed > 0:
        killed_text = title_font.render(f"Viruses Killed: {viruses_killed}", True, (255, 0, 0))
        screen.blit(killed_text, (width - 300, y_offset + 50))

def run_visual_simulation():
    # Simulation parameters
    pop_size = 30
    genome_size = 8
    generations = 100
    mutation_rate = 0.1
    env_change_rate = 0.02
    drug_generation = 20
    
    # Initialize environment and population
    environment = Environment(factors_size=genome_size, change_rate=env_change_rate)
    
    # Initialize viruses
    viruses = [VirusGenome(genome_size) for _ in range(pop_size)]
    visual_viruses = [VisualVirus(v) for v in viruses]
    
    # Tracking variables
    generation = 0
    last_evolution_time = time.time()
    evolution_interval = 2.0  # seconds between generations
    apply_drug = False
    drug_applied = False
    viruses_killed = 0
    killing_threshold = 0.3  # Fitness threshold below which viruses die
    
    # Main loop
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Manually trigger evolution
                    last_evolution_time = time.time() - evolution_interval
                elif event.key == pygame.K_d:
                    # Manually apply drug
                    apply_drug = True
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    # Speed up evolution
                    evolution_interval = max(0.2, evolution_interval - 0.2)
                elif event.key == pygame.K_MINUS:
                    # Slow down evolution
                    evolution_interval = min(5.0, evolution_interval + 0.2)
        
        # Check if it's time for next generation
        current_time = time.time()
        if current_time - last_evolution_time >= evolution_interval and generation < generations:
            # Apply drug if it's time
            if generation == drug_generation and not drug_applied:
                apply_drug = True
                drug_applied = True
            
            if apply_drug:
                print(f"Generation {generation}: Applying drug treatment")
                # Create a drug that targets specific virus characteristics
                drug_vector = np.zeros(genome_size)
                drug_vector[0:3] = 0.0  # Target the first 3 genes - lower values are better against drug
                # Set other genes to higher values
                drug_vector[3:] = 1.0
                drug_vector = drug_vector / np.sum(drug_vector)
                environment.apply_drug_treatment(drug_vector, strength=0.7)
                apply_drug = False
            
            # Calculate fitness for each virus
            env_factors = environment.get_factors()
            fitness_scores = [v.get_fitness(env_factors) for v in viruses]
            
            # Check for viruses to kill if drug is applied
            if drug_applied:
                # Mark viruses for death if fitness is below threshold
                for i, (vv, fitness) in enumerate(zip(visual_viruses, fitness_scores)):
                    if fitness < killing_threshold and not vv.dying:
                        vv.dying = True
                        viruses_killed += 1
            
            # Filter out dead viruses
            alive_indices = [i for i, vv in enumerate(visual_viruses) if not vv.dying]
            
            if len(alive_indices) == 0:
                # If all viruses are killed, create some new random ones
                print("All viruses killed! Creating new population...")
                viruses = [VirusGenome(genome_size) for _ in range(pop_size)]
                visual_viruses = [VisualVirus(v) for v in viruses]
                # Reset drug application to give new viruses a chance
                drug_applied = False
                environment.update()  # Reset environment
                generation += 1
                last_evolution_time = current_time
                continue
            
            # Calculate average fitness of alive viruses
            alive_fitness = [fitness_scores[i] for i in alive_indices]
            avg_fitness = sum(alive_fitness) / len(alive_fitness)
            
            # Normalize fitness for visualization (size)
            max_fitness = max(alive_fitness)
            min_fitness = min(alive_fitness)
            fitness_range = max_fitness - min_fitness if max_fitness > min_fitness else 1
            
            for i in alive_indices:
                # Update size based on fitness
                normalized_fitness = (fitness_scores[i] - min_fitness) / fitness_range
                visual_viruses[i].size = int(VIRUS_SIZE * (0.5 + 0.5 * normalized_fitness))
            
            # Selection: tournament selection (only from alive viruses)
            alive_viruses = [viruses[i] for i in alive_indices]
            alive_visual_viruses = [visual_viruses[i] for i in alive_indices]
            alive_fitness_scores = [fitness_scores[i] for i in alive_indices]
            
            new_viruses = []
            
            # Elitism: keep top 2 viruses if we have enough
            if len(alive_viruses) >= 2:
                elite_indices = np.argsort(alive_fitness_scores)[-2:]
                for idx in elite_indices:
                    new_virus = VirusGenome(genome_size)
                    new_virus.genome = alive_viruses[idx].genome.copy()
                    new_viruses.append(new_virus)
            
            # Fill rest with offspring
            while len(new_viruses) < pop_size:
                # Tournament selection
                tournament_size = min(3, len(alive_viruses))
                parent_indices = []
                
                for _ in range(2):  # Select 2 parents
                    if len(alive_viruses) <= 1:
                        # If only one virus is alive, clone it
                        parent_indices.append(0)
                    else:
                        competitors = random.sample(range(len(alive_viruses)), tournament_size)
                        winner = max(competitors, key=lambda idx: alive_fitness_scores[idx])
                        parent_indices.append(winner)
                
                # Crossover
                parent1 = alive_viruses[parent_indices[0]]
                parent2 = alive_viruses[parent_indices[1]]
                
                child = VirusGenome(genome_size)
                crossover_point = random.randint(1, genome_size - 1)
                child.genome = np.concatenate([
                    parent1.genome[:crossover_point],
                    parent2.genome[crossover_point:]
                ])
                
                # Mutation
                child.mutate(mutation_rate=mutation_rate)
                
                new_viruses.append(child)
            
            # Create new positions, avoiding positions of dying viruses
            old_positions = []
            for vv in alive_visual_viruses:
                old_positions.append((vv.x, vv.y))
            
            # Fill remaining positions with random locations
            while len(old_positions) < pop_size:
                old_positions.append((random.randint(VIRUS_SIZE, WIDTH - VIRUS_SIZE),
                                      random.randint(VIRUS_SIZE, SIMULATION_HEIGHT - VIRUS_SIZE)))
            
            # Update viruses for next generation
            viruses = new_viruses
            
            # Create new visual viruses while keeping dying ones for animation
            new_visual_viruses = []
            # Keep dying viruses
            for vv in visual_viruses:
                if vv.dying and vv.alpha > 0:
                    new_visual_viruses.append(vv)
            
            # Add new viruses
            for v, pos in zip(viruses, old_positions):
                new_visual_viruses.append(VisualVirus(v, pos))
                
            visual_viruses = new_visual_viruses
            
            # Update environment
            environment.update()
            
            # Increment generation
            generation += 1
            last_evolution_time = current_time
            
            print(f"Generation {generation}: Alive={len(alive_viruses)}, Killed={viruses_killed}, Avg Fitness={avg_fitness:.4f}")
        
        # Clear screen
        screen.fill(BACKGROUND_COLOR)
        
        # Update and draw viruses
        for vv in visual_viruses:
            vv.update_position()
            vv.draw(screen)
        
        # Draw horizontal line to separate simulation area
        pygame.draw.line(screen, (100, 100, 100), (0, SIMULATION_HEIGHT), (WIDTH, SIMULATION_HEIGHT), 2)
        
        # Draw environment bar
        draw_environment(screen, environment, WIDTH, ENV_DISPLAY_HEIGHT, SIMULATION_HEIGHT + 2)
        
        # Draw info section
        draw_info(screen, generation, avg_fitness if 'avg_fitness' in locals() else 0.0,
                  drug_applied, viruses_killed, WIDTH, INFO_HEIGHT, SIMULATION_HEIGHT + ENV_DISPLAY_HEIGHT + 2)
        
        # Draw controls info
        controls_text = font.render("Controls: SPACE = Evolve, D = Apply Drug, +/- = Speed Up/Down, ESC = Exit", 
                                    True, (255, 255, 255))
        screen.blit(controls_text, (10, HEIGHT - 30))
        
        # Update display
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_visual_simulation()
