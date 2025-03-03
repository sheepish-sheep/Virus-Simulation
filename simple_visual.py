import pygame
import sys
import numpy as np
import random
import math

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Virus Evolution")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Virus properties
class Virus:
    def __init__(self, genome_size=3):
        self.genome = np.random.random(genome_size)
        self.x = random.randint(20, WIDTH - 20)
        self.y = random.randint(20, HEIGHT - 100)
        self.size = 10
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.color = self.get_color()
        self.fitness = 0
        self.dying = False
        self.alpha = 255  # For fade-out effect
        
    def get_color(self):
        r = int(self.genome[0] * 255)
        g = int(self.genome[1] * 255)
        b = int(self.genome[2] * 255)
        return (r, g, b)
    
    def mutate(self, rate=0.1):
        for i in range(len(self.genome)):
            if random.random() < rate:
                change = random.uniform(-0.1, 0.1)
                self.genome[i] += change
                self.genome[i] = max(0, min(1, self.genome[i]))
        self.color = self.get_color()
    
    def update(self):
        if self.dying:
            # Fade out when dying
            self.alpha = max(0, self.alpha - 10)  # Slower fade
            self.size = max(1, self.size - 0.3)   # Slower shrink
            return
            
        # Move
        self.x += self.vx
        self.y += self.vy
        
        # Bounce off walls
        if self.x < 10 or self.x > WIDTH - 10:
            self.vx *= -1
        if self.y < 10 or self.y > HEIGHT - 100:
            self.vy *= -1
        
        # Keep in bounds
        self.x = max(10, min(WIDTH - 10, self.x))
        self.y = max(10, min(HEIGHT - 100, self.y))
    
    def draw(self, surface):
        if self.dying:
            # Create surface with per-pixel alpha
            s = pygame.Surface((int(self.size*2) + 1, int(self.size*2) + 1), pygame.SRCALPHA)
            color_with_alpha = (*self.color, self.alpha)
            pygame.draw.circle(s, color_with_alpha, (int(self.size), int(self.size)), int(self.size))
            # Add a red outline to dying viruses
            pygame.draw.circle(s, (255, 0, 0, self.alpha), (int(self.size), int(self.size)), int(self.size), 2)
            surface.blit(s, (int(self.x - self.size), int(self.y - self.size)))
        else:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.size))
            pygame.draw.circle(surface, BLACK, (int(self.x), int(self.y)), int(self.size), 1)

# Environment (target genome)
class Environment:
    def __init__(self, genome_size=3):
        self.target = np.random.random(genome_size)
        self.drug_active = False
        self.original_target = self.target.copy()
        
    def get_fitness(self, virus):
        # Calculate distance between virus genome and target
        distance = sum((virus.genome - self.target)**2)**0.5
        # Convert to fitness (1 = perfect match, 0 = furthest possible)
        fitness = 1 - distance / math.sqrt(len(virus.genome))
        return max(0, fitness)
    
    def apply_drug(self):
        # Save original target if not saved
        if not hasattr(self, 'original_target'):
            self.original_target = self.target.copy()
            
        # Set target to make white viruses less fit
        self.target = np.array([0.1, 0.1, 0.1])  # Low values make white viruses have low fitness
        self.drug_active = True
    
    def remove_drug(self):
        # Return to original target
        if hasattr(self, 'original_target'):
            self.target = self.original_target.copy()
        else:
            self.target = np.random.random(len(self.target))
        self.drug_active = False
    
    def draw(self, surface):
        # Draw environment bar at bottom
        bar_width = WIDTH / len(self.target)
        for i, value in enumerate(self.target):
            # Draw rectangles representing target values
            color = (int(value * 255), int(value * 255), int(value * 255))
            pygame.draw.rect(surface, color, 
                            (i * bar_width, HEIGHT - 80, bar_width, 30))
            
        # Draw drug status
        status = "DRUG ACTIVE" if self.drug_active else "NO DRUG" 
        font = pygame.font.SysFont(None, 36)
        text = font.render(status, True, RED if self.drug_active else GREEN)
        surface.blit(text, (WIDTH - 200, HEIGHT - 70))

# Main simulation
def run_simulation():
    # Create viruses
    num_viruses = 20
    viruses = [Virus() for _ in range(num_viruses)]
    environment = Environment()
    
    # Stats
    generation = 0
    viruses_killed = 0
    killing_threshold = 0.3  # Fitness below this will be killed
    font = pygame.font.SysFont(None, 24)
    
    # Main loop
    clock = pygame.time.Clock()
    evolve_timer = 0
    running = True
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Force evolution
                    evolve_timer = 1000
                elif event.key == pygame.K_d:
                    # Toggle drug
                    if environment.drug_active:
                        environment.remove_drug()
                    else:
                        environment.apply_drug()
        
        # Clear screen
        screen.fill(WHITE)
        
        # Update viruses
        for virus in viruses:
            virus.update()
            virus.draw(screen)
            # Calculate fitness
            virus.fitness = environment.get_fitness(virus)
            # Update size based on fitness
            if not virus.dying:
                virus.size = 5 + int(virus.fitness * 15)
            
            # Check if virus should be killed by drug
            # For more dramatic effect:
            # White-ish viruses (high in all RGB values) are killed by the drug
            if environment.drug_active and not virus.dying:
                # Calculate brightness (sum of RGB values)
                brightness = sum(virus.genome) / 3.0
                # Kill bright/white viruses
                if brightness > 0.6:  # More sensitive threshold
                    virus.dying = True
                    viruses_killed += 1
                    # Print for debugging
                    print(f"Killing virus: RGB={virus.genome}, brightness={brightness}")
        
        # Evolution timer
        evolve_timer += clock.get_time()
        if evolve_timer >= 3000:  # Evolve every 3 seconds
            evolve_timer = 0
            generation += 1
            
            # Filter out dead viruses
            alive_viruses = [v for v in viruses if not v.dying]
            
            # If all viruses are killed, create new random population
            if not any(not v.dying for v in viruses):
                print("All viruses killed! Creating new population...")
                viruses = [Virus() for _ in range(num_viruses)]
                environment.remove_drug()  # Give them a chance
                continue
                
            # Get fitness scores of alive viruses
            fitness_scores = [v.fitness for v in viruses if not v.dying]
            avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0
            
            # Select parents through tournament selection
            new_viruses = []
            
            # Keep dying viruses for animation
            for v in viruses:
                if v.dying and v.alpha > 0:
                    new_viruses.append(v)
            
            # Only consider alive viruses for reproduction
            alive_viruses = [v for v in viruses if not v.dying]
            
            # If we have enough viruses, keep elite performers
            if len(alive_viruses) >= 2:
                # Keep top performers (elitism)
                sorted_viruses = sorted(alive_viruses, key=lambda v: v.fitness, reverse=True)
                for i in range(min(2, len(sorted_viruses))):
                    elite = Virus()
                    elite.genome = sorted_viruses[i].genome.copy()
                    elite.x = sorted_viruses[i].x
                    elite.y = sorted_viruses[i].y
                    elite.color = elite.get_color()
                    new_viruses.append(elite)
            
            # Create rest through reproduction
            while len(new_viruses) < num_viruses + len([v for v in viruses if v.dying and v.alpha > 0]):
                if len(alive_viruses) <= 1:
                    # If only one virus is alive, clone it with mutation
                    if len(alive_viruses) == 1:
                        child = Virus()
                        child.genome = alive_viruses[0].genome.copy()
                        child.x = alive_viruses[0].x + random.uniform(-20, 20)
                        child.y = alive_viruses[0].y + random.uniform(-20, 20)
                        child.mutate(rate=0.2)  # Higher mutation to increase diversity
                        child.color = child.get_color()
                        new_viruses.append(child)
                    continue
                    
                # Tournament selection - pick 3 random viruses (or fewer if not enough), take the best
                parent_indices = []
                for _ in range(2):
                    tournament_size = min(3, len(alive_viruses))
                    tournament = random.sample(range(len(alive_viruses)), tournament_size)
                    winner = max(tournament, key=lambda i: alive_viruses[i].fitness)
                    parent_indices.append(winner)
                
                # Create child
                child = Virus()
                parent1 = alive_viruses[parent_indices[0]]
                parent2 = alive_viruses[parent_indices[1]]
                
                # Crossover
                crossover_point = random.randint(0, len(child.genome)-1)
                for i in range(len(child.genome)):
                    if i < crossover_point:
                        child.genome[i] = parent1.genome[i]
                    else:
                        child.genome[i] = parent2.genome[i]
                
                # Position near parents
                child.x = (parent1.x + parent2.x) / 2 + random.uniform(-20, 20)
                child.y = (parent1.y + parent2.y) / 2 + random.uniform(-20, 20)
                
                # Mutate
                child.mutate()
                child.color = child.get_color()
                
                new_viruses.append(child)
            
            # Update population
            viruses = new_viruses
            
            print(f"Generation {generation}: Alive={len(alive_viruses)}, Killed={viruses_killed}, Avg Fitness={avg_fitness:.4f}")
        
        # Draw environment
        environment.draw(screen)
        
        # Draw stats
        alive_count = sum(1 for v in viruses if not v.dying)
        stats_text = f"Generation: {generation}   Alive: {alive_count}   Killed: {viruses_killed}"
        text = font.render(stats_text, True, BLACK)
        screen.blit(text, (10, 10))
        
        # Draw controls
        controls = "Controls: SPACE = Force Evolution, D = Toggle Drug, ESC = Exit"
        controls_text = font.render(controls, True, BLACK)
        screen.blit(controls_text, (10, 40))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_simulation()
