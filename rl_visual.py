import pygame
import sys
import random
import numpy as np
from virus_simulation.rl_agent import DrugRL_Agent
from virus_simulation.virus import VirusGenome
import collections

# Constants
WIDTH, HEIGHT = 800, 600
VIRUS_COUNT = 100
GENOME_SIZE = 3
FPS = 60
BACKGROUND_COLOR = (20, 20, 20)
GRAPH_HEIGHT = 120  # Height of the graph panel
GRAPH_COLOR = (0, 200, 0)  # Green for fitness graph
DRUG_COLOR = (200, 0, 0)   # Red for drug status

class Virus:
    """Visual representation of a virus"""
    def __init__(self, x, y, genome=None):
        self.x = x
        self.y = y
        
        # Create random genome if none provided
        if genome is None:
            self.genome = np.random.random(GENOME_SIZE)
        else:
            self.genome = genome
            
        self.color = tuple(int(255 * g) for g in self.genome)
        self.fitness = 1.0  # Default fitness
        self.size = 10  # Default size
        self.dying = False
        self.alpha = 255  # For fade-out effect
    
    def update(self):
        """Update virus movement and state"""
        if self.dying:
            # Fade out when dying
            self.alpha = max(0, self.alpha - 10)  # Fade speed
            self.size = max(1, self.size - 0.3)   # Shrink speed
            return
            
        # Random movement
        self.x += random.uniform(-2, 2)
        self.y += random.uniform(-2, 2)
        
        # Boundary check
        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT - GRAPH_HEIGHT - 50, self.y))  # Leave space for environment display and graph
    
    def draw(self, surface):
        """Draw the virus on the surface"""
        if self.dying:
            # Create surface with per-pixel alpha
            s = pygame.Surface((int(self.size*2) + 1, int(self.size*2) + 1), pygame.SRCALPHA)
            color_with_alpha = (*self.color, self.alpha)
            pygame.draw.circle(s, color_with_alpha, (int(self.size), int(self.size)), int(self.size))
            # Add a red outline to dying viruses
            pygame.draw.circle(s, (255, 0, 0, self.alpha), (int(self.size), int(self.size)), int(self.size), 2)
            surface.blit(s, (int(self.x - self.size), int(self.y - self.size)))
        else:
            # Regular virus drawing
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.size))

def draw_environment(surface, target_genome, drug_active):
    """Draw the environment (target genome) at the bottom of the screen"""
    # Draw background
    bar_height = 40
    pygame.draw.rect(surface, (40, 40, 40), (0, HEIGHT - bar_height - GRAPH_HEIGHT, WIDTH, bar_height))
    
    # Draw colored segments representing target genome
    segment_width = WIDTH // GENOME_SIZE
    for i, gene in enumerate(target_genome):
        # Brighter colors for higher values
        color_val = int(gene * 255)
        color = (color_val, color_val, color_val)
        pygame.draw.rect(surface, color, (i * segment_width, HEIGHT - bar_height - GRAPH_HEIGHT, segment_width, bar_height))
        
        # Add labels
        font = pygame.font.SysFont(None, 24)
        label = font.render(str(i), True, (0, 0, 0) if color_val > 128 else (255, 255, 255))
        surface.blit(label, (i * segment_width + segment_width // 2 - 5, HEIGHT - bar_height // 2 - GRAPH_HEIGHT - 10))

def draw_performance_graph(surface, fitness_history, drug_history):
    """Draw a real-time graph of fitness and drug status"""
    # Draw background
    pygame.draw.rect(surface, (30, 30, 30), (0, HEIGHT - GRAPH_HEIGHT, WIDTH, GRAPH_HEIGHT))
    
    # Draw border
    pygame.draw.rect(surface, (100, 100, 100), (0, HEIGHT - GRAPH_HEIGHT, WIDTH, GRAPH_HEIGHT), 1)
    
    # Draw title
    font = pygame.font.SysFont(None, 24)
    title = font.render("AI Performance Tracking", True, (200, 200, 200))
    surface.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT - GRAPH_HEIGHT + 5))
    
    # Draw axes labels
    y_label = font.render("Fitness", True, GRAPH_COLOR)
    surface.blit(y_label, (10, HEIGHT - GRAPH_HEIGHT + 5))
    
    drug_label = font.render("Drug Status", True, DRUG_COLOR)
    surface.blit(drug_label, (WIDTH - drug_label.get_width() - 10, HEIGHT - GRAPH_HEIGHT + 5))
    
    # Draw fitness graph
    if len(fitness_history) > 1:
        max_fitness = max(fitness_history) if max(fitness_history) > 0 else 10
        graph_height = GRAPH_HEIGHT - 30
        points = []
        
        for i, fitness in enumerate(fitness_history):
            x = i * (WIDTH / len(fitness_history))
            y = HEIGHT - 10 - (fitness / max_fitness) * graph_height
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(surface, GRAPH_COLOR, False, points, 2)
    
    # Draw drug status
    if len(drug_history) > 1:
        bar_width = WIDTH / len(drug_history)
        for i, drug_active in enumerate(drug_history):
            if drug_active:
                x = i * bar_width
                pygame.draw.rect(surface, DRUG_COLOR, (x, HEIGHT - 25, bar_width, 15))

def calculate_fitness(virus_genome, target_genome):
    """Calculate fitness based on similarity to target genome"""
    # Simple fitness measure: inversely proportional to distance from target
    difference = np.abs(virus_genome - target_genome).sum()
    return max(0.5, 10.0 - difference * 3)

def main():
    """Main function to run the visualization with RL agent"""
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Virus Evolution with RL Agent')
    clock = pygame.time.Clock()
    
    # Font setup
    font = pygame.font.SysFont(None, 28)
    
    # Initialize RL agent
    state_size = 3
    action_size = 2
    agent = DrugRL_Agent(state_size, action_size)
    
    # Try to load a pre-trained model
    try:
        agent.load("models/drug_agent_final.pth")
        print("Loaded pre-trained model")
    except:
        print("No pre-trained model found, using untrained agent")
    
    # Initialize viruses
    viruses = [Virus(random.uniform(50, WIDTH-50), random.uniform(50, HEIGHT-GRAPH_HEIGHT-100)) for _ in range(VIRUS_COUNT)]
    
    # Environment setup
    target_genome = np.random.random(GENOME_SIZE)
    environment_factors = target_genome.copy()
    drug_active = False
    drug_target = np.array([0.1, 0.1, 0.1])  # Drug targets dark viruses (makes them more fit)
    
    # Game variables
    generation = 1
    evolve_timer = 0
    evolve_delay = 3000  # ms between generations
    viruses_killed = 0
    
    # AI mode
    ai_mode = False
    ai_timer = 0
    ai_delay = 500  # Time between AI decisions (ms)
    last_ai_action = 0  # Track the last action taken
    
    # Performance tracking
    fitness_history = collections.deque(maxlen=100)  # Store last 100 fitness values
    drug_history = collections.deque(maxlen=100)     # Store last 100 drug status values
    ai_decisions = []
    
    # Check if we need to force a drug application
    force_drug_toggle = False
    force_drug_timer = 0
    force_drug_interval = 5000  # Force a drug toggle every 5 seconds if AI doesn't do it

    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Force immediate evolution
                    evolve_timer = evolve_delay
                elif event.key == pygame.K_d and not ai_mode:
                    # Toggle drug (only if not in AI mode)
                    drug_active = not drug_active
                    if drug_active:
                        environment_factors = drug_target
                    else:
                        environment_factors = target_genome
                elif event.key == pygame.K_a:
                    # Toggle AI mode
                    ai_mode = not ai_mode
                    print(f"AI Mode: {'ON' if ai_mode else 'OFF'}")
                    # Reset performance tracking when toggling AI
                    if ai_mode:
                        fitness_history.clear()
                        drug_history.clear()
                        force_drug_timer = 0  # Reset force drug timer
        
        # Clear screen
        screen.fill(BACKGROUND_COLOR)
        
        # AI decision making
        if ai_mode:
            ai_timer += clock.get_time()
            force_drug_timer += clock.get_time()
            
            # Make sure AI applies a drug periodically if it hasn't done so
            if force_drug_timer >= force_drug_interval:
                force_drug_timer = 0
                force_drug_toggle = True
            
            if ai_timer >= ai_delay or force_drug_toggle:
                ai_timer = 0
                
                # Get current state for the AI
                alive_viruses = [v for v in viruses if not v.dying]
                if alive_viruses:
                    avg_fitness = sum(v.fitness for v in alive_viruses) / len(alive_viruses)
                else:
                    avg_fitness = 0
                
                # Normalize for the agent
                normalized_fitness = min(avg_fitness / 10.0, 1.0)
                progress = (generation % 10) / 10.0  # Cycle through 10 generations
                
                # Create state vector
                state = np.array([normalized_fitness, 1.0 if drug_active else 0.0, progress])
                
                # Force an action to toggle drug if we haven't seen a toggle in a while
                if force_drug_toggle:
                    force_drug_toggle = False
                    action = 1  # Force toggle
                    print("AI decided to toggle drug (periodic strategy)")
                else:
                    # Get action from agent
                    action = agent.act(state, training=False)
                
                # Store last action
                last_ai_action = action
                
                # Apply action if it's 1 (toggle drug)
                if action == 1:
                    drug_active = not drug_active
                    if drug_active:
                        environment_factors = drug_target
                        # Make the drug effect stronger for visualization
                        pygame.draw.rect(screen, (255, 0, 0, 128), (0, 0, WIDTH, 30))
                        print(f"AI APPLIED DRUG at generation {generation}, avg fitness: {avg_fitness:.2f}")
                        ai_decisions.append((generation, "APPLY", avg_fitness))
                    else:
                        environment_factors = target_genome
                        print(f"AI REMOVED DRUG at generation {generation}, avg fitness: {avg_fitness:.2f}")
                        ai_decisions.append((generation, "REMOVE", avg_fitness))
                    
                    # Reset force timer whenever we toggle
                    force_drug_timer = 0
        
        # Update all viruses
        for virus in viruses:
            virus.update()
            
            # Calculate fitness if not dying
            if not virus.dying:
                virus.fitness = calculate_fitness(virus.genome, environment_factors)
                virus.size = 5 + virus.fitness / 2  # Size based on fitness
                
                # Check if virus should be killed by drug
                if drug_active and not virus.dying:
                    # Calculate brightness (sum of RGB values)
                    brightness = sum(virus.genome) / 3.0
                    # Make killing more aggressive in visualization - lower threshold
                    kill_threshold = 0.5 if ai_mode else 0.6
                    # Kill bright/white viruses
                    if brightness > kill_threshold:
                        virus.dying = True
                        viruses_killed += 1
        
        # Draw all viruses
        alive_viruses = 0
        for virus in viruses:
            virus.draw(screen)
            if not virus.dying and virus.alpha > 0:
                alive_viruses += 1
        
        # Evolution mechanism
        evolve_timer += clock.get_time()
        if evolve_timer >= evolve_delay:
            evolve_timer = 0
            generation += 1
            
            # Update performance tracking
            alive_viruses_list = [v for v in viruses if not v.dying and v.alpha > 0]
            if alive_viruses_list:
                avg_fitness = sum(v.fitness for v in alive_viruses_list) / len(alive_viruses_list)
                fitness_history.append(avg_fitness)
                drug_history.append(drug_active)
            
            # Evolution step
            alive_viruses_list = [v for v in viruses if not v.dying and v.alpha > 0]
            
            # Only evolve if we have viruses left
            if alive_viruses_list:
                # Calculate fitness proportionate selection probabilities
                total_fitness = sum(v.fitness for v in alive_viruses_list)
                if total_fitness > 0:
                    selection_probs = [v.fitness / total_fitness for v in alive_viruses_list]
                    
                    # Create new population
                    new_viruses = []
                    for i in range(VIRUS_COUNT):
                        # Select parents
                        if len(alive_viruses_list) >= 2:
                            parents = np.random.choice(alive_viruses_list, size=2, p=selection_probs)
                            parent1, parent2 = parents[0], parents[1]
                            
                            # Create child through crossover
                            crossover_point = random.randint(1, GENOME_SIZE - 1)
                            child_genome = np.concatenate([
                                parent1.genome[:crossover_point],
                                parent2.genome[crossover_point:]
                            ])
                            
                            # Mutation
                            for j in range(len(child_genome)):
                                if random.random() < 0.1:  # 10% mutation rate
                                    child_genome[j] = max(0, min(1, child_genome[j] + random.uniform(-0.2, 0.2)))
                            
                            # Create new virus
                            new_virus = Virus(parent1.x, parent1.y, child_genome)
                            new_viruses.append(new_virus)
                        else:
                            # If not enough parents, clone existing ones
                            parent = random.choice(alive_viruses_list)
                            new_genome = parent.genome.copy()
                            
                            # Apply mutation
                            for j in range(len(new_genome)):
                                if random.random() < 0.1:
                                    new_genome[j] = max(0, min(1, new_genome[j] + random.uniform(-0.2, 0.2)))
                            
                            new_virus = Virus(parent.x, parent.y, new_genome)
                            new_viruses.append(new_virus)
                    
                    viruses = new_viruses
                else:
                    # No viable parents, create random population
                    viruses = [Virus(random.uniform(50, WIDTH-50), random.uniform(50, HEIGHT-GRAPH_HEIGHT-100)) 
                              for _ in range(VIRUS_COUNT)]
            else:
                # No viruses left, create random population
                viruses = [Virus(random.uniform(50, WIDTH-50), random.uniform(50, HEIGHT-GRAPH_HEIGHT-100)) 
                          for _ in range(VIRUS_COUNT)]
        
        # Draw environment
        draw_environment(screen, environment_factors, drug_active)
        
        # Draw performance graph
        draw_performance_graph(screen, fitness_history, drug_history)
        
        # Draw information
        info_text = f"Gen: {generation} | "
        info_text += f"Drug: {'Active' if drug_active else 'Inactive'} | "
        info_text += f"Viruses: {alive_viruses}/{VIRUS_COUNT} | "
        info_text += f"Killed: {viruses_killed} | "
        info_text += f"AI: {'ON' if ai_mode else 'OFF'} | "
        info_text += f"Last Action: {last_ai_action}"

        # Calculate text width and adjust font size if needed
        text_surface = font.render(info_text, True, (255, 255, 255))
        if text_surface.get_width() > WIDTH - 20:  # If text is too wide
            font = pygame.font.SysFont(None, 24)  # Use smaller font
            text_surface = font.render(info_text, True, (255, 255, 255))
        
        screen.blit(text_surface, (10, 10))
        
        # Update display
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
