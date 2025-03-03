import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import os

base_path = os.path.dirname(os.path.abspath(__file__))
print("Virus Simulation Results")
print("------------------------")

# Display fitness evolution
fitness_img_path = os.path.join(base_path, "virus_sim_fitness.png")
if os.path.exists(fitness_img_path):
    print("Fitness Evolution Graph:")
    img = mpimg.imread(fitness_img_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Display virus genomes
genomes_img_path = os.path.join(base_path, "virus_sim_genomes.png")
if os.path.exists(genomes_img_path):
    print("Virus Genomes Heatmap:")
    img = mpimg.imread(genomes_img_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Display environment changes
env_img_path = os.path.join(base_path, "virus_sim_environment.png")
if os.path.exists(env_img_path):
    print("Environment Changes Graph:")
    img = mpimg.imread(env_img_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
