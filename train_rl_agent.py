import numpy as np
import matplotlib.pyplot as plt
from virus_simulation.rl_environment import VirusRLEnvironment
from virus_simulation.rl_agent import DrugRL_Agent
import os

def train_agent(episodes=200, batch_size=64, save_interval=100):
    """
    Train the RL agent to learn optimal drug administration strategies
    
    Args:
        episodes: Number of episodes to train
        batch_size: Size of batch for experience replay
        save_interval: Interval for saving the model
    """
    # Create environment and agent
    env = VirusRLEnvironment(pop_size=100, genome_size=3, episode_length=50)
    state_size = 3  # [normalized_fitness, drug_status, progress]
    action_size = 2  # [do_nothing, toggle_drug]
    agent = DrugRL_Agent(state_size, action_size)
    
    # Lists to track progress
    scores = []
    avg_fitness_values = []
    drug_actions = []
    
    print("Starting training for {} episodes...".format(episodes))
    
    # Training loop
    for e in range(1, episodes + 1):
        # Reset environment
        state = env.reset()
        total_reward = 0
        actions_taken = []
        episode_avg_fitness = []
        
        # Episode loop
        for step in range(env.episode_length):
            # Choose action
            action = agent.act(state)
            actions_taken.append(action)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward
            episode_avg_fitness.append(env.avg_fitness)
            
            # Train model on batch of experiences
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            if done:
                break
        
        # After episode finished, update target model periodically
        if e % 5 == 0:
            agent.update_target_model()
        
        # Record scores and statistics
        scores.append(total_reward)
        avg_fitness = np.mean(episode_avg_fitness)
        avg_fitness_values.append(avg_fitness)
        drug_use_percent = (sum(actions_taken) / len(actions_taken)) * 100
        drug_actions.append(drug_use_percent)
        
        # Print progress
        print("Episode {}/{}: Score: {:.2f}, Avg Fitness: {:.2f}, Drug Use: {:.1f}%, Epsilon: {:.2f}"
              .format(e, episodes, total_reward, avg_fitness, drug_use_percent, agent.epsilon))
        
        # Save model periodically
        if e % save_interval == 0:
            if not os.path.exists("models"):
                os.makedirs("models")
            agent.save("models/drug_agent_ep{}.pth".format(e))
    
    # Save final model
    agent.save("models/drug_agent_final.pth")
    
    # Plot training results
    plot_training_results(scores, avg_fitness_values, drug_actions, episodes)
    
    print("Training completed!")
    return agent

def plot_training_results(scores, avg_fitness, drug_actions, episodes):
    """
    Plot the training results
    
    Args:
        scores: List of episode scores
        avg_fitness: List of average fitness values
        drug_actions: List of drug usage percentages
        episodes: Number of episodes trained
    """
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot episode rewards
    ax1.plot(range(1, episodes + 1), scores)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    
    # Plot average virus fitness
    ax2.plot(range(1, episodes + 1), avg_fitness, color='red')
    ax2.set_title('Average Virus Fitness')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Fitness')
    
    # Plot drug usage percentage
    ax3.plot(range(1, episodes + 1), drug_actions, color='green')
    ax3.set_title('Drug Usage (%)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Percentage')
    
    plt.tight_layout()
    
    # Save figure
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig("results/training_results.png")
    plt.close()

if __name__ == "__main__":
    # Train the agent
    agent = train_agent(episodes=200, batch_size=64)
