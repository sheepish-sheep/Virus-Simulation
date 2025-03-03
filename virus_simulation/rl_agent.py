import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    """
    Deep Q-Network for learning the optimal drug administration policy
    """
    def __init__(self, state_size, action_size):
        """
        Initialize the DQN model
        
        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input state
            
        Returns:
            Q-values for each action
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DrugRL_Agent:
    """
    Reinforcement Learning agent for learning optimal drug strategies
    """
    def __init__(self, state_size, action_size):
        """
        Initialize the agent
        
        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Memory for experience replay
        self.memory = deque(maxlen=2000)
        
        # Q-Network and Target Network
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Update target network initially
        self.update_target_model()
        
    def update_target_model(self):
        """Update the target model to match the policy model"""
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory for replay
        """
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        """
        Choose an action based on the current state
        
        Args:
            state: Current state observation
            training: Whether the agent is training or evaluating
            
        Returns:
            Chosen action (0=no drug change, 1=toggle drug)
        """
        # During training, use epsilon-greedy policy
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Convert state to tensor for model input
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state)
        
        # Return action with highest Q-value
        return np.argmax(act_values.numpy())
    
    def replay(self, batch_size):
        """
        Train the model on a batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
        """
        # Check if we have enough samples
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            # Convert to tensors
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            
            # Target Q-value
            if done:
                target = reward
            else:
                # Double DQN: Use main network to select action, target network to evaluate
                with torch.no_grad():
                    next_action = torch.argmax(self.model(next_state.unsqueeze(0))).item()
                    target = reward + self.gamma * self.target_model(next_state.unsqueeze(0))[0][next_action].item()
            
            # Current Q-value
            current = self.model(state.unsqueeze(0))[0][action].item()
            
            # Create target vector
            target_f = self.model(state.unsqueeze(0)).detach().numpy()
            target_f[0][action] = target
            
            # Convert to tensor
            target_f = torch.FloatTensor(target_f)
            
            # Train the model
            self.optimizer.zero_grad()
            output = self.model(state.unsqueeze(0))
            loss = self.criterion(output, target_f)
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model weights from file"""
        self.model.load_state_dict(torch.load(name))
        self.update_target_model()
    
    def save(self, name):
        """Save model weights to file"""
        torch.save(self.model.state_dict(), name)
