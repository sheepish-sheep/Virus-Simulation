# Virus "Evolution" with Reinforcement Learning 

## Overview
(This is my first project so the simulation is very simple)
This project simulates virus population evolution and demonstrates how AI can discover optimal drug treatment strategies. The reinforcement learning agent learns when to apply and remove drugs to minimize virus fitness while preventing resistance development.

## Key Features

- **Interactive Visualization**: Watch viruses evolve in real-time
- **AI Drug Control**: Toggle between manual and AI-controlled drug administration
- **Performance Tracking**: Real-time graph showing virus fitness and drug application
- **Strategic Drug Application**: AI learns optimal drug cycling patterns

## How to Run

```bash
python rl_visual.py
```

## Controls

- **A**: Toggle AI mode (on/off)
- **D**: Manually apply/remove drug (when AI mode is off)
- **SPACE**: Force immediate evolution to next generation
- **ESC**: Exit simulation

## Visual Elements

### Viruses
- **Circles**: Each represents a virus
- **Color**: Represents genome (RGB values = gene values)
- **Size**: Larger circles = higher fitness
- **Red Outline**: Indicates a virus being killed by drugs

### Environment Bar
- Shows current environmental conditions at the bottom
- Brighter sections = conditions favoring specific genes

### Information Panel
- **Gen**: Current generation number
- **Drug**: Whether drug is active/inactive
- **Viruses**: Count of living viruses
- **Killed**: Number of viruses killed by drugs
- **AI**: Whether AI mode is on/off
- **Last Action**: AI's most recent action (0=no change, 1=toggle drug)

### Performance Graph
- **Green Line**: Virus population fitness over time (lower is better)
- **Red Bars**: Periods when drugs are active
- Shows the relationship between drug application and fitness changes

## How the AI Works

The reinforcement learning agent:

- **Observes**: Current virus fitness, drug status, generation progress
- **Actions**: Can maintain or toggle drug status
- **Goal**: Minimize virus fitness while preventing resistance
- **Strategy**: Learns when to apply/remove drugs for optimal control
- **Decision Making**: Combines learned policy with periodic drug cycling
- **Training**: 
  - Initial training: 500 episodes with batch size of 32
  - Additional training: 200 episodes with batch size of 64 for more stable learning

## Understanding AI Performance

- **Effective Strategies** typically show:
  - Drug application when fitness gets too high
  - Drug removal before resistance fully develops
  - Repeated cycling to prevent sustained resistance
  - Overall lower fitness than manual control

- **AI Decision Indicators**:
  - Action 0: Maintain current drug status
  - Action 1: Toggle drug (apply if inactive, remove if active)

## Tech Stack

- **Python**: Primary language 
- **PyTorch**: For reinforcement learning
- **NumPy**: For numerical operations
- **Pygame**: For visualization

## Requirements

```bash
pip install torch numpy pygame
