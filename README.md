# Advanced Machine Learning Assignment — 2024

**Author:** Midhun Shyam  
**Student ID:** 22058122

## Description

This repository contains the notebook **_AML.ipynb_**, which walks through a series of advanced machine learning and reinforcement learning tasks using a custom MiniPong environment. You will:

1. **Task 1**: Train a convolutional neural network (CNN) to predict the `x`‐coordinate of a moving marker.  
2. **Task 2**: Train a convolutional autoencoder to learn a compressed representation of the game frames.  
3. **Task 3**: Build and train a basic RL agent for MiniPong (level 1) using a neural‐network‐based policy.  
4. **Task 4**: Extend the RL agent to handle level 3, which adds ball–paddle offset tracking (`dz`).  
5. **Bonus**: Implement Q‐learning via a Deep Q‑Network (DQN) for extra exploration of value‐based methods. Time-step reinforcement learning using negative reward structure using superclass. 

## Task Overviews

- Task 1: CNN for Position Prediction
  - Goal: Predict the ball’s x‐coordinate from a single frame.
  - Approach: Build a small CNN classifier/regressor; use MSE or cross‐entropy based on binning.
-Task 2: Convolutional Autoencoder
  - Goal: Learn a low‐dimensional embedding of game frames.
  - Approach: Train an undercomplete autoencoder; inspect reconstruction quality.
- Task 3: RL Agent (Level 1)
  - Goal: Train a policy network to play MiniPong at the simplest level.
  - Approach: Define policy architecture; apply policy gradient or actor‐critic.
- Task 4: RL Agent (Level 3)
  - Goal: Extend the agent to handle paddle offset (dz) tracking.
  - Approach: Add extra inputs and tune γ, learning rate, and network size.
- Bonus: DQN Q‐Learning
  - Goal: Compare value‐based learning to policy methods.
  - **Approach**: Implement experience replay, ε‐greedy exploration.
 
## Results & Visualisation

All plots (training curves, reconstructions, agent performance) are generated inline in the notebook under the “Visualisation” sections. You can export figures via Jupyter or save them programmatically by uncommenting the plt.savefig(...) lines.
