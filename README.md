# Advanced Machine Learning Assignment 


**Author:** Midhun Shyam  
**Student ID:** 22058122
Date: October 2024

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
- Task 2: Convolutional Autoencoder
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
 
## Results Summary

| Task/Bonus                                      | Model / Agent                             | Key Hyperparameters                            | Test Metric                                             |
|-------------------------------------------------|-------------------------------------------|-------------------------------------------------|---------------------------------------------------------|
| **Task 1: CNN Position Prediction**             | 3‑layer CNN                               | LR = 1 e‑3 ; Batch = 64 ; Epochs = 30            | **Avg Test Loss** = 0.0004                              |
| **Task 2: Convolutional Autoencoder**           | 4‑block Conv Autoencoder                  | LR = 5 e‑4 ; Batch = 32 ; Epochs = 50            | **Avg Test Loss** = 0.0183                              |
| **Task 3: RL Agent (Level 1)**                  | Policy‑Gradient (Actor‑Critic, PGD)       | LR = 5 e‑3 ; γ = 0.99 ; Episodes = 500           | **Test‑Average** = 227.76 ± 185.17                      |
| **Task 4: RL Agent (Level 3)**                  | SARSA‑style Actor‑Critic                  | LR = 5 e‑4 ; γ = 0.95 ; Episodes = 1 000         | **Test‑Average** = 58.24 ± 53.87                        |
| **Bonus 1: DZ Prediction (NN)**                 | Feed‑forward CNN regressor                | LR = 1 e‑2 ; Batch = 32 ; Epochs = 200           | **Avg Loss on Test Set** = 0.0552                       |
| **Bonus 2: DQN Q‑Learning**                     | DQN w/ Replay Buffer                      | LR = 1 e‑4 ; γ = 0.99 ; ε decays 1→0.01          | **Test‑Average** = 152.88 ± 163.56                      |
| **Bonus 3: ModifiedEnv (Negative‐Reward PGD)**  | PGD Policy in `ModifiedMiniPongEnv`       | Same as Task 3 PGD; env adds –1 per time‐step    | **Test‑Average** = 263.88 ± 219.38                      |

### Factors & Differences

- **Supervised vs. RL vs. Regres­sion**  
  - Tasks 1–2 are pure supervised; Bonuses 1 and DZ use MSE loss.  
  - Tasks 3–4 and Bonuses 2–3 are reinforcement‐learning, comparing policy (PGD, SARSA) vs. value (DQN) methods.
- **Env Complexity**  
  - Level 1 (Task 3) only x‑coordinate → high reward (227.8 avg).  
  - Level 3 (Task 4) adds offset tracking dz → much lower reward (58.2 avg).  
  - ModifiedEnv (Bonus 3) adds a −1 penalty per step → forces faster solutions, boosting avg reward back up to 263.9.
- **Exploration & Stability**  
  - PGD and SARSA use on‑policy updates; DQN uses off‑policy replay and ε‑greedy.  
  - DQN’s high variance (±163.6) reflects instability with large Q‑updates.
- **Loss Scales**  
  - CNN and autoencoder losses are near zero (0.0004 vs. 0.0183).  
  - DZ regression sits at ~0.0552, indicating slightly noisier mapping from pixels to dz.

### Visualisation

All plots (training curves, reconstructions, agent performance) are generated inline in the notebook under the “Visualisation” sections. You can export figures via Jupyter or save them programmatically by uncommenting the plt.savefig(...) lines.

