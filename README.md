# Vanilla Option-Critic

This repository contains a PyTorch implementation of the **Option-Critic** architecture for skill discovery in reinforcement learning, along with DQN and PPO baselines. The primary evaluation domain is a custom Four Rooms environment.

---

## Project Structure

```
option_critic_dissertation/
├── agents/                # Option-Critic agent definitions (MLP and CNN)
├── buffers/               # Experience replay buffer
├── envs/                  # Custom Four Rooms environment
├── learners/              # Gradient computation for actor & critic
├── train/                 # Main training scripts
├── utils/                 # Environment setup and logging
├── visuals.py             # Plotting reward curves and metrics
├── dqn_agent.py           # DQN baseline agent
├── dqn_baseline.py        # DQN training script
├── ppo_baseline.py        # PPO training script
└── README.md              # (You're here)
```

---

## Option-Critic Overview

The Option-Critic (OC) architecture enables temporal abstraction by learning:
- Intra-option policies
- Termination probabilities
- Option-value functions

This codebase supports both MLP (for gridworld) and CNN (for image-based tasks) variants.

---

## Environments
- **Four Rooms**: A sparse-reward navigation task implemented in `envs/four_rooms_env.py`

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/AkshayaJeyaram/vanilla-option-critic.git
cd vanilla-option-critic
```

### 2. Set Up Python Environment
We recommend using `pyenv` and `virtualenv`:
```bash
pyenv install 3.10.13
pyenv virtualenv 3.10.13 oc-env
pyenv activate oc-env
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Training Option-Critic

```bash
python train/train_option_critic.py --env fourrooms
```

Key arguments:
- `--num-options`: number of options (default: 2)
- `--max-steps-total`: total environment steps (default: 4M)
- `--logdir`: directory to save logs and models
- `--switch-goal`: enable dynamic goal switching during training

---

## Visualizations
Run the following to generate reward and option usage plots:
```bash
python visuals.py
```
This produces graphs in the `graphs/` directory:
- `reward_curve.png`
- `option_lengths.png`
- `termination_probs.png`
- `comparison_reward_curve_all.png`

---

## Baselines

### DQN:
```bash
python dqn_baseline.py
```

### PPO:
```bash
python ppo_baseline.py
```

Logs will be saved to `dqn_logs/` and `ppo_logs/` respectively.

---

## Results Summary
| Agent           | Final Reward | Convergence Rate | Interpretability |
|----------------|---------------|------------------|------------------|
| Option-Critic  | High          | Fast             | ✔️ Skills emerge |
| PPO            | High          | Fast             | ❌ Flat policy   |
| DQN            | Lower         | Slower           | ❌              |

OC agents learned interpretable skills and subgoal-directed behavior, especially under entropy annealing and regularization.

---

## Next Steps
This implementation serves as a foundation for building **Soft Option-Critic (SOC)** and advanced skill discovery algorithms in continuous or visual domains.

