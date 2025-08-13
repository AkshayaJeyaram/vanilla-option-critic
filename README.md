# Vanilla Option-Critic

This repository contains a PyTorch implementation of the **Option-Critic** architecture for skill discovery in reinforcement learning, along with DQN and PPO baselines. The primary evaluation domain is a custom Four Rooms environment.

---

## Project Structure

```
vanilla-option-critic/
├── README.md
├── requirements.txt
│
├── agents/
│   ├── __init__.py
│   └── option_critic.py
│
├── buffers/
│   ├── __init__.py
│   └── experience_buffer.py
│
├── learners/
│   ├── __init__.py
│   └── gradients.py
│
├── utils/
│   ├── __init__.py
│   ├── environment_utils.py
│   └── logging_utils.py
│
├── envs/
│   ├── __init__.py
│   └── four_rooms_env.py
│
├── train/
│   ├── __init__.py
│   └── train_option_critic.py
│
├── models/
│   ├── vanilla_oc_fourrooms_seed0_ep1000.pth
│   ├── vanilla_oc_fourrooms_seed0_ep2000.pth
│   └── vanilla_oc_fourrooms_seed0_ep4000.pth
│
├── graphs/
│   ├── comparison_reward_curve.png
│   ├── comparison_reward_curve_all.png
│   ├── entropy_curve.png
│   ├── loss_curves.png
│   ├── option_frequency.png
│   ├── option_lengths.png
│   ├── reward_curve.png
│   └── termination_probs.png
│
├── documents/
│   └── Temporary Writeup.pdf
│
├── runs/
│   ├── PPO-fourrooms/
│   │   └── training_metrics.csv
│   └── OptionCriticMLP-fourrooms-default/
│       ├── logger.log
│       ├── session_log.log
│       └── training_metrics.csv
│
├── dqn_logs/
│   └── fourrooms_rewards.csv
│
├── ppo_logs/
│   └── fourrooms_rewards.csv
│
├── dqn_agent.py
├── dqn_baseline.py
├── ppo_baseline.py
├── visuals.py

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

