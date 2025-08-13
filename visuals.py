import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import numpy as np

os.makedirs("graphs", exist_ok=True)

def extract_rewards(log_path):
    rewards = []
    with open(log_path, 'r') as f:
        for line in f:
            # Accept both "Reward: 1.00" and "reward=1.00"
            if "Reward:" in line:
                try:
                    # ... | Reward: 1.00 | ...
                    seg = [p for p in line.split("|") if "Reward" in p][0]
                    reward = float(seg.split(":")[1].strip())
                    rewards.append(reward)
                except Exception:
                    pass
            elif "reward=" in line:
                try:
                    seg = [p for p in line.split("|") if "reward=" in p][0]
                    reward = float(seg.split("=")[1].strip())
                    rewards.append(reward)
                except Exception:
                    pass
    return rewards

def smooth(data, window=100):
    return pd.Series(data).rolling(window, min_periods=1).mean()

def plot_reward_curve():
    rewards = extract_rewards("runs/OptionCriticMLP-fourrooms-default/logger.log")
    plt.figure(figsize=(10, 5))
    plt.plot(smooth(rewards, window=100), label="Episode Reward (smoothed)")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Smoothed Reward Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/reward_curve.png")
    plt.close()

# ========== Entropy Curve ==========
def plot_entropy():
    df = pd.read_csv("runs/OptionCriticMLP-fourrooms-default/training_metrics.csv")
    plt.figure(figsize=(10, 5))
    plt.plot(smooth(df["entropy"], window=500), label="Entropy (smoothed)")
    plt.xlabel("Steps")
    plt.ylabel("Entropy")
    plt.title("Intra-Option Policy Entropy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/entropy_curve.png")
    plt.close()

# ========== Option Length Plot ==========
def extract_option_lengths(log_path):
    option_lens = defaultdict(list)
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Option" in line and "avg len" in line:
                parts = line.strip().split(":")
                option = int(parts[0].split(" ")[-1])
                avg_len = float(parts[1].split(",")[0].split("=")[-1])
                option_lens[option].append(avg_len)
    return option_lens

def plot_option_lengths():
    option_lens = extract_option_lengths("runs/OptionCriticMLP-fourrooms-default/logger.log")
    options = list(option_lens.keys())
    means = [np.mean(option_lens[o]) for o in options]

    plt.figure(figsize=(8, 5))
    plt.bar(options, means)
    plt.xlabel("Option Index")
    plt.ylabel("Average Option Length")
    plt.title("Average Duration of Options")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("graphs/option_lengths.png")
    plt.close()

# ========== Option Frequency Plot ==========
def extract_option_counts(log_path):
    counts = Counter()
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Option" in line and "count" in line:
                parts = line.strip().split("count = ")
                count = int(parts[-1])
                option = int(line.strip().split("Option ")[1].split(":")[0])
                counts[option] += count
    return counts

def plot_option_frequency():
    option_counts = extract_option_counts("runs/OptionCriticMLP-fourrooms-default/logger.log")
    options = list(option_counts.keys())
    counts = [option_counts[o] for o in options]

    plt.figure(figsize=(8, 5))
    plt.bar(options, counts)
    plt.xlabel("Option Index")
    plt.ylabel("Times Selected")
    plt.title("Option Selection Frequency")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("graphs/option_frequency.png")
    plt.close()

# ========== Loss Curves ==========
def plot_loss_curves():
    df = pd.read_csv("runs/OptionCriticMLP-fourrooms-default/training_metrics.csv")
    df = df.dropna(subset=["actor_loss", "critic_loss"])
    window = 100
    df["actor_loss_smooth"] = df["actor_loss"].rolling(window).mean()
    df["critic_loss_smooth"] = df["critic_loss"].rolling(window).mean()

    plt.figure(figsize=(12, 5))
    plt.plot(df["steps"], df["actor_loss_smooth"], label="Actor Loss (smooth)")
    plt.plot(df["steps"], df["critic_loss_smooth"], label="Critic Loss (smooth)")
    plt.xlabel("Total Steps")
    plt.ylabel("Loss")
    plt.title("Option-Critic Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/loss_curves.png")
    plt.close()

# ========== Termination Probability Plot ==========
def extract_termination_probs(csv_path):
    df = pd.read_csv(csv_path)
    term_cols = [col for col in df.columns if col.startswith("beta_")]
    return df["steps"], df[term_cols]

def plot_termination_probs():
    steps, betas = extract_termination_probs("runs/OptionCriticMLP-fourrooms-default/training_metrics.csv")
    plt.figure(figsize=(10, 5))
    for col in betas.columns:
        if betas[col].notna().any():
            plt.plot(steps, smooth(betas[col]), label=col)
    plt.xlabel("Steps")
    plt.ylabel("β (termination probability)")
    plt.title("Termination Probability Trends")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/termination_probs.png")
    plt.close()

# ========== Reward CSV Extractor ==========
def extract_csv_rewards(csv_path, skip_header=True):
    rewards = []
    with open(csv_path, 'r') as f:
        if skip_header:
            next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    rewards.append(float(parts[1]))
                except ValueError:
                    continue
    return rewards

# ========== Comparison Plot ==========
def plot_comparison_curve():
    oc_rewards = extract_rewards("runs/OptionCriticMLP-fourrooms-default/logger.log")
    ppo_rewards = extract_csv_rewards("ppo_logs/fourrooms_rewards.csv")
    dqn_rewards = extract_csv_rewards("dqn_logs/fourrooms_rewards.csv")

    plt.figure(figsize=(10, 5))
    plt.plot(smooth(oc_rewards, 100), label="Option-Critic")
    plt.plot(smooth(ppo_rewards, 100), label="PPO")
    plt.plot(smooth(dqn_rewards, 100), label="DQN")
    plt.title("Reward Comparison: Option-Critic vs PPO vs DQN")
    plt.xlabel("Episodes")
    plt.ylabel("Smoothed Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/comparison_reward_curve_all.png")
    plt.show()

# ========== Run All Plots ==========
if __name__ == "__main__":
    plot_reward_curve()
    plot_entropy()
    plot_loss_curves()
    plot_option_lengths()
    plot_option_frequency()
    plot_termination_probs()

    ppo_exists = os.path.exists("ppo_logs/fourrooms_rewards.csv")
    dqn_exists = os.path.exists("dqn_logs/fourrooms_rewards.csv")

    if ppo_exists and dqn_exists:
        plot_comparison_curve()
    else:
        if not ppo_exists:
            print("Missing PPO log: ppo_logs/fourrooms_rewards.csv")
        if not dqn_exists:
            print("Missing DQN log: dqn_logs/fourrooms_rewards.csv")
