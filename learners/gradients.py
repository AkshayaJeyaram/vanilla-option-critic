import torch
import numpy as np
from utils.environment_utils import to_tensor

def compute_critic_gradient(agent, target_agent, batch_data, args):
    observations, options, rewards, next_observations, dones = batch_data
    indices = torch.arange(len(options)).long()

    options = torch.LongTensor(options).to(agent.device)
    rewards = torch.FloatTensor(rewards).to(agent.device)
    not_done = 1 - torch.FloatTensor(dones).to(agent.device)

    current_features = agent.extract_features(to_tensor(observations)).to(agent.device)
    current_q_values = agent.compute_Q_values(current_features)

    next_features_target = target_agent.extract_features(to_tensor(next_observations)).to(agent.device)
    target_next_q_values = target_agent.compute_Q_values(next_features_target)

    # Use TARGET β for consistent targets
    term_probs_target = target_agent.predict_termination_probs(next_features_target).detach()
    term_prob_for_option = term_probs_target[indices, options]

    target_values = rewards + not_done * args.gamma * (
        (1 - term_prob_for_option) * target_next_q_values[indices, options] +
        term_prob_for_option * target_next_q_values.max(dim=-1)[0]
    )

    td_loss = 0.5 * (current_q_values[indices, options] - target_values.detach()).pow(2).mean()
    return td_loss


def compute_actor_gradient(obs, option, logp, entropy, reward, done, next_obs, agent, target_agent, args, total_steps):
    current_state = agent.extract_features(to_tensor(obs))
    next_state = agent.extract_features(to_tensor(next_obs))
    next_state_target = target_agent.extract_features(to_tensor(next_obs))

    current_term_prob = agent.predict_termination_probs(current_state)[:, option]
    next_term_prob = target_agent.predict_termination_probs(next_state_target)[:, option].detach()

    current_q = agent.compute_Q_values(current_state).detach().squeeze()
    future_q_target = target_agent.compute_Q_values(next_state_target).detach().squeeze()

    # Advantage for policy
    target_value = reward + (1 - done) * args.gamma * (
        (1 - next_term_prob) * future_q_target[option] +
        next_term_prob * future_q_target.max(dim=-1)[0]
    )
    advantage = target_value.detach() - current_q[option]

    # Entropy annealing for intra-option policy
    entropy_weight = args.entropy_reg * np.exp(-total_steps / args.entropy_decay)
    policy_loss = -logp * advantage - entropy_weight * entropy

    # Termination objective: encourage termination if Q(s,o) < V(s)
    termination_penalty = current_term_prob * (
        current_q[option] - current_q.max(dim=-1)[0] + args.termination_reg
    ) * (1 - done)

    # ✅ β-entropy regularizer (prevents β from saturating at 0/1)
    beta_entropy_coeff = getattr(args, "beta_entropy_coeff", 0.0)
    if beta_entropy_coeff > 0.0:
        beta = current_term_prob.clamp(1e-6, 1 - 1e-6)
        beta_entropy = -(beta * torch.log(beta) + (1 - beta) * torch.log(1 - beta))
        beta_reg = -beta_entropy_coeff * beta_entropy.mean()
    else:
        beta_reg = 0.0

    return termination_penalty + policy_loss + beta_reg