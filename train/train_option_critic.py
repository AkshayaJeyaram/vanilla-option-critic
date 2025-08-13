import argparse
import torch
import numpy as np
from copy import deepcopy
import time
import os

from agents.option_critic import OptionCriticMLP, OptionCriticCNN
from buffers.experience_buffer import ExperienceBuffer
from learners.gradients import compute_actor_gradient, compute_critic_gradient
from utils.environment_utils import make_env, to_tensor
from utils.logging_utils import Logger


def train(args):
    env, is_atari = make_env(args.env)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    AgentClass = OptionCriticCNN if is_atari else OptionCriticMLP

    agent = AgentClass(
        input_dim=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        num_options=args.num_options,
        temperature=args.temp,
        eps_start=args.epsilon_initial,
        eps_min=args.epsilon_final,
        eps_decay=args.epsilon_decay_steps,
        eps_test=args.optimal_eps,
        device=device
    )
    target_agent = deepcopy(agent)
    optimizer = torch.optim.RMSprop(agent.parameters(), lr=args.learning_rate)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    replay_buffer = ExperienceBuffer(max_size=args.max_history, seed=args.seed)
    logger = Logger(logdir=args.logdir, run_name=f"{AgentClass.__name__}-{args.env}-{args.exp or 'default'}")

    total_steps = 0
    min_option_duration = args.min_option_duration

    if args.switch_goal:
        print(f"Current goal index {env.goal_index}")

    while total_steps < args.max_steps_total:
        obs = env.reset()
        state = agent.extract_features(to_tensor(obs))
        greedy_option = agent.select_greedy_option(state)
        # initial option selection via ε-greedy
        current_option = agent.select_option_epsilon_greedy(state)

        option_lengths = {opt: [] for opt in range(args.num_options)}

        if args.switch_goal and logger.episode_count == 1000:
            ckpt_path = f"models/vanilla_oc_{args.env}_seed{args.seed}_ep2000.pth"
            torch.save({'model_state_dict': agent.state_dict(), 'goal_state': env.goal_index}, ckpt_path)
            env.switch_goal()
            print(f"New goal {env.goal_index}")

        if args.switch_goal and logger.episode_count > 2000:
            ckpt_path = f"models/vanilla_oc_{args.env}_seed{args.seed}_ep4000.pth"
            torch.save({'model_state_dict': agent.state_dict(), 'goal_state': env.goal_index}, ckpt_path)
            break

        done = False
        ep_steps = 0
        rewards = 0
        curr_op_len = 0
        option_terminated = True

        actor_loss, critic_loss = None, None
        entropy = torch.tensor(0.0)

        while not done and ep_steps < args.max_steps_ep:
            # cache epsilon ONCE per step (don’t call again this iteration)
            epsilon = agent.epsilon

            # ε-greedy option switching when terminated AND min duration satisfied
            if option_terminated:
                if curr_op_len >= min_option_duration:
                    option_lengths[current_option].append(curr_op_len)
                    current_option = agent.select_option_epsilon_greedy(state)
                    curr_op_len = 0

            action, logp, entropy = agent.sample_action(state, current_option)
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.store(obs, current_option, reward, next_obs, done)
            rewards += reward

            if len(replay_buffer) > args.batch_size:
                actor_loss = compute_actor_gradient(
                    obs, current_option, logp, entropy, reward, done, next_obs,
                    agent, target_agent, args, total_steps
                )

                loss = actor_loss

                if total_steps % args.update_every == 0:
                    batch = replay_buffer.sample_batch(args.batch_size)
                    critic_loss = compute_critic_gradient(agent, target_agent, batch, args)
                    loss += critic_loss

                optimizer.zero_grad()
                loss.backward()
                # ✅ gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                optimizer.step()

                if total_steps % args.target_update_freq == 0:
                    target_agent.load_state_dict(agent.state_dict())

            state = agent.extract_features(to_tensor(next_obs))
            option_terminated, greedy_option = agent.should_terminate(state, current_option)

            total_steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs

            # compute beta_* for logging (use current state's β)
            with torch.no_grad():
                term_probs_now = agent.predict_termination_probs(state)  # (1, num_options)
            beta_log = {f"beta_{i}": float(term_probs_now[0, i].item()) for i in range(args.num_options)}

            # single logger.log_data call at END of step with float entropy + beta_*
            ent_val = float(entropy.item()) if hasattr(entropy, "item") else float(entropy)
            logger.log_data(
                total_steps,
                actor_loss,               # None on steps without updates is fine
                critic_loss,              # None on non-update steps is fine
                ent_val,
                epsilon,                  # cached; avoid calling agent.epsilon again
                **beta_log
            )

        logger.log_episode(total_steps, rewards, option_lengths, ep_steps, epsilon)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration for training an Option-Critic agent")

    # Environment setup and logging
    parser.add_argument("--env", type=str, default="fourrooms", help="Target environment name")
    parser.add_argument("--exp", type=str, default=None, help="Optional experiment label")
    parser.add_argument("--logdir", type=str, default="runs", help="Directory for logging training runs")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generation")
    parser.add_argument("--cuda", type=bool, default=True, help="Flag to enable CUDA acceleration")

    # Agent and learning parameters
    parser.add_argument("--num-options", type=int, default=2, help="Total number of discrete options")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards")
    parser.add_argument("--learning-rate", type=float, default=0.0007, help="Optimizer learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Size of minibatches for updates")
    parser.add_argument("--max-history", type=int, default=20000, help="Capacity of the replay buffer")

    # Epsilon-greedy exploration
    parser.add_argument("--epsilon-initial", type=float, default=1.0, help="Starting epsilon value")
    parser.add_argument("--epsilon-final", type=float, default=0.1, help="Minimum epsilon after decay")
    parser.add_argument("--epsilon-decay-steps", type=int, default=50000, help="Steps over which epsilon decays")

    # Termination and entropy settings
    parser.add_argument("--temp", type=float, default=1.5, help="Softmax temperature for action sampling")
    parser.add_argument("--termination-reg", type=float, default=0.012, help="Coefficient for termination loss regularization")
    parser.add_argument("--entropy-reg", type=float, default=0.015, help="Coefficient for entropy regularization")
    parser.add_argument("--entropy-decay", type=float, default=8e4, help="Exponential decay factor for entropy weight")
    parser.add_argument("--min-option-duration", type=int, default=3, help="Minimum number of steps before option can switch")
    parser.add_argument("--beta-entropy-coeff", type=float, default=0.0,
                        help="Entropy regularization on beta to prevent saturation (0 disables)")

    # Training control
    parser.add_argument("--max-steps-ep", type=int, default=16000, help="Maximum steps per episode")
    parser.add_argument("--max-steps-total", type=int, default=4_000_000, help="Total number of training steps")
    parser.add_argument("--update-every", type=int, default=6, help="Step interval for updating the model")
    parser.add_argument("--target-update-freq", type=int, default=400, help="Interval for syncing the target network")
    parser.add_argument("--switch-goal", type=bool, default=False, help="Enable switching goals during training")
    parser.add_argument("--optimal-eps", type=float, default=0.05, help="Epsilon used during evaluation/testing")

    args = parser.parse_args()
    train(args)