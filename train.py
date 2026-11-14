"""Train a DQN agent to play Atari Air Raid.

Example:
    uv run python train.py --member Amaliza --exp 1 --lr 0.00025
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import ale_py
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # no display is needed (works on Colab and servers)
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from replay_buffer import ReplayBuffer

# Register Atari environments with Gymnasium before calling gym.make.
gym.register_envs(ale_py)

# Fixed for every experiment, so only the six tuned hyperparameters change the result.
REPLAY_CAPACITY = 50_000
LEARNING_STARTS = 5_000       # collect this many transitions before learning
TRAIN_FREQUENCY = 4           # one gradient update every 4 environment steps
TARGET_UPDATE_STEPS = 2_000   # copy online weights to the target network this often
MOVING_AVERAGE_WINDOW = 20    # episodes used to judge "best" performance


class QNetwork(nn.Module):
    """CNN mapping four stacked Atari frames to one Q-value per action."""

    def __init__(self, input_channels: int, num_actions: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.value = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Pixel observations are uint8 in [0, 255]; normalize for the network.
        state = state.float() / 255.0
        return self.value(self.features(state))


def parse_args():
    """Defaults are the baseline configuration; each experiment overrides a few."""
    parser = argparse.ArgumentParser(description="Train a DQN agent on Air Raid")
    parser.add_argument("--member", required=True, help="Member name (output folder)")
    parser.add_argument("--exp", type=int, required=True, help="Experiment number, 1-10")
    parser.add_argument("--output-dir", default=None, help="Override the output folder")
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    # The six tuned hyperparameters.
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument(
        "--eps-decay",
        type=int,
        default=100_000,
        help="Time constant (in steps) of the exponential epsilon decay",
    )
    return parser.parse_args()


def initialize(config):
    """Create the preprocessed environment, networks, and replay buffer."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = gym.make("ALE/AirRaid-v5", frameskip=1)
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    env.action_space.seed(config.seed)

    num_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    online_network = QNetwork(input_channels=4, num_actions=num_actions).to(device)
    target_network = QNetwork(input_channels=4, num_actions=num_actions).to(device)
    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()

    replay_buffer = ReplayBuffer(capacity=REPLAY_CAPACITY)
    return env, online_network, target_network, replay_buffer, device


def get_epsilon(total_steps, config):
    """Exponential decay gradually shifts the agent from exploring to exploiting."""
    decay = math.exp(-total_steps / config.eps_decay)
    return config.eps_end + (config.eps_start - config.eps_end) * decay


def choose_action(state, epsilon, env, online_network, device):
    """Choose a random action for exploration, otherwise the best Q action."""
    if random.random() < epsilon:
        return env.action_space.sample()

    state_tensor = torch.as_tensor(state, dtype=torch.uint8, device=device).unsqueeze(0)
    with torch.no_grad():
        return online_network(state_tensor).argmax(dim=1).item()


def update_network(replay_buffer, online_network, target_network, optimizer, device, config):
    """Use one random batch to make a single DQN gradient update."""
    states, actions, rewards, next_states, dones = replay_buffer.sample(config.batch_size, device)

    # Q(s, a): predicted value for the action the agent actually took.
    predicted_q_values = online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Bellman target: reward plus discounted best next-state value.
    with torch.no_grad():
        next_q_values = target_network(next_states).max(dim=1).values
        target_q_values = rewards + config.gamma * next_q_values * (1.0 - dones)

    loss = nn.functional.smooth_l1_loss(predicted_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_network.parameters(), max_norm=10.0)
    optimizer.step()


def recent_average(history):
    """Mean game score of the most recent episodes."""
    return float(np.mean([row["reward"] for row in history[-MOVING_AVERAGE_WINDOW:]]))


def save_plot(history, path):
    """Plot reward, episode length and epsilon against total environment steps."""
    steps = [row["total_steps"] for row in history]
    rewards = [row["reward"] for row in history]
    smoothed = [
        np.mean(rewards[max(0, i - MOVING_AVERAGE_WINDOW + 1): i + 1])
        for i in range(len(rewards))
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(steps, rewards, alpha=0.3, label="episode reward")
    axes[0].plot(steps, smoothed, label=f"{MOVING_AVERAGE_WINDOW}-episode average")
    axes[0].set(title="Reward", xlabel="Total steps", ylabel="Reward")
    axes[0].legend()
    axes[1].plot(steps, [row["length"] for row in history])
    axes[1].set(title="Episode length", xlabel="Total steps", ylabel="Steps in episode")
    axes[2].plot(steps, [row["epsilon"] for row in history])
    axes[2].set(title="Epsilon", xlabel="Total steps", ylabel="Epsilon")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_results(run_dir, history, config, best_average, minutes):
    """Write metrics.csv, plot.png and summary.json (summary is written last)."""
    with open(run_dir / "metrics.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    save_plot(history, run_dir / "plot.png")

    best_episode = max(history, key=lambda row: row["reward"])
    summary = {
        "config": vars(config),
        "best_avg_reward": best_average,
        "last_avg_reward": recent_average(history),
        "best_episode_reward": best_episode["reward"],
        "best_episode_length": best_episode["length"],
        "best_episode_total_steps": best_episode["total_steps"],
        "episodes": len(history),
        "minutes": round(minutes, 1),
    }
    # The existence of summary.json means "this run finished".
    with open(run_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=2)


def train(config):
    run_dir = Path(config.output_dir or f"{config.member}/exp{config.exp:02d}")
    run_dir.mkdir(parents=True, exist_ok=True)

    env, online_network, target_network, replay_buffer, device = initialize(config)
    optimizer = torch.optim.Adam(online_network.parameters(), lr=config.lr)

    history = []
    total_steps = 0
    best_average = float("-inf")
    start_time = time.time()

    try:
        while total_steps < config.total_steps:
            state, _ = env.reset(seed=config.seed if not history else None)
            episode_reward = 0.0
            episode_length = 0
            terminated = truncated = False

            while not (terminated or truncated):
                epsilon = get_epsilon(total_steps, config)
                action = choose_action(state, epsilon, env, online_network, device)
                next_state, reward, terminated, truncated, _ = env.step(action)

                # Store only the reward's sign so large Air Raid scores do not destabilise
                # learning. Only a real game over (terminated) stops bootstrapping.
                replay_buffer.add(state, action, float(np.sign(reward)), next_state, terminated)
                state = next_state
                episode_reward += reward  # the logged reward stays the real game score
                episode_length += 1
                total_steps += 1

                if len(replay_buffer) >= LEARNING_STARTS and total_steps % TRAIN_FREQUENCY == 0:
                    update_network(
                        replay_buffer, online_network, target_network, optimizer, device, config
                    )

                if total_steps % TARGET_UPDATE_STEPS == 0:
                    target_network.load_state_dict(online_network.state_dict())

            history.append(
                {
                    "episode": len(history) + 1,
                    "total_steps": total_steps,
                    "length": episode_length,
                    "reward": episode_reward,
                    "epsilon": round(epsilon, 4),
                }
            )
            average = recent_average(history)
            print(
                f"Episode {len(history)} | steps: {total_steps}/{config.total_steps} | "
                f"length: {episode_length} | reward: {episode_reward:.0f} | "
                f"avg{MOVING_AVERAGE_WINDOW}: {average:.0f} | epsilon: {epsilon:.3f}"
            )

            # Keep the best *averaged* model rather than one lucky episode. The average is
            # only trusted once a full window of episodes has been played.
            if len(history) >= MOVING_AVERAGE_WINDOW and average > best_average:
                best_average = average
                torch.save(online_network.state_dict(), run_dir / "dqn_model.pth")
    finally:
        env.close()

    if best_average == float("-inf"):  # fewer episodes than the window: save the final model
        best_average = recent_average(history)
        torch.save(online_network.state_dict(), run_dir / "dqn_model.pth")

    minutes = (time.time() - start_time) / 60
    save_results(run_dir, history, config, best_average, minutes)
    print(f"Done in {minutes:.1f} min. Best {MOVING_AVERAGE_WINDOW}-episode average: {best_average:.0f}")
    print(f"Results saved to {run_dir}/")


if __name__ == "__main__":
    train(parse_args())