"""Train a DQN agent to play Atari Air Raid."""

import math
import random

import ale_py
import gymnasium as gym
import torch
from torch import nn

from replay_buffer import ReplayBuffer


# Hyperparameters for this initial experiment.
NUM_EPISODES = 500
REPLAY_CAPACITY = 10_000
BATCH_SIZE = 32
LEARNING_STARTS = 1_000
GAMMA = 0.99
LEARNING_RATE = 0.0001
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 100_000
TARGET_UPDATE_STEPS = 5_000

# Register Atari environments with Gymnasium before calling gym.make.
gym.register_envs(ale_py)


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


def initialize():
    """Create the preprocessed environment, networks, and replay buffer."""
    # rgb_array lets Gymnasium capture frames for RecordVideo.
    env = gym.make("ALE/AirRaid-v5", frameskip=1, render_mode="rgb_array")
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    # Save episode 0, then every 50th episode, under videos/air_raid/.
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="videos/air_raid",
        episode_trigger=lambda episode_id: episode_id % 50 == 0,
        name_prefix="air-raid",
    )

    num_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    online_network = QNetwork(input_channels=4, num_actions=num_actions).to(device)
    target_network = QNetwork(input_channels=4, num_actions=num_actions).to(device)
    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()

    replay_buffer = ReplayBuffer(capacity=REPLAY_CAPACITY)
    return env, online_network, target_network, replay_buffer, device


def choose_action(state, epsilon, env, online_network, device):
    """Choose a random action for exploration, otherwise the best Q action."""
    if random.random() < epsilon:
        return env.action_space.sample()

    state_tensor = torch.as_tensor(state, dtype=torch.uint8, device=device).unsqueeze(0)
    with torch.no_grad():
        return online_network(state_tensor).argmax(dim=1).item()


def update_network(replay_buffer, online_network, target_network, optimizer, device):
    """Use one random batch to make a single DQN gradient update."""
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE, device)

    # Q(s, a): predicted value for the action the agent actually took.
    predicted_q_values = online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Bellman target: reward plus discounted best next-state value.
    with torch.no_grad():
        next_q_values = target_network(next_states).max(dim=1).values
        target_q_values = rewards + GAMMA * next_q_values * (1.0 - dones)

    loss = nn.functional.smooth_l1_loss(predicted_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train():
    env, online_network, target_network, replay_buffer, device = initialize()
    optimizer = torch.optim.Adam(online_network.parameters(), lr=LEARNING_RATE)

    total_steps = 0
    best_reward = float("-inf")
    best_episode_length = 0
    best_total_steps = 0

    try:
        for episode in range(1, NUM_EPISODES + 1):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                # Exponential decay gradually shifts from exploration to learning.
                epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(
                    -total_steps / EPSILON_DECAY_STEPS
                )
                action = choose_action(state, epsilon, env, online_network, device)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                episode_length += 1
                total_steps += 1

                if len(replay_buffer) >= max(BATCH_SIZE, LEARNING_STARTS):
                    update_network(
                        replay_buffer,
                        online_network,
                        target_network,
                        optimizer,
                        device,
                    )

                if total_steps % TARGET_UPDATE_STEPS == 0:
                    target_network.load_state_dict(online_network.state_dict())

            print(
                f"Episode {episode}/{NUM_EPISODES} | "
                f"length: {episode_length} | total steps: {total_steps} | "
                f"reward: {episode_reward:.1f} | epsilon: {epsilon:.3f}"
            )

            if episode_reward > best_reward:
                best_reward = episode_reward
                best_episode_length = episode_length
                best_total_steps = total_steps
                torch.save(online_network.state_dict(), "dqn_model.pth")

    finally:
        env.close()

    print(
        f"Best episode: reward={best_reward:.1f}, "
        f"length={best_episode_length}, total steps={best_total_steps}"
    )
    print("Saved best model to dqn_model.pth")


if __name__ == "__main__":
    train()
