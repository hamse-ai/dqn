"""Air Raid DQN: environment and Q-network initialization."""

import gymnasium as gym
import ale_py
import torch
from torch import nn

# Gymnasium 1.x requires ALE's environments to be registered explicitly.
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

        # With 4x84x84 input, the convolution layers produce 64x7x7 features.
        self.value = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Pixel observations are uint8 in [0, 255]; normalize for the network.
        state = state.float() / 255.0
        return self.value(self.features(state))


def initialize() -> tuple[gym.Env, QNetwork, QNetwork, torch.device]:
    """Create the preprocessed Air Raid environment and the two DQN networks."""
    # rgb_array lets Gymnasium capture frames without opening a game window.
    env = gym.make("ALE/AirRaid-v5", frameskip=1, render_mode="rgb_array")
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    # Save episode 0 and then every 50th episode under videos/air_raid/.
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="videos/air_raid",
        episode_trigger=lambda episode_id: episode_id % 50 == 0,
        name_prefix="air-raid",
    )

    # Determine model dimensions from the wrapped environment itself.
    _, height, width = env.observation_space.shape
    num_actions = env.action_space.n
    input_channels = 4
    assert (height, width) == (84, 84)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    online_network = QNetwork(input_channels, num_actions).to(device)
    target_network = QNetwork(input_channels, num_actions).to(device)
    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()

    return env, online_network, target_network, device


if __name__ == "__main__":
    env, online_network, target_network, device = initialize()
    print(f"Observation shape: {env.observation_space.shape}")
    print(f"Number of actions: {env.action_space.n}")
    print(f"Online network device: {device}")
    print(f"Q-values shape: {online_network(torch.zeros(1, 4, 84, 84, device=device)).shape}")
    env.close()
