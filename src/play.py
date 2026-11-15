"""Watch the trained DQN agent play Air Raid."""
import sys
import gymnasium as gym
import torch

from train import QNetwork

model_path = sys.argv[1] if len(sys.argv) > 1 else "dqn_model.pth"

env = gym.make("ALE/AirRaid-v5", frameskip=1, render_mode = "human")
env = gym.wrappers.AtariPreprocessing(
    env, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=False
)

env = gym.wrappers.FrameStackObservation(env, stack_size=4)

model = QNetwork(input_channels=4, num_actions=env.action_space.n)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

for episode in range (1, 4):
    state, _ = env.reset()
    total_reward = 0 
    done = False

    while not done:
        state_tensor = torch.as_tensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)

        action = q_values.argmax().item()

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated


    print(f"Episode {episode} : reward = {total_reward}")


env.close()

