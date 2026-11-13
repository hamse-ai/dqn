"""A small uniform replay buffer for DQN experience tuples."""

import random
from collections import deque

import torch


class ReplayBuffer:
    """Store and randomly sample (state, action, reward, next_state, done)."""

    def __init__(self, capacity: int = 10_000):
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive")
        self.capacity = capacity
        self.transitions = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.transitions)

    def add(self, state, action: int, reward: float, next_state, done: bool) -> None:
        # Keep image observations compact, and copy them so later env steps
        # cannot change the states already stored in the buffer.
        state = torch.as_tensor(state, dtype=torch.uint8, device="cpu").clone()
        next_state = torch.as_tensor(next_state, dtype=torch.uint8, device="cpu").clone()
        self.transitions.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device):
        """Return a random batch as tensors on the requested device."""
        if batch_size > len(self.transitions):
            raise ValueError("Not enough transitions in the replay buffer to sample this batch")

        batch = random.sample(self.transitions, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states).to(device),
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.stack(next_states).to(device),
            torch.tensor(dones, dtype=torch.float32, device=device),
        )
