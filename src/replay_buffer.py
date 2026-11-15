"""A small uniform replay buffer for DQN experience tuples."""

import random

import torch


class ReplayBuffer:
    """Store and randomly sample (state, action, reward, next_state, done)."""

    def __init__(self, capacity: int = 50_000):
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive")
        self.capacity = capacity
        self.transitions = []
        self.position = 0  # where the next transition is written

    def __len__(self) -> int:
        return len(self.transitions)

    def add(self, state, action: int, reward: float, next_state, done: bool) -> None:
        # Keep frames compact (uint8) and copy them so later env steps cannot change them.
        state = torch.as_tensor(state, dtype=torch.uint8, device="cpu").clone()
        next_state = torch.as_tensor(next_state, dtype=torch.uint8, device="cpu").clone()
        transition = (state, action, reward, next_state, done)

        if len(self.transitions) < self.capacity:
            self.transitions.append(transition)
        else:
            self.transitions[self.position] = transition  # overwrite the oldest
        self.position = (self.position + 1) % self.capacity

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
