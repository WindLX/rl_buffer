# RL-Buffer

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

[English](README.md) | [中文](README_zh-cn.md) 

**`rl-buffer`** is a high-performance, flexible, and feature-rich PyTorch buffer library designed for modern reinforcement learning algorithms. It provides core components optimized for both Off-Policy and On-Policy algorithms, integrated with a powerful statistics tracker to accelerate your RL research and development workflow.

## ✨ Core Features

-   **High-Performance**: All buffer operations are based on `torch.Tensor`, seamlessly supporting GPU acceleration and minimizing data transfer between CPU and GPU.
-   **Two Core Buffers**:
    -   `ReplayBuffer`: A standard experience replay buffer designed for Off-Policy algorithms like DQN, SAC, and DDPG.
    -   `RolloutBuffer`: A rollout buffer designed for On-Policy algorithms like PPO and A2C, with built-in **GAE (Generalized Advantage Estimation)** computation.
-   **High Flexibility**:
    -   **Customizable Data Types**: Supports specifying `dtype` for observations (`obs`) and actions (`action`) (e.g., `torch.uint8` for image-based observations), effectively optimizing memory and VRAM usage.
    -   **Parallel Environment Support**: Natively supports interaction with multiple parallel environments (`num_envs`).
-   **Powerful Statistics & Diagnostics**:
    -   Built-in `StatsTracker` to easily track and log episodic rewards, lengths, custom metrics (like game scores), and termination reasons.
    -   Automatically calculates mean, standard deviation, min/max, and other statistics for easy logging and analysis.
-   **Modern API**:
    -   `RolloutBuffer`'s `get()` method returns a **generator**, enabling memory-efficient iteration, which is ideal for large buffers and multi-epoch training.
    -   Clean, unit-tested API that is easy to integrate into existing projects.

## 🚀 Installation

This project uses `uv` for package management.

1.  **Create and activate a virtual environment**:
    ```bash
    uv venv
    source .venv/bin/activate
    ```

2.  **Install the project and its dependencies**:
    For development and running tests, install the `dev` dependencies.
    ```bash
    uv pip install -e ".[dev]"
    ```
    For integration into another project, install the base package.
    ```bash
    uv pip install -e .
    ```

## 💡 Usage Examples

### Example 1: `ReplayBuffer` (for Off-Policy Algorithms)

This example simulates the data collection and sampling process of a typical Off-Policy algorithm (like SAC).

```python
import torch
from rl_buffer.replay_buffer import ReplayBuffer
from rl_buffer.stats_tracker import StatsTracker

# 1. Environment and buffer configuration
NUM_ENVS = 4
BUFFER_SIZE = 1000
BATCH_SIZE = 64
OBS_SHAPE = (16,)
ACTION_SHAPE = (4,)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Initialize StatsTracker and ReplayBuffer
stats_tracker = StatsTracker(num_envs=NUM_ENVS)
replay_buffer = ReplayBuffer(
    buffer_size=BUFFER_SIZE,
    num_envs=NUM_ENVS,
    obs_shape=OBS_SHAPE,
    action_shape=ACTION_SHAPE,
    stats_tracker=stats_tracker,
    device=DEVICE
)

print(f"--- Populating ReplayBuffer on {DEVICE} ---")

# 3. Simulate a data collection loop
for step in range(BUFFER_SIZE):
    # Generate dummy data
    obs = torch.randn(NUM_ENVS, *OBS_SHAPE)
    action = torch.randn(NUM_ENVS, *ACTION_SHAPE)
    reward = torch.randn(NUM_ENVS)
    # Simulate an environment terminating roughly every 100 steps
    done = torch.rand(NUM_ENVS) < 0.01
    truncated = torch.zeros(NUM_ENVS, dtype=torch.float32)
    next_obs = torch.randn(NUM_ENVS, *OBS_SHAPE)

    replay_buffer.add(obs, action, reward, done, next_obs, truncated)

print(f"Buffer populated. Current size: {len(replay_buffer)}")

# 4. Sample a batch from the buffer
if len(replay_buffer) > BATCH_SIZE:
    print(f"\n--- Sampling a batch of size {BATCH_SIZE} ---")
    batch = replay_buffer.get(batch_size=BATCH_SIZE)
    print(f"Sampled observations shape: {batch.observations.shape}")
    print(f"Sampled actions shape: {batch.actions.shape}")
    print(f"Sampled rewards shape: {batch.rewards.shape}")

# 5. View statistics
print("\n--- Episode Statistics ---")
stats = stats_tracker.get_statistics()
for key, value in stats.items():
    print(f"{key}: {value:.4f}")
```

### Example 2: `RolloutBuffer` (for On-Policy Algorithms)

This example simulates the complete workflow of an On-Policy algorithm (like PPO): collecting a rollout -> computing GAE -> iterating through the data for multiple training epochs.

```python
import torch
from rl_buffer.rollout_buffer import RolloutBuffer
from rl_buffer.stats_tracker import StatsTracker

# 1. Environment and buffer configuration
NUM_ENVS = 4
BUFFER_SIZE = 256  # Often called n_steps in on-policy settings
BATCH_SIZE = 64
N_EPOCHS = 4
OBS_SHAPE = (16,)
ACTION_SHAPE = (1,)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Initialize StatsTracker and RolloutBuffer
stats_tracker = StatsTracker(num_envs=NUM_ENVS)
rollout_buffer = RolloutBuffer(
    buffer_size=BUFFER_SIZE,
    num_envs=NUM_ENVS,
    obs_shape=OBS_SHAPE,
    action_shape=ACTION_SHAPE,
    stats_tracker=stats_tracker,
    device=DEVICE
)

print(f"--- Collecting one rollout of {BUFFER_SIZE} steps on {DEVICE} ---")

# 3. Simulate data collection (one full rollout)
for _ in range(BUFFER_SIZE):
    # Generate dummy data (including value and log_prob)
    obs = torch.randn(NUM_ENVS, *OBS_SHAPE)
    action = torch.randn(NUM_ENVS, *ACTION_SHAPE)
    reward = torch.randn(NUM_ENVS)
    done = torch.zeros(NUM_ENVS, dtype=torch.bool)
    value = torch.randn(NUM_ENVS)
    log_prob = torch.randn(NUM_ENVS)

    rollout_buffer.add(obs, action, reward, done, value, log_prob)

print("Rollout collection complete.")

# 4. Compute Advantages and Returns
# Simulate getting the value estimate for the last state
last_values = torch.randn(NUM_ENVS)
last_dones = torch.zeros(NUM_ENVS, dtype=torch.bool)
rollout_buffer.compute_returns_and_advantages(last_values, last_dones)

print("\n--- Simulating training loop for PPO ---")
# 5. Use the generator for multi-epoch training
for epoch in range(N_EPOCHS):
    batch_count = 0
    # The get() method returns a generator
    for batch in rollout_buffer.get(batch_size=BATCH_SIZE):
        # Your training step would go here...
        # train_on_batch(batch)
        batch_count += 1
    
    print(f"Epoch {epoch + 1}: Trained on {batch_count} mini-batches.")

# 6. View statistics
print("\n--- Episode Statistics ---")
stats = stats_tracker.get_statistics()
if not stats:
    print("No complete episodes during this short rollout.")
else:
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
```

## 🛠️ Development & Testing

To run the unit tests included with the project, ensure you have installed the `dev` dependencies, then execute:

```bash
pytest
```

## Roadmap

We plan to continue enhancing `rl-buffer` with new features in the future, including:

- [x] **Parallel Safe**
- [x] **Prioritized Experience Replay (PER)**
- [ ] **Native support for Dictionary Observation Spaces (`Dict Spaces`)**
- [ ] **Sequence buffers for Recurrent Policies**

Contributions and suggestions are welcome!

## 📄 License

This project is licensed under the [MIT License](LICENSE).