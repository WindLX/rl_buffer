# RL-Buffer

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

[English](README.md) | [中文](README_zh-cn.md) 

**`rl-buffer`** 是一个为现代强化学习算法设计的高性能、高灵活性、功能丰富的 PyTorch 缓冲库。它提供了专为 Off-Policy 和 On-Policy 算法优化的核心组件，并集成了强大的统计跟踪功能，旨在加速您的RL研究和开发流程。

## ✨ 核心特性

-   **高性能**: 所有缓冲操作都基于 `torch.Tensor`，无缝支持 GPU 加速，最大限度地减少了数据在 CPU 和 GPU 之间的传输。
-   **两种核心缓冲**:
    -   `ReplayBuffer`: 为 DQN, SAC, DDPG 等 Off-Policy 算法设计的标准经验回放缓冲。
    -   `RolloutBuffer`: 为 PPO, A2C 等 On-Policy 算法设计，内置 **GAE (Generalized Advantage Estimation)** 计算。
-   **高灵活性**:
    -   **自定义数据类型**: 支持为观测 (`obs`) 和动作 (`action`) 指定 `dtype`（例如，`torch.uint8` 用于图像观测），有效优化内存/显存占用。
    -   **多环境并行**: 原生支持与多个并行环境（`num_envs`）的交互。
-   **强大的统计与诊断**:
    -   内置 `StatsTracker`，可轻松跟踪和记录回合的奖励、长度、自定义指标（如游戏得分）以及终止原因。
    -   自动计算均值、标准差、最大/最小值等统计数据，便于日志记录和分析。
-   **现代化 API**:
    -   `RolloutBuffer` 的 `get()` 方法返回一个**生成器**，实现了高效的内存利用，特别适合于大型缓冲和多次迭代（epoch）的训练。
    -   清晰、经过单元测试的接口，易于集成到现有项目中。

## 🚀 安装

本项目使用 `uv` 进行包管理。

1.  **创建并激活虚拟环境**:
    ```bash
    uv venv
    source .venv/bin/activate
    ```

2.  **安装项目及其依赖**:
    要进行开发和运行测试，请安装 `dev` 依赖项。
    ```bash
    uv pip install -e ".[dev]"
    ```
    如果只用于项目集成，安装基础依赖即可：
    ```bash
    uv pip install -e .
    ```

## 💡 使用示例

### 示例 1: `ReplayBuffer` (用于 Off-Policy 算法)

这个例子模拟了一个典型的 Off-Policy 算法（如 SAC）的数据收集和采样过程。

```python
import torch
from rl_buffer.replay_buffer import ReplayBuffer
from rl_buffer.stats_tracker import StatsTracker

# 1. 环境和缓冲配置
NUM_ENVS = 4
BUFFER_SIZE = 1000
BATCH_SIZE = 64
OBS_SHAPE = (16,)
ACTION_SHAPE = (4,)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 初始化 StatsTracker 和 ReplayBuffer
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

# 3. 模拟数据收集循环
for step in range(BUFFER_SIZE):
    # 生成模拟数据
    obs = torch.randn(NUM_ENVS, *OBS_SHAPE)
    action = torch.randn(NUM_ENVS, *ACTION_SHAPE)
    reward = torch.randn(NUM_ENVS)
    # 模拟大约每 100 步有一个环境结束
    done = torch.rand(NUM_ENVS) < 0.01
    next_obs = torch.randn(NUM_ENVS, *OBS_SHAPE)

    replay_buffer.add(obs, action, reward, done, next_obs)

print(f"Buffer populated. Current size: {len(replay_buffer)}")

# 4. 从缓冲中采样一个批次
if len(replay_buffer) > BATCH_SIZE:
    print(f"\n--- Sampling a batch of size {BATCH_SIZE} ---")
    batch = replay_buffer.get(batch_size=BATCH_SIZE)
    print(f"Sampled observations shape: {batch.observations.shape}")
    print(f"Sampled actions shape: {batch.actions.shape}")
    print(f"Sampled rewards shape: {batch.rewards.shape}")

# 5. 查看统计结果
print("\n--- Episode Statistics ---")
stats = stats_tracker.get_statistics()
for key, value in stats.items():
    print(f"{key}: {value:.4f}")

```

### 示例 2: `RolloutBuffer` (用于 On-Policy 算法)

这个例子模拟了 On-Policy 算法（如 PPO）的完整流程：收集一个 rollout -> 计算 GAE -> 在多个 epoch 中迭代训练。

```python
import torch
from rl_buffer.rollout_buffer import RolloutBuffer
from rl_buffer.stats_tracker import StatsTracker

# 1. 环境和缓冲配置
NUM_ENVS = 4
BUFFER_SIZE = 256 # On-policy 中通常称为 n_steps
BATCH_SIZE = 64
N_EPOCHS = 4
OBS_SHAPE = (16,)
ACTION_SHAPE = (1,)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 初始化 StatsTracker 和 RolloutBuffer
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

# 3. 模拟数据收集 (一个 rollout)
for _ in range(BUFFER_SIZE):
    # 生成模拟数据 (包括 value 和 log_prob)
    obs = torch.randn(NUM_ENVS, *OBS_SHAPE)
    action = torch.randn(NUM_ENVS, *ACTION_SHAPE)
    reward = torch.randn(NUM_ENVS)
    done = torch.zeros(NUM_ENVS, dtype=torch.bool)
    value = torch.randn(NUM_ENVS)
    log_prob = torch.randn(NUM_ENVS)

    rollout_buffer.add(obs, action, reward, done, value, log_prob)

print("Rollout collection complete.")

# 4. 计算 Advantages 和 Returns
# 模拟获取最后一个状态的 value estimate
last_values = torch.randn(NUM_ENVS)
last_dones = torch.zeros(NUM_ENVS, dtype=torch.bool)
rollout_buffer.compute_returns_and_advantages(last_values, last_dones)

print("\n--- Simulating training loop for PPO ---")
# 5. 使用生成器进行多轮次训练
for epoch in range(N_EPOCHS):
    batch_count = 0
    # get() 方法返回一个生成器
    for batch in rollout_buffer.get(batch_size=BATCH_SIZE):
        # 在这里执行你的训练步骤...
        # train_on_batch(batch)
        batch_count += 1
    
    print(f"Epoch {epoch + 1}: Trained on {batch_count} mini-batches.")

# 6. 查看统计结果
print("\n--- Episode Statistics ---")
stats = stats_tracker.get_statistics()
if not stats:
    print("No complete episodes during this short rollout.")
else:
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

```

## 🛠️ 开发与测试

要运行项目附带的单元测试，请确保已安装 `dev` 依赖，然后执行：

```bash
pytest
```

## 展望

我们计划在未来继续增强 `rl-buffer` 的功能，包括：

-   [ ] **优先经验回放 (PER)**
-   [ ] **原生支持字典观测空间 (`Dict Spaces`)**
-   [ ] **支持循环策略 (Recurrent Policies) 的序列缓冲**

欢迎贡献和提出建议！

## 📄 许可证

本项目采用 [MIT License](LICENSE)。