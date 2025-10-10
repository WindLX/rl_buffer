from dataclasses import dataclass
import torch
import numpy as np
from typing import Mapping, Any

from .base_buffer import BaseBuffer, ResetStrategy
from .replay_buffer import ReplayBatch
from .stats_tracker import StatsTracker
from .utils.sum_tree import SumTree


@dataclass
class PrioritizedReplayBatch(ReplayBatch):
    """
    包含重要性采样权重和样本索引的数据批次。
    """

    indices: torch.Tensor
    weights: torch.Tensor


class PrioritizedReplayBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        stats_tracker: StatsTracker,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
        obs_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        store_device: torch.device | None = None,
        reset_strategy: ResetStrategy = ResetStrategy.RECURRENT,
    ):
        # -- Base-Inheritance ---
        super().__init__(
            buffer_size=buffer_size,
            num_envs=num_envs,
            stats_tracker=stats_tracker,
            device=device,
            store_device=store_device,
            reset_strategy=reset_strategy,
        )

        # --- PER 特有参数 ---
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        self.sum_tree = SumTree(self.total_capacity)
        self._max_priority = 1.0

        # --- Buffer States ---
        self._observations = torch.zeros(
            (buffer_size, num_envs, *obs_shape), dtype=obs_dtype, device=store_device
        )
        self._actions = torch.zeros(
            (buffer_size, num_envs, *action_shape),
            dtype=action_dtype,
            device=store_device,
        )
        self._rewards = torch.zeros(
            (buffer_size, num_envs), dtype=torch.float32, device=store_device
        )
        self._dones = torch.zeros(
            (buffer_size, num_envs), dtype=torch.bool, device=store_device
        )
        self._truncateds = torch.zeros(
            (buffer_size, num_envs), dtype=torch.bool, device=store_device
        )
        self._next_observations = torch.zeros(
            (buffer_size, num_envs, *obs_shape),
            dtype=torch.float32,
            device=store_device,
        )

    @torch.no_grad()
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        truncated: torch.Tensor,
        next_obs: torch.Tensor,
        infos: Mapping[str, Any] = {},
        done_reasons: list[str | None] = [],
    ) -> None:
        """
        将新的 transition 添加到 buffer 中，并赋予其当前最大优先级。
        """
        with self.add_context(done, reward, infos, done_reasons) as pos:
            # Ensure all inputs are on the correct device
            obs = obs.to(self.store_device)
            action = action.to(self.store_device)
            reward = reward.to(self.store_device)
            done = done.to(self.store_device)
            truncated = truncated.to(self.store_device)
            next_obs = next_obs.to(self.store_device)

            self._observations[pos] = obs
            self._actions[pos] = action
            self._rewards[pos] = reward
            self._dones[pos] = done
            self._truncateds[pos] = truncated
            self._next_observations[pos] = next_obs
            for i in range(self.num_envs):
                self.sum_tree.add(self._max_priority)

    @torch.no_grad()
    def get(self, batch_size: int, beta: float | None = None) -> PrioritizedReplayBatch:
        """
        根据优先级采样一批数据，并计算重要性采样权重。
        """
        if beta is None:
            beta = self.beta

        indices = np.zeros(batch_size, dtype=int)
        priorities = np.zeros(batch_size, dtype=float)

        # 1. 采样
        segment = self.sum_tree.total_priority / batch_size
        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, p = self.sum_tree.get(s)
            indices[i] = idx
            priorities[i] = p

        # 2. 计算重要性采样权重
        sampling_probs = priorities / self.sum_tree.total_priority
        # total_current_size 是当前 buffer 中实际的样本数
        weights = np.power(self.total_current_size * sampling_probs, -beta)
        # 归一化权重
        weights /= weights.max()

        # 3. 提取数据
        # 将一维索引转回二维 (time, env) 索引
        time_indices = indices // self.num_envs
        env_indices = indices % self.num_envs

        # 3. 根据二维索引提取数据
        obs = self._observations[time_indices, env_indices]
        actions = self._actions[time_indices, env_indices]
        rewards = self._rewards[time_indices, env_indices]
        next_obs = self._next_observations[time_indices, env_indices]
        dones = self._dones[time_indices, env_indices]
        truncateds = self._truncateds[time_indices, env_indices]

        return PrioritizedReplayBatch(
            observations=obs.to(self.device),
            actions=actions.to(self.device),
            rewards=rewards.to(self.device),
            next_observations=next_obs.to(self.device),
            dones=dones.to(self.device),
            truncateds=truncateds.to(self.device),
            indices=torch.from_numpy(indices).to(self.device),
            weights=torch.from_numpy(weights).float().to(self.device),
        )

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor) -> None:
        """
        在训练步骤之后，根据新的 TD-Error 更新样本的优先级。
        """
        if isinstance(indices, torch.Tensor):
            indices_np = indices.cpu().numpy()
        if isinstance(td_errors, torch.Tensor):
            td_errors_np = td_errors.cpu().numpy()

        priorities = (np.abs(td_errors_np) + self.epsilon) ** self.alpha

        for idx, p in zip(indices_np, priorities):
            self.sum_tree.update(idx, p)

        # 更新记录的最大优先级
        self._max_priority = max(self._max_priority, priorities.max())
