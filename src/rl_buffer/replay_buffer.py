from dataclasses import dataclass
from typing import Mapping, Any

import torch

from .base_buffer import BaseBuffer, ResetStrategy
from .stats_tracker import StatsTracker


@dataclass
class ReplayBatch:
    """
    Data batch for training a policy.

    Attributes:
        obss (torch.Tensor): Observations, shape (batch_size, *obs_shape).
        actions (torch.Tensor): Actions taken, shape (batch_size, *action_shape).
        rewards (torch.Tensor): Rewards received, shape (batch_size,).
        dones (torch.Tensor): Done flags, shape (batch_size,).
        truncateds (torch.Tensor): Truncated flags, shape (batch_size,).
        next_obss (torch.Tensor): Next observations, shape (batch_size, *obs_shape).
    """

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    truncateds: torch.Tensor
    next_observations: torch.Tensor


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer for RL with parallel environments.
    All operations are performed on GPU for maximum performance.

    Attributes:
        _buffer_size (int): Maximum number of transitions to store in the buffer.
        _pos (int): Current position in the buffer for adding new transitions.
        _full (bool): Whether the buffer is full.
        _stats_tracker (StatsTracker): Tracker for episode statistics and metrics.

        num_envs (int): Number of parallel environments.
        device (torch.device): Device on which to store tensors (CPU or GPU).

        obs_shape (tuple): Shape of the observation space.
        action_shape (tuple): Shape of the action space.

        gamma (float): Discount factor for future rewards.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation (GAE).

        _observations (torch.Tensor): Tensor to store observations.
        _actions (torch.Tensor): Tensor to store actions.
        _rewards (torch.Tensor): Tensor to store rewards.
        _dones (torch.Tensor): Tensor to store done flags for each transition.
        _truncateds (torch.Tensor): Tensor to store truncated flags for each transition.
        _next_observations (torch.Tensor): Tensor to store next observations.
    """

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        stats_tracker: StatsTracker,
        obs_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        store_device: torch.device | None = None,
        reset_strategy: ResetStrategy = ResetStrategy.RECURRENT,
    ) -> None:
        # -- Base-Inheritance ---
        super().__init__(
            buffer_size=buffer_size,
            num_envs=num_envs,
            stats_tracker=stats_tracker,
            device=device,
            store_device=store_device,
            reset_strategy=reset_strategy,
        )

        # --- Buffer Parameters ---
        self.obs_shape = obs_shape
        self.action_shape = action_shape

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
        Add a transition to the buffer.
        Also updates episode-wise statistics.

        Args:
            obs: Observation at time t (torch.Tensor).
            action: Action taken at time t (torch.Tensor).
            reward: Reward received at time t (torch.Tensor).
            done: Done flag at time t (torch.Tensor).
            truncated: Truncated flag at time t (torch.Tensor).
            next_obs: Next observation at time t (torch.Tensor).
            done_reason: Optional list of done reasons.
            infos: Additional metrics to track per episode.

        Raises:
            RuntimeError: If the buffer is full and cannot accept more transitions.
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

    @torch.no_grad()
    def get(self, batch_size: int) -> ReplayBatch:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size: Size of each mini-batch.

        Returns:
            ReplayBatch: A batch of rollout data.
        """
        # buffer_size 是指时间步长
        upper_bound = self.current_size

        available_samples = self.total_current_size

        if available_samples == 0:
            raise ValueError("Cannot sample from empty buffer")

        if batch_size > available_samples:
            raise ValueError(
                f"Batch size {batch_size} is larger than available samples {available_samples}"
            )

        # 1. 采样时间步索引 (from [0, upper_bound-1])
        batch_inds = torch.randint(0, upper_bound, size=(batch_size,))

        # 2. 采样环境索引 (from [0, num_envs-1])
        env_indices = torch.randint(0, self.num_envs, size=(batch_size,))

        # 3. 根据二维索引提取数据
        obs = self._observations[batch_inds, env_indices]
        actions = self._actions[batch_inds, env_indices]
        rewards = self._rewards[batch_inds, env_indices]
        next_obs = self._next_observations[batch_inds, env_indices]
        dones = self._dones[batch_inds, env_indices]
        truncateds = self._truncateds[batch_inds, env_indices]

        return ReplayBatch(
            observations=obs.to(self.device),
            actions=actions.to(self.device),
            rewards=rewards.to(self.device),
            next_observations=next_obs.to(self.device),
            dones=dones.to(self.device),
            truncateds=truncateds.to(self.device),
        )
