from typing import Mapping, Any, NamedTuple

import torch

from .shared_base_buffer import SharedBaseBuffer, ResetStrategy
from .replay_buffer import ReplayBatch
from .stats_tracker import StatsTracker


class SharedStates(NamedTuple):
    pos: torch.Tensor
    full: torch.Tensor


class SharedMemoryComponents(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    truncateds: torch.Tensor
    next_observations: torch.Tensor


class SharedReplayBuffer(SharedBaseBuffer):
    """
    SharedReplayBuffer is a high-performance, multiprocessing-compatible replay buffer designed for reinforcement learning (RL) with parallel environments, especially in distributed data parallel (DDP) settings. All buffer operations are performed on GPU (or specified device) for maximum efficiency, and the buffer supports sharing memory across multiple processes for scalable RL training.

    Key Features:
    -------------
    - Supports parallel environments: Stores transitions for multiple environments running in parallel, enabling efficient experience collection in vectorized or distributed RL setups.
    - Multiprocessing safe: Utilizes shared memory tensors and locks to allow safe concurrent access from multiple processes (main and worker processes).
    - Device flexibility: All tensors can be allocated on CPU or GPU, with efficient device transfers for sampling and storage.
    - Episode statistics tracking: Integrates with a StatsTracker to record episode-level metrics and statistics.
    - Efficient sampling: Provides fast, random sampling of transitions for off-policy RL algorithms.
    - Handles truncated and done flags: Supports both environment terminations (done) and truncations (e.g., time limits), which is important for correct RL training.

    Usage Pattern:
    --------------
    - The main process creates the buffer and owns the shared memory tensors and lock.
    - Worker processes receive references to the shared tensors and lock via get_shared_components(), allowing them to add or sample transitions safely.
    - Transitions are added with add(), which updates the buffer and episode statistics.
    - Batches of transitions are sampled with get(), supporting efficient training.

    Parameters:
    -----------
    buffer_size (int): Maximum number of time steps (transitions) to store per environment.
    obs_shape (tuple[int, ...]): Shape of the observation space.
    action_shape (tuple[int, ...]): Shape of the action space.
    stats_tracker (StatsTracker): Tracks episode statistics and metrics.
    is_main_process (bool): Whether this instance is the main process (creates shared memory) or a worker (attaches to shared memory).
    shared_tensors (dict, optional): Shared memory tensors for worker processes.
    shared_lock (LockType, optional): Shared lock for synchronizing access.
    obs_dtype (torch.dtype): Data type for observations.
    action_dtype (torch.dtype): Data type for actions.
    device (torch.device): Device for sampling output tensors.
    store_device (torch.device, optional): Device for storing buffer tensors.
    reset_strategy (ResetStrategy): Strategy for resetting buffer state.

    -----------
    _observations (torch.Tensor): Shared tensor for observations.
    _actions (torch.Tensor): Shared tensor for actions.
    _rewards (torch.Tensor): Shared tensor for rewards.
    _dones (torch.Tensor): Shared tensor for done flags.
    _truncateds (torch.Tensor): Shared tensor for truncated flags.
    _next_observations (torch.Tensor): Shared tensor for next observations.
    _lock (LockType): Lock for synchronizing buffer access.

    Methods:
    --------
    get_shared_components() -> tuple[dict, LockType]:
        Returns the shared tensors and lock for worker processes to attach to the buffer.

    add(obs, action, reward, done, truncated, next_obs, infos, done_reasons):
        Adds a transition to the buffer and updates episode statistics. Must be called by the main process.

    get(batch_size) -> ReplayBatch:
        Samples a batch of transitions from the buffer for training.

    -------
    ValueError: If required shared components are missing or if sampling from an empty buffer.

    Typical Use Case:
    -----------------
    - In distributed RL, the main process creates the buffer and shares its components with worker processes.
    - Workers add transitions concurrently, and the main process (or any process) can sample batches for training.
    - All operations are protected by a lock to ensure data consistency.
    """

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        stats_tracker: StatsTracker,
        is_main_process: bool,
        shared_memory_components: SharedMemoryComponents | None = None,
        shared_states: SharedStates | None = None,
        obs_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        store_device: torch.device | None = None,
        reset_strategy: ResetStrategy = ResetStrategy.RECURRENT,
    ) -> None:

        # --- Buffer Parameters ---
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        # --- Buffer States ---
        if is_main_process:
            if shared_memory_components is not None or shared_states is not None:
                raise ValueError(
                    "Main process should not receive shared_memory_components or shared_states"
                )

            # -- Base-Inheritance ---
            super().__init__(
                buffer_size=buffer_size,
                num_envs=num_envs,
                stats_tracker=stats_tracker,
                device=device,
                store_device=store_device,
                reset_strategy=reset_strategy,
            )

            self._pos_tensor = torch.tensor([0], dtype=torch.long).share_memory_()
            self._full_tensor = torch.tensor([0], dtype=torch.bool).share_memory_()

            self._observations = torch.zeros(
                (buffer_size, num_envs, *obs_shape),
                dtype=obs_dtype,
                device=store_device,
            ).share_memory_()
            self._actions = torch.zeros(
                (buffer_size, num_envs, *action_shape),
                dtype=action_dtype,
                device=store_device,
            ).share_memory_()
            self._rewards = torch.zeros(
                (buffer_size, num_envs), dtype=torch.float32, device=store_device
            ).share_memory_()
            self._dones = torch.zeros(
                (buffer_size, num_envs), dtype=torch.bool, device=store_device
            ).share_memory_()
            self._truncateds = torch.zeros(
                (buffer_size, num_envs), dtype=torch.bool, device=store_device
            ).share_memory_()
            self._next_observations = torch.zeros(
                (buffer_size, num_envs, *obs_shape),
                dtype=torch.float32,
                device=store_device,
            ).share_memory_()

        else:
            if shared_memory_components is None or shared_states is None:
                raise ValueError(
                    "Worker process must receive shared_memory_components and shared_states"
                )

            # -- Base-Inheritance ---
            super().__init__(
                buffer_size=buffer_size,
                num_envs=num_envs,
                stats_tracker=None,
                device=device,
                store_device=store_device,
                reset_strategy=reset_strategy,
            )

            self._pos_tensor = shared_states.pos
            self._full_tensor = shared_states.full

            self._observations = shared_memory_components.observations
            self._actions = shared_memory_components.actions
            self._rewards = shared_memory_components.rewards
            self._dones = shared_memory_components.dones
            self._truncateds = shared_memory_components.truncateds
            self._next_observations = shared_memory_components.next_observations

    def get_shared_components(self) -> tuple[SharedMemoryComponents, SharedStates]:
        """
        Main process uses this to share buffer components with worker processes.
        Returns:
            tuple: (SharedMemoryComponents, SharedStates)
        """
        return SharedMemoryComponents(
            observations=self._observations,
            actions=self._actions,
            rewards=self._rewards,
            dones=self._dones,
            truncateds=self._truncateds,
            next_observations=self._next_observations,
        ), SharedStates(
            pos=self._pos_tensor,
            full=self._full_tensor,
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
        Must be called by Main Process only.

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
        # 使用锁来保护对 _pos 和 _full 的读取，防止在读取时被其他进程修改
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

    def pos(self) -> int:
        return int(self._pos_tensor[0].item())

    def set_pos(self, value: int) -> None:
        self._pos_tensor[0] = value

    def full(self) -> bool:
        return bool(self._full_tensor[0].item())

    def set_full(self, value: bool) -> None:
        self._full_tensor[0] = value
