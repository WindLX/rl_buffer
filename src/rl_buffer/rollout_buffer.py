from typing import Generator
from dataclasses import dataclass

import torch

from .base_buffer import BaseBuffer
from .stats_tracker import StatsTracker


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = torch.var(y_true, unbiased=False)
    return (
        float("nan")
        if var_y == 0
        else float(1 - torch.var(y_true - y_pred, unbiased=False) / var_y)
    )


@dataclass
class RolloutBatch:
    """
    Data batch for training a policy.

    Attributes:
        obss (torch.Tensor): Observations, shape (batch_size, *obs_shape).
        actions (torch.Tensor): Actions taken, shape (batch_size, *action_shape).
        returns (torch.Tensor): Returns (advantages + values), shape (batch_size,).
        advantages (torch.Tensor): Advantages, shape (batch_size,).
        values (torch.Tensor): Value function estimates, shape (batch_size,).
        log_probs (torch.Tensor): Log probabilities of actions, shape (batch_size,).
    """

    observations: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor


class RolloutBuffer(BaseBuffer):
    """
    Generalized Advantage Estimation (GAE) rollout buffer for RL with parallel environments.
    All operations are performed on GPU for maximum performance.

    Attributes:
        obs_shape (tuple): Shape of the observation space.
        action_shape (tuple): Shape of the action space.

        gamma (float): Discount factor for future rewards.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation (GAE).

        _observations (torch.Tensor): Tensor to store observations.
        _actions (torch.Tensor): Tensor to store actions.
        _rewards (torch.Tensor): Tensor to store rewards.
        _dones (torch.Tensor): Tensor to store done flags for each transition.
        _values (torch.Tensor): Tensor to store value estimates.
        _log_probs (torch.Tensor): Tensor to store log probabilities of actions.
        _advantages (torch.Tensor): Tensor to store computed advantages.
        _returns (torch.Tensor): Tensor to store computed returns.
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
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        # -- Base-Inheritance ---
        super().__init__(
            buffer_size=buffer_size,
            num_envs=num_envs,
            stats_tracker=stats_tracker,
            device=device,
        )

        # --- Buffer Parameters ---
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # --- Buffer States ---
        self._observations = torch.zeros(
            (buffer_size, num_envs, *obs_shape), dtype=obs_dtype, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, num_envs, *action_shape), dtype=action_dtype, device=device
        )
        self._rewards = torch.zeros(
            (buffer_size, num_envs), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros(
            (buffer_size, num_envs), dtype=torch.bool, device=device
        )
        self._values = torch.zeros(
            (buffer_size, num_envs), dtype=torch.float32, device=device
        )
        self._log_probs = torch.zeros(
            (buffer_size, num_envs), dtype=torch.float32, device=device
        )

        self._advantages = torch.zeros(
            (buffer_size, num_envs), dtype=torch.float32, device=device
        )
        self._returns = torch.zeros(
            (buffer_size, num_envs), dtype=torch.float32, device=device
        )

    @torch.no_grad()
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done_reasons: list[str | None] | None = None,
        **infos,
    ) -> None:
        """
        Add a transition to the buffer.
        Also updates episode-wise statistics.

        Args:
            obs: Observation at time t (torch.Tensor).
            action: Action taken at time t (torch.Tensor).
            reward: Reward received at time t (torch.Tensor).
            done: Done flag at time t (torch.Tensor).
            value: Value function estimate at time t (torch.Tensor).
            log_prob: Log probability of the action at time t (torch.Tensor).
            done_reason: Optional list of done reasons.
            infos: Additional metrics to track per episode.

        Raises:
            RuntimeError: If the buffer is full and cannot accept more transitions.
        """
        with self.add_context(done, reward, infos, done_reasons) as pos:
            # Ensure all inputs are on the correct device
            obs = obs.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)
            done = done.to(self.device)
            value = value.to(self.device)
            log_prob = log_prob.to(self.device)

            self._observations[pos] = obs
            self._actions[pos] = action
            self._rewards[pos] = reward
            self._dones[pos] = done
            self._values[pos] = value
            self._log_probs[pos] = log_prob

    @torch.no_grad()
    def get(
        self, batch_size: int, shuffle: bool = True
    ) -> Generator[RolloutBatch, None, None]:
        """
        Yield mini-batches of rollout data for training.

        Args:
            batch_size: Size of each mini-batch.
            shuffle: Whether to shuffle data before yielding.

        Returns:
            Generator of RolloutBatch mini-batches.
        """
        total_steps = self._buffer_size * self.num_envs
        indices = torch.arange(total_steps, device=self.device)
        if shuffle:
            indices = indices[torch.randperm(total_steps, device=self.device)]

        # Flatten the tensors for easy batching
        obss = self._observations.reshape(total_steps, *self._observations.shape[2:])
        actions = self._actions.reshape(total_steps, *self._actions.shape[2:])
        values = self._values.reshape(total_steps)
        returns = self._returns.reshape(total_steps)
        advantages = self._advantages.reshape(total_steps)
        log_probs = self._log_probs.reshape(total_steps)

        for start in range(0, total_steps, batch_size):
            end = start + batch_size
            batch_inds = indices[start:end]
            yield RolloutBatch(
                observations=obss[batch_inds],  # shape: (batch_size, *obs_shape)
                actions=actions[batch_inds],  # shape: (batch_size, *action_shape)
                returns=returns[batch_inds],  # shape: (batch_size)
                advantages=advantages[batch_inds],  # shape: (batch_size,)
                values=values[batch_inds],  # shape: (batch_size,)
                log_probs=log_probs[batch_inds],  # shape: (batch_size,)
            )

    def compute_returns_and_advantages(
        self, last_values: torch.Tensor, last_dones: torch.Tensor
    ) -> None:
        """
        Compute GAE advantages and returns in a vectorized manner.

        This method calculates the advantages using Generalized Advantage Estimation (GAE)
        for each transition in the buffer. It operates backwards in time, starting
        from the last transition.

        Args:
            last_values (torch.Tensor): Value estimates for the final state of each env,
                                        used for bootstrapping. Shape (num_envs,).
            last_dones (torch.Tensor): Done flags for the final state of each env.
                                    Shape (num_envs,).
        """
        last_values = last_values.to(self.device).flatten()
        last_dones = last_dones.to(self.device).flatten()

        # The GAE calculation requires the value of the state *after* the last action.
        # If the episode terminated at the last step, the bootstrapped value should be 0.
        last_gae_lambda = torch.zeros(self.num_envs, device=self.device)

        # Iterate backwards through the buffer's timesteps
        for step in reversed(range(self._buffer_size)):
            # Determine the value and done flag for the *next* state (t+1)
            if step == self._buffer_size - 1:
                next_non_terminal = 1.0 - last_dones.float()
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self._dones[step + 1].float()
                next_values = self._values[step + 1]

            # Calculate the TD error (delta) for the current step t
            # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
            delta = (
                self._rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self._values[step]
            )

            # Update the GAE estimate using the recursive formula
            # A_t = delta_t + gamma * lambda * A_{t+1} * (1 - done_{t+1})
            last_gae_lambda = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
            )
            self._advantages[step] = last_gae_lambda

        # Once advantages are computed, returns are simply advantages + values
        self._returns = self._advantages + self._values

    def normalize_advantages(self) -> None:
        """
        Normalize advantages to have zero mean and unit variance.
        This is useful for stabilizing training.
        """
        advantages_slice = self._advantages[: self._pos]
        mean_adv = torch.mean(advantages_slice)
        std_adv = torch.std(advantages_slice, unbiased=False) + 1e-6
        self._advantages[: self._pos] = (advantages_slice - mean_adv) / std_adv

    def normalize_returns(self) -> None:
        """
        Normalize returns to have zero mean and unit variance.
        """
        returns_slice = self._returns[: self._pos]
        mean_return = torch.mean(returns_slice)
        std_return = torch.std(returns_slice, unbiased=False) + 1e-6
        self._returns[: self._pos] = (returns_slice - mean_return) / std_return

    @property
    def explained_variance(self) -> float:
        """
        Compute explained variance for the returns.
        """
        returns_flat = self._returns[: self._pos].flatten()
        values_flat = self._values[: self._pos].flatten()
        return explained_variance(returns_flat, values_flat)

    @property
    def advantages_mean(self) -> float:
        """
        Get the mean of the advantages.

        Returns:
            float: Mean of the advantages.
        """
        return float(torch.mean(self._advantages[: self._pos]))

    @property
    def advantages_std(self) -> float:
        """
        Get the standard deviation of the advantages.

        Returns:
            float: Standard deviation of the advantages.
        """
        advantages_slice = self._advantages[: self._pos]
        if advantages_slice.numel() <= 1:
            return 0.0
        return float(torch.std(advantages_slice, unbiased=True))

    @property
    def returns_mean(self) -> float:
        """
        Get the mean of the returns.

        Returns:
            float: Mean of the returns.
        """
        return float(torch.mean(self._returns[: self._pos]))

    @property
    def returns_std(self) -> float:
        """
        Get the standard deviation of the returns.

        Returns:
            float: Standard deviation of the returns.
        """
        returns_slice = self._returns[: self._pos]
        if returns_slice.numel() <= 1:
            return 0.0
        return float(torch.std(returns_slice, unbiased=True))
