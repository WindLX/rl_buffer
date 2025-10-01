from contextlib import contextmanager
from typing import Iterator, Mapping, Any
from enum import Enum

import torch

from .stats_tracker import StatsTracker


class ResetStrategy(Enum):
    ERROR = "error"
    RECURRENT = "recurrent"
    AUTO = "auto"


class BaseBuffer:
    """
    Base class for buffers.

    Attributes:
        _buffer_size (int): Maximum number of transitions to store in the buffer.
        _pos (int): Current position in the buffer for adding new transitions.
        _full (bool): Whether the buffer is full.
        _stats_tracker (StatsTracker): Tracker for episode statistics and metrics.

        num_envs (int): Number of parallel environments.
        device (torch.device): Device on which to store tensors (CPU or GPU).
    """

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        stats_tracker: StatsTracker,
        device: torch.device = torch.device("cpu"),
        store_device: torch.device | None = None,
        reset_strategy: ResetStrategy = ResetStrategy.ERROR,
    ) -> None:
        # --- Buffer States ---
        self._buffer_size = buffer_size

        # Current position to insert the next transition
        self._pos = 0
        self._full = False
        self.reset_strategy = reset_strategy

        self._stats_tracker = stats_tracker

        # --- Buffer Parameters ---
        self.num_envs = num_envs
        self.device = device
        self.store_device = store_device if store_device is not None else device

    def __len__(self) -> int:
        """
        Get the number of filled steps in the buffer.

        Returns:
            int: Number of filled steps.
        """
        return self.current_size

    def reset(self, reset_stats: bool = False) -> None:
        """
        Clear the buffer content and reset episode tracking.
        """
        self._pos = 0
        self._full = False

        if reset_stats:
            self._stats_tracker.reset()

    @contextmanager
    def add_context(
        self,
        done: torch.Tensor,
        reward: torch.Tensor,
        infos: Mapping[str, Any] = {},
        done_reasons: list[str | None] = [],
    ) -> Iterator[int]:
        """
        A context manager to handle the logic of updating the buffer position `_pos` and the `_full` flag.

        This context manager:
        1.  Provides the starting position for the new data.
        2.  Updates the internal position pointer `_pos` after the data is added.
        3.  Sets the `_full` flag if the buffer becomes full.

        Yields:
            int: The position in the buffer where the new data should be inserted.
        """
        if self.full:
            match self.reset_strategy:
                case ResetStrategy.ERROR:
                    raise RuntimeError("Buffer is full. Cannot add more transitions.")
                case ResetStrategy.RECURRENT:
                    self._pos = 0
                case ResetStrategy.AUTO:
                    self.reset(reset_stats=False)

        start_pos = self._pos
        try:
            # Update episode-wise statistics (convert to numpy for stats tracker)
            self.stats_tracker.update(
                dones=done.cpu().numpy(),
                rewards=reward.cpu().numpy(),
                infos=infos,
                done_reasons=done_reasons,
            )
            yield start_pos
        finally:
            self._pos += 1
            if self._pos >= self._buffer_size:
                self._full = True

    @property
    def full(self) -> bool:
        """
        Check if the buffer is full.

        Returns:
            bool: True if the buffer is full, False otherwise.
        """
        return self._full

    @property
    def capacity(self) -> int:
        """
        Get the capacity of the buffer.

        Returns:
            int: Number of steps the buffer can hold.
        """
        return self._buffer_size

    @property
    def total_capacity(self) -> int:
        """
        Get the total capacity of the buffer.

        Returns:
            int: Total number of steps the buffer can hold.
        """
        return self._buffer_size * self.num_envs

    @property
    def current_size(self) -> int:
        """
        Get the current size of the buffer.

        Returns:
            int: Number of filled steps in the buffer.
        """
        if self._full:
            return self._buffer_size
        else:
            return self._pos

    @property
    def total_current_size(self) -> int:
        """
        Get the total size of the buffer, which is the product of buffer size and number of environments.

        Returns:
            int: Total size of the buffer.
        """
        if self._full:
            return self._buffer_size * self.num_envs
        else:
            return self._pos * self.num_envs

    @property
    def stats_tracker(self) -> StatsTracker:
        """
        Get the stats tracker for episode statistics and metrics.

        Returns:
            StatsTracker: The stats tracker instance.
        """
        return self._stats_tracker
