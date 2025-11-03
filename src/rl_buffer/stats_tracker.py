from typing import Deque, Mapping, Any, Set

from collections import deque
import numpy as np


class StatsTracker:
    """
    A dedicated class for tracking, processing, and reporting episode statistics
    during rollouts.

    It accumulates metrics for ongoing episodes (e.g., rewards, lengths) and,
    upon episode completion, stores them in a buffer to calculate moving averages.

    Attributes:
        category (str): The category name for logging (e.g., "Rollout").
        num_envs (int): The number of parallel environments.
        ep_info_buffer (Deque[Dict[str, float]]): A buffer to store info of completed episodes.
        metrics_keys (List[str]): A list of metric keys to track (e.g., 'rewards', 'lengths').
    """

    def __init__(
        self,
        num_envs: int,
        category: str = "Rollout",
        max_ep_info_buffer: int = 100,
        extra_metrics_keys: list[str] | None = None,
        done_reason_keys: list[str] | None = None,
    ) -> None:
        """
        Initializes the StatsTracker.

        Args:
            num_envs: The number of parallel environments.
            category: The category name for logging purposes.
            max_ep_info_buffer: The maximum number of completed episodes to store
                                for calculating average statistics.
            extra_metrics_keys: Keys for additional metrics to track from the env's `info` dict.
            done_reason_keys: A list of possible reasons for an episode to end.
        """
        self.category = category
        self.num_envs = num_envs
        self.ep_info_buffer: Deque[dict[str, float]] = deque(maxlen=max_ep_info_buffer)

        # Use a set for efficient handling and to avoid duplicates, then convert to list
        default_keys: Set[str] = {"rewards", "lengths"}
        extra_keys: Set[str] = set(extra_metrics_keys) if extra_metrics_keys else set()
        self.metrics_keys: list[str] = sorted(list(default_keys | extra_keys))

        self.done_reason_keys: list[str] = done_reason_keys or []

        # --- Refactored State Management ---
        # Use a dictionary to store current episode stats instead of dynamic attributes.
        self._current_episode_stats: dict[str, np.ndarray] = {
            key: np.zeros(self.num_envs, dtype=np.float32) for key in self.metrics_keys
        }

    def reset(self) -> None:
        """
        Resets all statistics, including current episode accumulators and the
        completed episode buffer.
        """
        for key in self.metrics_keys:
            self._current_episode_stats[key][:] = 0.0
        self.ep_info_buffer.clear()

    def update(
        self,
        dones: np.ndarray,
        rewards: np.ndarray,
        infos: Mapping[str, Any] = {},
        done_reasons: list[str | None] = [],
    ) -> None:
        """
        Updates the statistics with data from a single timestep.

        Args:
            dones: An array of done signals from the environments.
            rewards: An array of rewards from the environments.
            infos: A dictionary of info from the environments, where values are arrays
                   aligned with the env dimension.
            done_reasons: A list of strings explaining why each environment terminated.
        """
        # Ensure rewards are of shape (num_envs,)
        if rewards.ndim == 2:
            rewards = rewards.sum(axis=1)

        self._current_episode_stats["rewards"] += rewards
        self._current_episode_stats["lengths"] += 1

        # Accumulate extra metrics
        for key in self.metrics_keys:
            if key not in ["rewards", "lengths"] and key in infos:
                values = np.asarray(infos[key], dtype=np.float32)
                self._current_episode_stats[key] += values

        # Process environments that have finished
        for env_idx in np.where(dones)[0]:
            ep_info: dict[str, float] = {}

            # 1. Store the final accumulated stats for this episode
            for key in self.metrics_keys:
                ep_info[f"ep_{key}"] = float(self._current_episode_stats[key][env_idx])

            # 2. Store the done reason if provided
            if env_idx < len(done_reasons):
                reason = done_reasons[env_idx]
                if reason in self.done_reason_keys:
                    ep_info[f"ep_done_{reason}"] = 1.0

            self.ep_info_buffer.append(ep_info)

            # 3. Reset the stats for this environment
            for key in self.metrics_keys:
                self._current_episode_stats[key][env_idx] = 0.0

    def _compute_metric_stats(
        self, values: np.ndarray
    ) -> tuple[float, dict[str, float]]:
        """Helper function to compute mean, std, min, and max for a set of values."""
        if values.size == 0:
            return 0.0, {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        mean = float(np.mean(values))

        return mean, {
            "mean": mean,
            "std": float(np.std(values)) if values.size > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    def get_statistics(self) -> tuple[dict[str, float], dict[str, float]]:
        """
        Computes and returns the summary statistics for all completed episodes in the buffer.

        Returns:
            A dictionary containing mean, std, min, and max for rewards, lengths,
            and other tracked metrics, as well as the total number of completed episodes.
            Returns an empty dictionary if no episodes have been completed.
        """
        if not self.ep_info_buffer:
            return {}, {}

        stats: dict[str, float] = {}
        stats_detailed: dict[str, float] = {}

        # Compute stats for all primary metrics
        for key in self.metrics_keys:
            values = np.array([ep.get(f"ep_{key}", 0.0) for ep in self.ep_info_buffer])
            metric_means, metric_stats = self._compute_metric_stats(values)

            stats[f"{self.category}/{key}"] = metric_means
            for stat_name, stat_value in metric_stats.items():
                stats_detailed[f"{self.category}/{key}_{stat_name}"] = stat_value

        # Compute stats for done reasons
        for reason in self.done_reason_keys:
            # The rate is just the mean of the 0/1 indicator
            done_counts = np.array(
                [ep.get(f"ep_done_{reason}", 0.0) for ep in self.ep_info_buffer]
            )
            stats[f"{self.category}/Done/{reason}_rate"] = float(np.mean(done_counts))
            stats_detailed[f"{self.category}/Done/{reason}_rate"] = float(
                np.mean(done_counts)
            )

        # Compute stats for step-wise reward
        lengths = np.array([ep.get("ep_lengths", 1.0) for ep in self.ep_info_buffer])
        # Avoid division by zero
        valid_lengths = lengths[lengths > 0]

        # Also compute per-step stats for other metrics
        if valid_lengths.size > 0:
            for key in self.metrics_keys:
                if key in ["lengths"]:
                    continue  # rewards is handled, lengths is the divisor

                values = np.array(
                    [ep.get(f"ep_{key}", 0.0) for ep in self.ep_info_buffer]
                )

                step_values = values[lengths > 0] / valid_lengths
                step_means, step_metric_stats = self._compute_metric_stats(step_values)

                stats[f"{self.category}/Step/{key}"] = step_means
                for stat_name, stat_value in step_metric_stats.items():
                    stats_detailed[f"{self.category}/Step/{key}_{stat_name}"] = (
                        stat_value
                    )

        return stats, stats_detailed

    def get_raw_values(self) -> dict[str, np.ndarray]:
        """
        Returns the raw values of completed episode metrics for visualization.

        Returns:
            A dictionary mapping metric names to a numpy array of their raw values.
            Returns an empty dictionary if no episodes have been completed.
        """
        if not self.ep_info_buffer:
            return {}

        raw_values: dict[str, np.ndarray] = {}
        for key in self.metrics_keys:
            raw_values[f"{self.category}/{key}"] = np.array(
                [ep.get(f"ep_{key}", 0.0) for ep in self.ep_info_buffer]
            )

        lengths = raw_values.get(f"{self.category}/lengths", np.array([]))

        if lengths.size > 0:
            valid_mask = lengths > 0
            valid_lengths = lengths[valid_mask]

            if valid_lengths.size > 0:
                for key in self.metrics_keys:
                    if key == "lengths":
                        continue

                    metric_values = raw_values.get(f"{self.category}/{key}")
                    if metric_values is not None and metric_values.size == lengths.size:
                        raw_values[f"{self.category}/Step/{key}"] = (
                            metric_values[valid_mask] / valid_lengths
                        )

        return raw_values

    @property
    def trajectory_num(self) -> int:
        """Returns the number of completed episodes in the buffer."""
        return len(self.ep_info_buffer)
