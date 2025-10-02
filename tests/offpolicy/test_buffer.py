import pytest

import torch
import numpy as np

from rl_buffer import ReplayBuffer, ReplayBatch, StatsTracker, ResetStrategy


@pytest.fixture
def setup_buffer() -> tuple[ReplayBuffer, StatsTracker]:
    """Fixture to create a common ReplayBuffer instance for tests."""
    buffer_size = 100
    num_envs = 4
    obs_shape = (8,)
    action_shape = (2,)
    device = torch.device("cpu")
    stats_tracker = StatsTracker(
        num_envs=num_envs,
        max_ep_info_buffer=50,
        extra_metrics_keys=["custom_metric"],
    )
    return (
        ReplayBuffer(
            buffer_size=buffer_size,
            num_envs=num_envs,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            stats_tracker=stats_tracker,
        ),
        stats_tracker,
    )


def generate_sample_data(num_envs: int, obs_shape: tuple, action_shape: tuple):
    """Generates a dictionary of sample data for a single step."""
    return {
        "obs": torch.randn(num_envs, *obs_shape, dtype=torch.float32),
        "action": torch.randn(num_envs, *action_shape, dtype=torch.float32),
        "reward": torch.randn(num_envs, dtype=torch.float32),
        "done": torch.randint(0, 2, (num_envs,), dtype=torch.float32),
        "truncated": torch.randint(0, 2, (num_envs,), dtype=torch.float32),
        "next_obs": torch.randn(num_envs, *obs_shape, dtype=torch.float32),
        "infos": {"custom_metric": torch.randn(num_envs, dtype=torch.float32)},
    }


def test_initialization(setup_buffer: tuple[ReplayBuffer, StatsTracker]) -> None:
    """Test if the buffer initializes correctly."""
    buffer, _ = setup_buffer
    assert buffer._buffer_size == 100
    assert buffer.num_envs == 4
    assert buffer.obs_shape == (8,)
    assert buffer.action_shape == (2,)
    assert buffer.device == torch.device("cpu")
    assert buffer.current_size == 0
    assert buffer.total_capacity == 100 * 4
    assert not buffer.full
    assert len(buffer) == 0

    # Test internal array shapes
    assert buffer._observations.shape == (100, 4, 8)
    assert buffer._actions.shape == (100, 4, 2)
    assert buffer._rewards.shape == (100, 4)
    assert buffer._next_observations.shape == (100, 4, 8)
    assert buffer._dones.shape == (100, 4)


def test_add_single_step(setup_buffer: tuple[ReplayBuffer, StatsTracker]) -> None:
    """Test adding a single transition."""
    buffer, _ = setup_buffer
    sample_data = generate_sample_data(
        buffer.num_envs, buffer.obs_shape, buffer.action_shape
    )

    buffer.add(**sample_data, done_reasons=[])

    assert buffer.current_size == 1
    assert not buffer.full
    np.testing.assert_array_equal(buffer._observations[0], sample_data["obs"])
    np.testing.assert_array_equal(buffer._actions[0], sample_data["action"])
    np.testing.assert_array_equal(buffer._rewards[0], sample_data["reward"])
    np.testing.assert_array_equal(buffer._next_observations[0], sample_data["next_obs"])
    np.testing.assert_array_equal(buffer._dones[0], sample_data["done"])


def test_buffer_overflow(setup_buffer: tuple[ReplayBuffer, StatsTracker]) -> None:
    """Test buffer overflow and circular writing."""
    buffer, _ = setup_buffer
    buffer.reset_strategy = ResetStrategy.ERROR
    sample_data = generate_sample_data(
        buffer.num_envs, buffer.obs_shape, buffer.action_shape
    )

    # Fill the buffer
    for _ in range(buffer._buffer_size):
        buffer.add(**sample_data, done_reasons=[])

    assert buffer.current_size == buffer._buffer_size
    assert buffer.full

    # Add one more step to trigger overflow
    new_obs = torch.randn(buffer.num_envs, *buffer.obs_shape, dtype=torch.float32)
    sample_data["obs"] = new_obs
    with pytest.raises(RuntimeError):
        buffer.add(**sample_data, done_reasons=[])


def test_buffer_overflow_recurrent(
    setup_buffer: tuple[ReplayBuffer, StatsTracker],
) -> None:
    """Test buffer overflow and circular writing with Recurrent reset strategy."""
    buffer, _ = setup_buffer
    buffer.reset_strategy = ResetStrategy.RECURRENT
    sample_data = generate_sample_data(
        buffer.num_envs, buffer.obs_shape, buffer.action_shape
    )

    # Fill the buffer
    for _ in range(buffer._buffer_size):
        buffer.add(**sample_data, done_reasons=[])

    assert buffer.current_size == buffer._buffer_size
    assert buffer.full

    # Add one more step to trigger overflow and reset position
    new_obs = torch.randn(buffer.num_envs, *buffer.obs_shape, dtype=torch.float32)
    sample_data["obs"] = new_obs
    buffer.add(**sample_data, done_reasons=[])

    assert buffer.current_size == buffer._buffer_size
    assert buffer.full
    np.testing.assert_array_equal(buffer._observations[0], new_obs)


def test_get_batch(setup_buffer: tuple[ReplayBuffer, StatsTracker]) -> None:
    """Test sampling a batch of transitions."""
    buffer, _ = setup_buffer
    sample_data = generate_sample_data(
        buffer.num_envs, buffer.obs_shape, buffer.action_shape
    )

    # Add some data
    for _ in range(20):
        buffer.add(**sample_data, done_reasons=[])

    batch_size = 32
    batch = buffer.get(batch_size)

    assert isinstance(batch, ReplayBatch)
    assert batch.observations.shape == (batch_size, *buffer.obs_shape)
    assert batch.actions.shape == (batch_size, *buffer.action_shape)
    assert batch.rewards.shape == (batch_size,)
    assert batch.next_observations.shape == (batch_size, *buffer.obs_shape)
    assert batch.dones.shape == (batch_size,)

    # Check device and dtype
    assert batch.observations.device == buffer.device
    assert batch.observations.dtype == torch.float32


def test_reset_buffer(setup_buffer: tuple[ReplayBuffer, StatsTracker]) -> None:
    """Test resetting the buffer."""
    buffer, _ = setup_buffer
    sample_data = generate_sample_data(
        buffer.num_envs, buffer.obs_shape, buffer.action_shape
    )

    # Add some data
    for _ in range(10):
        buffer.add(**sample_data, done_reasons=[])

    assert buffer.current_size == 10
    buffer.reset()
    assert buffer.current_size == 0
    assert not buffer.full
    assert len(buffer) == 0


def test_empty_buffer_sampling(setup_buffer: tuple[ReplayBuffer, StatsTracker]) -> None:
    """Test that sampling from an empty buffer raises an error."""
    buffer, _ = setup_buffer
    with pytest.raises(ValueError, match="Cannot sample from empty buffer"):
        buffer.get(10)


def test_batch_size_larger_than_buffer(
    setup_buffer: tuple[ReplayBuffer, StatsTracker],
) -> None:
    """Test that sampling more than available raises an error."""
    buffer, _ = setup_buffer
    sample_data = generate_sample_data(
        buffer.num_envs, buffer.obs_shape, buffer.action_shape
    )

    # Add only one step (4 samples)
    buffer.add(**sample_data, done_reasons=[])
    available_samples = buffer.current_size * buffer.num_envs
    assert available_samples == 4

    with pytest.raises(
        ValueError, match="Batch size .* is larger than available samples"
    ):
        buffer.get(available_samples + 1)


def test_stats_tracker_integration(
    setup_buffer: tuple[ReplayBuffer, StatsTracker],
) -> None:
    """Test that the stats tracker is updated on 'done'."""
    buffer, stats_tracker = setup_buffer
    sample_data = generate_sample_data(
        buffer.num_envs, buffer.obs_shape, buffer.action_shape
    )
    # Ensure at least one env is done
    sample_data["done"][0] = True
    sample_data["done"][1:] = False
    sample_data["reward"][0] = 10.0
    sample_data["infos"]["custom_metric"][0] = 5.0

    buffer.add(**sample_data, done_reasons=[])

    # Check if the episode info buffer in the tracker was populated
    assert len(stats_tracker.ep_info_buffer) > 0
    stats = stats_tracker.get_statistics()
    assert "Rollout/rewards_mean" in stats
    assert "Rollout/lengths_mean" in stats
    assert "Rollout/custom_metric_mean" in stats
    np.testing.assert_allclose(stats["Rollout/rewards_mean"], 10.0)
    np.testing.assert_allclose(stats["Rollout/custom_metric_mean"], 5.0)


def test_multidimensional_observations() -> None:
    """Test support for multi-dimensional observation spaces like images."""
    buffer = ReplayBuffer(
        buffer_size=10,
        num_envs=2,
        obs_shape=(3, 64, 64),
        action_shape=(4,),
        device=torch.device("cpu"),
        stats_tracker=StatsTracker(num_envs=2, max_ep_info_buffer=100),
    )

    for _ in range(2):
        sample_data = generate_sample_data(
            buffer.num_envs, buffer.obs_shape, buffer.action_shape
        )
        buffer.add(**sample_data, done_reasons=[])

    assert buffer._observations.shape == (10, 2, 3, 64, 64)
    batch = buffer.get(4)
    assert batch.observations.shape == (4, 3, 64, 64)
    assert batch.next_observations.shape == (4, 3, 64, 64)


def test_edge_case_single_env() -> None:
    """Test the buffer with only a single environment."""
    buffer = ReplayBuffer(
        buffer_size=5,
        num_envs=1,
        obs_shape=(4,),
        action_shape=(2,),
        stats_tracker=StatsTracker(num_envs=1, max_ep_info_buffer=10),
    )
    sample_data = generate_sample_data(
        buffer.num_envs, buffer.obs_shape, buffer.action_shape
    )

    buffer.add(**sample_data, done_reasons=[])

    assert buffer.current_size == 1
    assert buffer.total_current_size == 1

    batch = buffer.get(1)
    assert batch.observations.shape == (1, 4)
    assert batch.actions.shape == (1, 2)


def test_replay_batch_creation() -> None:
    """Test the creation and properties of a ReplayBatch object."""
    batch_size = 4
    obs_shape = (8,)
    action_shape = (2,)

    batch = ReplayBatch(
        observations=torch.randn(batch_size, *obs_shape),
        actions=torch.randn(batch_size, *action_shape),
        rewards=torch.randn(batch_size),
        next_observations=torch.randn(batch_size, *obs_shape),
        dones=torch.randint(0, 2, (batch_size,)).float(),
        truncated=torch.randint(0, 2, (batch_size,)).float(),
    )

    assert batch.observations.shape == (batch_size, *obs_shape)
    assert batch.actions.shape == (batch_size, *action_shape)
    assert batch.rewards.shape == (batch_size,)
    assert batch.next_observations.shape == (batch_size, *obs_shape)
    assert batch.dones.shape == (batch_size,)
    assert batch.truncated.shape == (batch_size,)
