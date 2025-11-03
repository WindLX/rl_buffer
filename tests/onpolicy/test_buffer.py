import pytest

import torch
import numpy as np

from rl_buffer import RolloutBuffer, StatsTracker


@pytest.fixture
def setup_buffer() -> tuple[RolloutBuffer, StatsTracker]:
    """Fixture to create a common RolloutBuffer instance for tests."""
    buffer_size = 5
    num_envs = 2
    obs_shape = (4,)
    action_shape = (2,)
    stats_tracker = StatsTracker(
        num_envs=num_envs, extra_metrics_keys=["custom_metric"]
    )
    return (
        RolloutBuffer(
            buffer_size,
            num_envs,
            obs_shape,
            action_shape,
            stats_tracker=stats_tracker,
        ),
        stats_tracker,
    )


def test_initialization(setup_buffer: tuple[RolloutBuffer, StatsTracker]):
    """Test if the buffer initializes correctly."""
    buffer = setup_buffer[0]
    assert buffer._buffer_size == 5
    assert buffer.num_envs == 2
    assert buffer.gamma == 0.99
    assert buffer.gae_lambda == 0.95
    assert not buffer.full
    assert buffer.current_size == 0


def test_add_transition(setup_buffer: tuple[RolloutBuffer, StatsTracker]):
    """Test adding a single transition."""
    buffer = setup_buffer[0]
    obs = torch.rand(buffer.num_envs, *buffer._observations.shape[2:])
    action = torch.rand(buffer.num_envs, *buffer._actions.shape[2:])
    reward = torch.tensor([1.0, 2.0])
    done = torch.tensor([False, False])
    value = torch.tensor([0.5, 0.6])
    log_prob = torch.tensor([-0.1, -0.2])

    buffer.add(
        obs,
        action,
        reward,
        done,
        value,
        log_prob,
    )

    assert buffer.current_size == 1
    assert not buffer.full
    torch.testing.assert_close(buffer._observations[0], obs, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(buffer._rewards[0], reward, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(buffer._dones[0], done, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(buffer._values[0], value, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(buffer._log_probs[0], log_prob, atol=1e-5, rtol=1e-5)


def test_fill_buffer(setup_buffer: tuple[RolloutBuffer, StatsTracker]):
    """Test filling the buffer to its capacity."""
    buffer = setup_buffer[0]
    for i in range(buffer._buffer_size):
        obs = torch.rand(buffer.num_envs, *buffer._observations.shape[2:])
        action = torch.rand(buffer.num_envs, *buffer._actions.shape[2:])
        reward = torch.tensor([float(i), float(i + 1)])
        done = torch.tensor([False, False])
        value = torch.tensor([0.5, 0.6])
        log_prob = torch.tensor([-0.1, -0.2])
        buffer.add(
            obs,
            action,
            reward,
            done,
            value,
            log_prob,
        )

    assert buffer.current_size == buffer._buffer_size
    assert buffer.full

    # Try adding to a full buffer
    with pytest.raises(RuntimeError, match="Buffer is full."):
        buffer.add(obs, action, reward, done, value, log_prob)  # type: ignore


def test_reset_buffer(setup_buffer: tuple[RolloutBuffer, StatsTracker]):
    """Test resetting the buffer."""
    buffer = setup_buffer[0]
    # Fill partially
    for i in range(buffer._buffer_size // 2):
        obs = torch.rand(buffer.num_envs, *buffer._observations.shape[2:])
        action = torch.rand(buffer.num_envs, *buffer._actions.shape[2:])
        reward = torch.tensor([1.0, 2.0])
        done = torch.tensor([False, False])
        value = torch.tensor([0.5, 0.6])
        log_prob = torch.tensor([-0.1, -0.2])
        buffer.add(
            obs,
            action,
            reward,
            done,
            value,
            log_prob,
        )

    buffer.reset()
    assert buffer.current_size == 0
    assert not buffer.full


def test_compute_returns_and_advantages(
    setup_buffer: tuple[RolloutBuffer, StatsTracker],
):
    """Test GAE and return computation."""
    stats_tracker = setup_buffer[1]
    buffer_size = 3
    num_envs = 1
    obs_shape = (1,)
    action_shape = (1,)
    gamma = 0.99
    gae_lambda = 0.95
    buffer = RolloutBuffer(
        buffer_size,
        num_envs,
        obs_shape,
        action_shape,
        stats_tracker=stats_tracker,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    # Define a simple sequence of rewards, values, and dones
    rewards = torch.tensor([[1.0], [2.0], [3.0]])  # Shape (buffer_size, num_envs)
    values = torch.tensor([[0.1], [0.2], [0.3]])  # Shape (buffer_size, num_envs)
    dones = torch.tensor([[False], [False], [True]])  # Episode ends at step 2 (index 2)

    # Fill the buffer
    for i in range(buffer_size):
        buffer.add(
            torch.tensor([[0.0]]),
            torch.tensor([[0.0]]),
            rewards[i],
            dones[i],
            values[i],
            torch.tensor([-0.5]),
        )

    last_values = torch.tensor([0.4])  # Value of the state after the last step
    last_dones = torch.tensor(
        [True]
    )  # The episode ended, so effectively done from here

    buffer.compute_returns_and_advantages(last_values, last_dones)

    # Manual calculation for expected GAE and returns
    # R_t = r_t + gamma * V_{t+1} * (1 - d_{t+1})
    # GAE_t = delta_t + gamma * lambda * (1 - d_{t+1}) * GAE_{t+1}
    # delta_t = R_t + gamma * V_{t+1} * (1 - d_{t+1}) - V_t

    # Step 2 (index 2): reward=3.0, value=0.3, done=True
    # delta_2 = 3.0 + gamma * last_value * (1 - last_done) - value_2
    #         = 3.0 + 0.99 * 0.4 * (1 - 1) - 0.3 = 3.0 - 0.3 = 2.7
    # GAE_2 = delta_2 + gamma * lambda * (1 - last_done) * GAE_3 (GAE_3 is 0)
    #       = 2.7 + 0.99 * 0.95 * 0 * 0 = 2.7
    # Return_2 = GAE_2 + Value_2 = 2.7 + 0.3 = 3.0

    expected_delta_2 = (
        rewards[2][0]
        + buffer.gamma * last_values[0] * (1.0 - last_dones[0].float())
        - values[2][0]
    )
    expected_gae_2 = (
        expected_delta_2
        + buffer.gamma * buffer.gae_lambda * (1.0 - last_dones[0].float()) * 0
    )  # GAE from next step is 0
    expected_returns_2 = expected_gae_2 + values[2][0]

    # Step 1 (index 1): reward=2.0, value=0.2, done=False
    # delta_1 = 2.0 + gamma * value_2 * (1 - done_2) - value_1
    #         = 2.0 + 0.99 * 0.3 * (1 - 1) - 0.2 = 2.0 - 0.2 = 1.8
    # GAE_1 = delta_1 + gamma * lambda * (1 - done_2) * GAE_2
    #         = 1.8 + 0.99 * 0.95 * (1 - 1) * 2.7 = 1.8
    # Return_1 = GAE_1 + Value_1 = 1.8 + 0.2 = 2.0

    expected_delta_1 = (
        rewards[1][0]
        + buffer.gamma * values[2][0] * (1.0 - dones[2][0].float())
        - values[1][0]
    )
    expected_gae_1 = (
        expected_delta_1
        + buffer.gamma
        * buffer.gae_lambda
        * (1.0 - dones[2][0].float())
        * expected_gae_2
    )  # Using expected_gae_2 for calculation
    expected_returns_1 = expected_gae_1 + values[1][0]

    # Step 0 (index 0): reward=1.0, value=0.1, done=False
    # delta_0 = 1.0 + gamma * value_1 * (1 - done_1) - value_0
    #         = 1.0 + 0.99 * 0.2 * (1 - 0) - 0.1 = 1.0 + 0.198 - 0.1 = 1.098
    # GAE_0 = delta_0 + gamma * lambda * (1 - done_1) * GAE_1
    #         = 1.098 + 0.99 * 0.95 * (1 - 0) * 1.8 = 1.098 + 1.6983 = 2.7963
    # Return_0 = GAE_0 + Value_0 = 2.7963 + 0.1 = 2.8963

    expected_delta_0 = (
        rewards[0][0]
        + buffer.gamma * values[1][0] * (1.0 - dones[1][0].float())
        - values[0][0]
    )
    expected_gae_0 = (
        expected_delta_0
        + buffer.gamma
        * buffer.gae_lambda
        * (1.0 - dones[1][0].float())
        * expected_gae_1
    )  # Using expected_gae_1
    expected_returns_0 = expected_gae_0 + values[0][0]

    torch.testing.assert_close(
        buffer._advantages[2][0], expected_gae_2, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        buffer._returns[2][0], expected_returns_2, atol=1e-5, rtol=1e-5
    )

    torch.testing.assert_close(
        buffer._advantages[1][0], expected_gae_1, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        buffer._returns[1][0], expected_returns_1, atol=1e-5, rtol=1e-5
    )

    torch.testing.assert_close(
        buffer._advantages[0][0], expected_gae_0, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        buffer._returns[0][0], expected_returns_0, atol=1e-5, rtol=1e-5
    )


def test_get_mini_batches(setup_buffer: tuple[RolloutBuffer, StatsTracker]):
    """Test getting mini-batches and shuffling."""
    buffer = setup_buffer[0]
    total_steps = buffer._buffer_size * buffer.num_envs
    for i in range(buffer._buffer_size):
        obs = torch.rand(buffer.num_envs, *buffer._observations.shape[2:]) + i
        action = torch.rand(buffer.num_envs, *buffer._actions.shape[2:]) + i
        reward = torch.tensor([float(i), float(i + 1)])
        done = torch.tensor([False, False])
        value = torch.tensor([0.5, 0.6]) + i
        log_prob = torch.tensor([-0.1, -0.2]) - i
        buffer.add(
            obs,
            action,
            reward,
            done,
            value,
            log_prob,
        )

    last_values = torch.tensor([0.4, 0.5])
    last_dones = torch.tensor([False, False])
    buffer.compute_returns_and_advantages(
        last_values,
        last_dones,
    )

    batch_size = 3
    batches = list(
        buffer.get(batch_size=batch_size, shuffle=False)
    )  # No shuffle for predictable order

    # Check number of batches
    assert len(batches) == (total_steps + batch_size - 1) // batch_size

    # Check shapes of the first batch
    first_batch = batches[0]
    assert first_batch.observations.shape[0] == batch_size
    assert first_batch.actions.shape[0] == batch_size
    assert first_batch.log_probs.shape[0] == batch_size
    assert first_batch.returns.shape[0] == batch_size
    assert first_batch.advantages.shape[0] == batch_size
    assert first_batch.values.shape[0] == batch_size

    # Check if data is present and correctly reshaped (without explicit content check for shuffled)
    # For shuffle=False, we can check content of first batch
    expected_obs_flat = buffer._observations.reshape(
        total_steps, *buffer._observations.shape[2:]
    ).clone()
    torch.testing.assert_close(first_batch.observations, expected_obs_flat[:batch_size])

    # Test with shuffle=True to ensure it doesn't break
    shuffled_batches = list(buffer.get(batch_size=batch_size, shuffle=True))
    assert len(shuffled_batches) == len(batches)
    # The content order will be different, but shapes should be consistent


def test_episode_statistics_no_episodes_completed(
    setup_buffer: tuple[RolloutBuffer, StatsTracker],
):
    """Test episode statistics when no episodes are completed."""
    buffer = setup_buffer[0]
    # Add steps without any 'done'
    for _ in range(buffer._buffer_size // 2):
        obs = torch.rand(buffer.num_envs, *buffer._observations.shape[2:])
        action = torch.rand(buffer.num_envs, *buffer._actions.shape[2:])
        reward = torch.tensor([1.0, 2.0])
        done = torch.tensor([False, False])
        value = torch.tensor([0.5, 0.6])
        log_prob = torch.tensor([-0.1, -0.2])
        buffer.add(
            obs,
            action,
            reward,
            done,
            value,
            log_prob,
        )

    _, stats = buffer.stats_tracker.get_statistics()
    assert stats == {}


def test_episode_statistics_single_episode_per_env(
    setup_buffer: tuple[RolloutBuffer, StatsTracker],
):
    """Test episode statistics with a single completed episode for each environment."""
    buffer = setup_buffer[0]
    # Env 0: rewards [1, 1, 1], done at step 2 (length 3, total reward 3)
    # Env 1: rewards [2, 2, 2], done at step 2 (length 3, total reward 6)
    rewards_per_step = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    dones_per_step = torch.tensor(
        [[False, False], [False, False], [True, True]]
    )  # Both envs done at this step

    custom_metrics_per_step = torch.tensor(
        [[0.1, 0.2], [0.2, 0.4], [0.3, 0.6]]
    )  # Custom metrics for each

    for i in range(len(rewards_per_step)):
        obs = torch.rand(buffer.num_envs, *buffer._observations.shape[2:])
        action = torch.rand(buffer.num_envs, *buffer._actions.shape[2:])
        value = torch.tensor([0.5, 0.6])
        log_prob = torch.tensor([-0.1, -0.2])
        buffer.add(
            obs,
            action,
            rewards_per_step[i],
            dones_per_step[i],
            value,
            log_prob,
            {"custom_metric": custom_metrics_per_step[i]},  # extra info
        )

    _, stats = buffer.stats_tracker.get_statistics()

    torch.testing.assert_close(stats["Rollout/rewards_mean"], (3.0 + 6.0) / 2)
    torch.testing.assert_close(stats["Rollout/lengths_mean"], (3 + 3) / 2)
    torch.testing.assert_close(
        stats["Rollout/custom_metric_mean"], (0.1 + 0.2 + 0.3 + 0.2 + 0.4 + 0.6) / 2
    )


def test_episode_statistics_multiple_episodes_different_lengths(
    setup_buffer: tuple[RolloutBuffer, StatsTracker],
):
    """Test episode statistics with multiple episodes of varying lengths."""
    buffer = setup_buffer[0]

    # Env 0: Ep1 (len 2, rew 2)
    # Env 1: Ep1 (len 3, rew 3)
    # Env 0: Ep2 (len 2, rew 4)

    # Step 0
    buffer.add(
        torch.zeros((2, 4)),
        torch.zeros((2, 2)),
        torch.tensor([1.0, 1.0]),
        torch.tensor([False, False]),
        torch.zeros(2),
        torch.zeros(2),
        {"custom_metric": torch.tensor([0.1, 0.2])},  # extra info
    )
    # Step 1
    buffer.add(
        torch.zeros((2, 4)),
        torch.zeros((2, 2)),
        torch.tensor([1.0, 1.0]),
        torch.tensor([True, False]),
        torch.zeros(2),
        torch.zeros(2),
        {"custom_metric": torch.tensor([0.2, 0.4])},  # extra info
    )  # Env 0 done (rew=2, len=2)
    # Step 2
    buffer.add(
        torch.zeros((2, 4)),
        torch.zeros((2, 2)),
        torch.tensor([2.0, 1.0]),
        torch.tensor([False, True]),
        torch.zeros(2),
        torch.zeros(2),
        {"custom_metric": torch.tensor([0.3, 0.6])},  # extra info
    )  # Env 1 done (rew=3, len=3)
    # Step 3
    buffer.add(
        torch.zeros((2, 4)),
        torch.zeros((2, 2)),
        torch.tensor([2.0, 0.0]),
        torch.tensor([True, False]),
        torch.zeros(2),
        torch.zeros(2),
        {"custom_metric": torch.tensor([0.4, 0.0])},  # extra info
    )  # Env 0 done (rew=4, len=2)

    _, stats = buffer.stats_tracker.get_statistics()

    expected_rewards = [2.0, 3.0, 4.0]  # Env0-Ep1, Env1-Ep1, Env0-Ep2
    expected_lengths = [2, 3, 2]  # Env0-Ep1, Env1-Ep1, Env0-Ep2
    expected_custom_metric = [
        (0.1 + 0.2),
        (0.2 + 0.4 + 0.6),
        (0.3 + 0.4),
    ]  # Sum per episode

    np.testing.assert_allclose(stats["Rollout/rewards_mean"], np.mean(expected_rewards))
    np.testing.assert_allclose(stats["Rollout/lengths_mean"], np.mean(expected_lengths))
    np.testing.assert_allclose(
        stats["Rollout/custom_metric_mean"], np.mean(expected_custom_metric)
    )


def test_max_ep_info_buffer_limit():
    """Test if max_ep_info_buffer limits the stored episode info."""
    buffer_size = 5
    num_envs = 1
    obs_shape = (1,)
    action_shape = (1,)
    max_ep_info = 3
    stats_tracker = StatsTracker(num_envs=1, max_ep_info_buffer=max_ep_info)
    buffer = RolloutBuffer(
        buffer_size, num_envs, obs_shape, action_shape, stats_tracker=stats_tracker
    )

    # Complete 5 episodes
    for i in range(5):
        buffer.add(
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            torch.tensor([float(i)]),
            torch.tensor([True]),
            torch.zeros(1),
            torch.zeros(1),
        )

    assert len(buffer.stats_tracker.ep_info_buffer) == max_ep_info
    # The buffer should contain the last 'max_ep_info' episodes
    assert (
        buffer.stats_tracker.ep_info_buffer[0]["ep_rewards"] == 2.0
    )  # Rewards were 0, 1, 2, 3, 4. So 2, 3, 4 remain.
    assert buffer.stats_tracker.ep_info_buffer[1]["ep_rewards"] == 3.0
    assert buffer.stats_tracker.ep_info_buffer[2]["ep_rewards"] == 4.0


def test_gae_with_dones(setup_buffer: tuple[RolloutBuffer, StatsTracker]):
    """Test GAE calculation when done flags are present within the rollout."""
    stats_tracker = setup_buffer[1]
    buffer_size = 5
    num_envs = 1
    obs_shape = (1,)
    action_shape = (1,)
    gamma = 0.9
    gae_lambda = 0.8
    buffer = RolloutBuffer(
        buffer_size,
        num_envs,
        obs_shape,
        action_shape,
        stats_tracker=stats_tracker,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    # Scenario: Episode ends at step 2 (index 2)
    rewards = torch.tensor([[10.0], [0.0], [0.0], [100.0], [0.0]])
    values = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    dones = torch.tensor([[False], [False], [True], [False], [False]])

    for i in range(buffer_size):
        buffer.add(
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            rewards[i],
            dones[i],
            values[i],
            torch.zeros(1),
        )

    last_values = torch.tensor(
        [0.0]
    )  # Assuming new episode starts, value of next state is 0
    last_dones = torch.tensor([False])

    buffer.compute_returns_and_advantages(last_values, last_dones)

    # Expected GAE/Returns calculation (simplified for manual check, focused on restart)
    # The key is that GAE chain should break at `done=True`

    # For Env 0 (the only env):
    # Step 4: R=0, V=5. last_val=0, last_done=False
    # delta_4 = 0 + 0.9*0*(1-False) - 5 = -5
    # GAE_4 = -5 + 0.9*0.8*(1-False)*0 = -5
    # Ret_4 = -5 + 5 = 0

    # Step 3: R=100, V=4. next_val=5, next_done=False
    # delta_3 = 100 + 0.9*5*(1-False) - 4 = 100 + 4.5 - 4 = 100.5
    # GAE_3 = 100.5 + 0.9*0.8*(1-False)*(-5) = 100.5 - 3.6 = 96.9
    # Ret_3 = 96.9 + 4 = 100.9

    # Step 2: R=0, V=3. next_val=4, next_done=False (but current step is done=True)
    # delta_2 = 0 + 0.9*values[3]*(1-dones[3]) - values[2] = 0 + 0.9*4*(1-False) - 3 = 3.6 - 3 = 0.6
    # GAE_2 = 0.6 + 0.9*0.8*(1-dones[3])*GAE_3 = 0.6 + 0.9*0.8*(1-False)*96.9 = 0.6 + 69.768 = 70.368
    # Ret_2 = 70.368 + 3 = 73.368
    # The GAE should reset when the current step's done is True for that env's GAE calculation,
    # as the `next_non_terminal` for the *previous* step (step 1) looking at step 2 will be 0.

    # Let's re-calculate more carefully considering the `next_non_terminal` logic:
    # GAE calculation works backward. For step `s`, it looks at `s+1`.
    # If `dones[s+1]` is True, `next_non_terminal` for step `s` is 0.

    # Step 4 (idx 4):
    # next_non_terminal_4 = 1.0 - last_dones[0] = 1.0 - False = 1.0
    # next_value_4 = last_values[0] = 0.0
    # delta_4 = rewards[4][0] + gamma * next_value_4 * next_non_terminal_4 - values[4][0]
    #         = 0 + 0.9 * 0 * 1.0 - 5 = -5.0
    # GAE_4 = delta_4 + gamma * gae_lambda * next_non_terminal_4 * 0 = -5.0
    # Returns_4 = GAE_4 + values[4][0] = -5.0 + 5.0 = 0.0

    # Step 3 (idx 3):
    # next_non_terminal_3 = 1.0 - dones[4][0] = 1.0 - False = 1.0
    # next_value_3 = values[4][0] = 5.0
    # delta_3 = rewards[3][0] + gamma * next_value_3 * next_non_terminal_3 - values[3][0]
    #         = 100 + 0.9 * 5 * 1.0 - 4 = 100 + 4.5 - 4 = 100.5
    # GAE_3 = delta_3 + gamma * gae_lambda * next_non_terminal_3 * GAE_4
    #       = 100.5 + 0.9 * 0.8 * 1.0 * (-5.0) = 100.5 - 3.6 = 96.9
    # Returns_3 = GAE_3 + values[3][0] = 96.9 + 4 = 100.9

    # Step 2 (idx 2): `dones[2]` is True, but this doesn't directly affect GAE_2 calculation.
    # It affects GAE_1 calculation (next_non_terminal for step 1 will be 0 due to dones[2]).
    # next_non_terminal_2 = 1.0 - dones[3][0] = 1.0 - False = 1.0
    # next_value_2 = values[3][0] = 4.0
    # delta_2 = rewards[2][0] + gamma * next_value_2 * next_non_terminal_2 - values[2][0]
    #         = 0 + 0.9 * 4 * 1.0 - 3 = 3.6 - 3 = 0.6
    # GAE_2 = delta_2 + gamma * gae_lambda * next_non_terminal_2 * GAE_3
    #       = 0.6 + 0.9 * 0.8 * 1.0 * 96.9 = 0.6 + 69.768 = 70.368
    # Returns_2 = GAE_2 + values[2][0] = 70.368 + 3 = 73.368

    # Step 1 (idx 1):
    # next_non_terminal_1 = 1.0 - dones[2][0] = 1.0 - True = 0.0 (GAE chain broken)
    # next_value_1 = values[2][0] = 3.0
    # delta_1 = rewards[1][0] + gamma * next_value_1 * next_non_terminal_1 - values[1][0]
    #         = 0 + 0.9 * 3 * 0.0 - 2 = -2.0
    # GAE_1 = delta_1 + gamma * gae_lambda * next_non_terminal_1 * GAE_2
    #       = -2.0 + 0.9 * 0.8 * 0.0 * 70.368 = -2.0
    # Returns_1 = GAE_1 + values[1][0] = -2.0 + 2.0 = 0.0

    # Step 0 (idx 0):
    # next_non_terminal_0 = 1.0 - dones[1][0] = 1.0 - False = 1.0
    # next_value_0 = values[1][0] = 2.0
    # delta_0 = rewards[0][0] + gamma * next_value_0 * next_non_terminal_0 - values[0][0]
    #         = 10 + 0.9 * 2 * 1.0 - 1 = 10 + 1.8 - 1 = 10.8
    # GAE_0 = delta_0 + gamma * gae_lambda * next_non_terminal_0 * GAE_1
    #       = 10.8 + 0.9 * 0.8 * 1.0 * (-2.0) = 10.8 - 1.44 = 9.36
    # Returns_0 = GAE_0 + values[0][0] = 9.36 + 1 = 10.36

    expected_gae = np.array([9.36, -2.0, 70.368, 96.9, -5.0]).reshape(
        buffer_size, num_envs
    )
    expected_returns = np.array([10.36, 0.0, 73.368, 100.9, 0.0]).reshape(
        buffer_size, num_envs
    )

    np.testing.assert_allclose(buffer._advantages, expected_gae, rtol=1e-5)
    np.testing.assert_allclose(buffer._returns, expected_returns, rtol=1e-5)
