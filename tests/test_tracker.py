import pytest
from collections import deque
import numpy as np

from rl_buffer import StatsTracker


# --- 1. 使用 Fixture 减少重复代码 ---
@pytest.fixture
def tracker_two_envs():
    """Fixture to create a standard StatsTracker with 2 environments."""
    return StatsTracker(
        num_envs=2,
        category="test",
        max_ep_info_buffer=10,
        extra_metrics_keys=["score"],
        done_reason_keys=["timeout", "success"],
    )


# --- 2. 基础功能测试 (合并和增强) ---
def test_initial_state():
    """Test the initial state of the tracker."""
    tracker = StatsTracker(
        num_envs=3,
        extra_metrics_keys=["score", "energy"],
        done_reason_keys=["timeout"],
        category="MyCategory",
    )
    assert tracker.num_envs == 3
    assert tracker.category == "MyCategory"
    assert tracker.trajectory_num == 0
    assert "rewards" in tracker.metrics_keys
    assert "lengths" in tracker.metrics_keys
    assert "score" in tracker.metrics_keys
    assert "energy" in tracker.metrics_keys
    assert tracker.get_statistics() == {}
    assert tracker.get_raw_values() == {}


def test_basic_tracking_and_statistics(tracker_two_envs):
    """Test basic accumulation of rewards, lengths and calculation of full statistics."""
    # Step 1: No envs done
    tracker_two_envs.update(
        dones=np.array([0, 0]),
        rewards=np.array([1.0, 2.0]),
        infos={"score": np.array([10, 20])},
    )
    assert tracker_two_envs.trajectory_num == 0
    assert tracker_two_envs.get_statistics() == {}

    # Step 2: Env 0 finishes
    tracker_two_envs.update(
        dones=np.array([1, 0]),
        rewards=np.array([3.0, 4.0]),
        infos={"score": np.array([30, 40])},
    )
    assert tracker_two_envs.trajectory_num == 1

    # Check current stats after one episode
    # Episode 1: reward=1+3=4, length=2, score=10+30=40
    stats = tracker_two_envs.get_statistics()
    assert np.isclose(stats["test/rewards_mean"], 4.0)
    assert np.isclose(stats["test/rewards_std"], 0.0)  # std of one element is 0
    assert np.isclose(stats["test/rewards_min"], 4.0)
    assert np.isclose(stats["test/rewards_max"], 4.0)

    assert np.isclose(stats["test/lengths_mean"], 2.0)
    assert np.isclose(stats["test/score_mean"], 40.0)

    # Check that env 0's accumulators are reset, but env 1's are not
    assert tracker_two_envs._current_episode_stats["rewards"][0] == 0.0
    assert tracker_two_envs._current_episode_stats["rewards"][1] == 2.0 + 4.0

    # Step 3: Env 1 finishes
    tracker_two_envs.update(
        dones=np.array([0, 1]),
        rewards=np.array([5.0, 6.0]),  # Env 0 starts a new episode
        infos={"score": np.array([50, 60])},
    )
    assert tracker_two_envs.trajectory_num == 2

    # Now check stats with two completed episodes
    # Episode 1: reward=4, length=2, score=40
    # Episode 2: reward=2+4+6=12, length=3, score=20+40+60=120
    stats = tracker_two_envs.get_statistics()

    # --- 3. 精确断言 (Precise Assertions) ---
    rewards = np.array([4.0, 12.0])
    lengths = np.array([2.0, 3.0])
    scores = np.array([40.0, 120.0])
    step_rewards = rewards / lengths

    assert np.isclose(stats["test/rewards_mean"], np.mean(rewards))
    assert np.isclose(stats["test/rewards_std"], np.std(rewards))
    assert np.isclose(stats["test/rewards_min"], np.min(rewards))
    assert np.isclose(stats["test/rewards_max"], np.max(rewards))

    assert np.isclose(stats["test/lengths_mean"], np.mean(lengths))
    assert np.isclose(stats["test/lengths_std"], np.std(lengths))

    assert np.isclose(stats["test/score_mean"], np.mean(scores))
    assert np.isclose(stats["test/score_std"], np.std(scores))

    assert np.isclose(stats["test/step_reward_mean"], np.mean(step_rewards))
    assert np.isclose(stats["test/step_reward_std"], np.std(step_rewards))
    assert np.isclose(stats["test/step_reward_min"], np.min(step_rewards))
    assert np.isclose(stats["test/step_reward_max"], np.max(step_rewards))


# --- 4. 边缘情况测试 (Edge Cases) ---
def test_buffer_limit():
    """Test that the episode info buffer respects the maxlen limit."""
    tracker = StatsTracker(num_envs=1, max_ep_info_buffer=2, category="test")
    tracker.update(dones=np.array([1]), rewards=np.array([10.0]), infos={})  # Ep 1
    tracker.update(dones=np.array([1]), rewards=np.array([20.0]), infos={})  # Ep 2
    tracker.update(
        dones=np.array([1]), rewards=np.array([30.0]), infos={}
    )  # Ep 3, pushes out Ep 1

    assert tracker.trajectory_num == 2
    stats = tracker.get_statistics()
    # Should only average the last two episodes (20 and 30)
    assert np.isclose(stats["test/rewards_mean"], 25.0)


def test_no_completed_episodes():
    """Test behavior when no episodes have completed."""
    tracker = StatsTracker(num_envs=1)
    tracker.update(dones=np.array([0]), rewards=np.array([1.0]), infos={})
    tracker.update(dones=np.array([0]), rewards=np.array([1.0]), infos={})
    assert tracker.trajectory_num == 0
    assert tracker.get_statistics() == {}
    assert tracker.get_raw_values() == {}


def test_all_envs_done_simultaneously(tracker_two_envs):
    """Test when all environments terminate at the same time."""
    tracker_two_envs.update(
        dones=np.array([1, 1]),
        rewards=np.array([10.0, 20.0]),
        infos={"score": np.array([100, 200])},
    )
    assert tracker_two_envs.trajectory_num == 2
    stats = tracker_two_envs.get_statistics()
    assert np.isclose(stats["test/rewards_mean"], 15.0)
    assert np.isclose(stats["test/score_mean"], 150.0)


def test_zero_length_episode_does_not_crash():
    """Test a done signal on the very first step (length=1, not 0, but good to check)."""
    tracker = StatsTracker(num_envs=1)
    # Episode has length 1.
    tracker.update(dones=np.array([1]), rewards=np.array([5.0]))
    stats = tracker.get_statistics()
    # Step reward should be 5.0 / 1.0 = 5.0
    assert np.isclose(stats["Rollout/step_reward_mean"], 5.0)


def test_reset(tracker_two_envs):
    """Test that reset clears all internal states correctly."""
    tracker_two_envs.update(dones=np.array([1, 1]), rewards=np.array([1.0, 2.0]))
    assert tracker_two_envs.trajectory_num == 2
    assert len(tracker_two_envs.get_statistics()) > 0

    tracker_two_envs.reset()

    assert tracker_two_envs.trajectory_num == 0
    assert tracker_two_envs.get_statistics() == {}
    assert tracker_two_envs.ep_info_buffer == deque()
    # Check that accumulators are also zeroed
    assert np.all(tracker_two_envs._current_episode_stats["rewards"] == 0)
    assert np.all(tracker_two_envs._current_episode_stats["lengths"] == 0)


# --- 5. 特定功能测试 (增强) ---
def test_done_reasons_tracking():
    """Test tracking of episode termination reasons."""
    tracker = StatsTracker(
        num_envs=3, done_reason_keys=["timeout", "success", "failure"], category="test"
    )

    # Round 1: env 0 timeouts, env 1 succeeds, env 2 continues
    tracker.update(
        dones=np.array([1, 1, 0]),
        rewards=np.ones(3),
        done_reasons=["timeout", "success", None],
    )
    # Round 2: env 2 fails, env 0 starts new, env 1 starts new
    tracker.update(
        dones=np.array([0, 0, 1]),
        rewards=np.ones(3),
        done_reasons=[None, None, "failure"],
    )
    # Round 3: env 0 succeeds, env 1 has unrecognized reason
    tracker.update(
        dones=np.array([1, 1, 0]),
        rewards=np.ones(3),
        done_reasons=["success", "alien_invasion", None],
    )

    # Total completed episodes: 5
    # timeout: 1, success: 2, failure: 1, other: 1
    assert tracker.trajectory_num == 5
    stats = tracker.get_statistics()

    # Rates are calculated over the total number of episodes in the buffer
    assert np.isclose(stats["test/done_timeout_rate"], 1 / 5)
    assert np.isclose(stats["test/done_success_rate"], 2 / 5)
    assert np.isclose(stats["test/done_failure_rate"], 1 / 5)
    # Check that a key not in done_reason_keys is not present
    assert "test/done_alien_invasion_rate" not in stats


def test_multidimensional_rewards():
    """Test that multi-dimensional rewards are correctly summed."""
    tracker = StatsTracker(num_envs=2)
    rewards_step1 = np.array([[1.0, 0.5], [2.0, 0.0]])  # Sums to [1.5, 2.0]
    rewards_step2 = np.array([[3.0, 0.5], [1.0, 1.0]])  # Sums to [3.5, 2.0]

    tracker.update(dones=np.array([0, 0]), rewards=rewards_step1)
    tracker.update(dones=np.array([1, 1]), rewards=rewards_step2)

    stats = tracker.get_statistics()
    # Env 0 total reward: 1.5 + 3.5 = 5.0
    # Env 1 total reward: 2.0 + 2.0 = 4.0
    # Mean reward: (5.0 + 4.0) / 2 = 4.5
    assert np.isclose(stats["Rollout/rewards_mean"], 4.5)


def test_get_raw_values(tracker_two_envs):
    """Test the get_raw_values method."""
    tracker_two_envs.update(
        dones=np.array([1, 0]),
        rewards=np.array([10, 2]),
        infos={"score": np.array([100, 20])},
    )
    tracker_two_envs.update(
        dones=np.array([0, 1]),
        rewards=np.array([0, 8]),
        infos={"score": np.array([0, 80])},
    )

    # Ep 1 (env 0): reward=10, length=1, score=100
    # Ep 2 (env 1): reward=2+8=10, length=2, score=20+80=100

    raw = tracker_two_envs.get_raw_values()

    assert "test/rewards" in raw
    assert "test/lengths" in raw
    assert "test/score" in raw
    assert "test/step_reward" in raw

    np.testing.assert_array_equal(raw["test/rewards"], np.array([10.0, 10.0]))
    np.testing.assert_array_equal(raw["test/lengths"], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(raw["test/score"], np.array([100.0, 100.0]))
    np.testing.assert_array_equal(raw["test/step_reward"], np.array([10.0, 5.0]))
