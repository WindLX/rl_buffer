import pytest
import torch
import numpy as np

# 假设 PrioritizedReplayBuffer 和相关类在这个路径
from rl_buffer.prioritized_replay_buffer import (
    PrioritizedReplayBuffer,
    PrioritizedReplayBatch,
)
from rl_buffer.stats_tracker import StatsTracker


# 创建一个 MockStatsTracker，避免引入真实 StatsTracker 的复杂性
class MockStatsTracker:
    def update(self, *args, **kwargs):
        pass


@pytest.fixture
def per_buffer():
    """Fixture to create a PrioritizedReplayBuffer for testing."""
    return PrioritizedReplayBuffer(
        buffer_size=4,
        num_envs=2,
        obs_shape=(1,),
        action_shape=(1,),
        stats_tracker=MockStatsTracker(),  # type: ignore
        alpha=1.0,  # alpha=1.0 使得 p = |TD-error|，便于测试
        beta=1.0,  # beta=1.0 使得权重计算更直接
        epsilon=0.0,
    )


def test_per_buffer_init(per_buffer):
    """Test the initialization of the buffer."""
    assert per_buffer.total_capacity == 8
    assert per_buffer.sum_tree.capacity == 8
    assert per_buffer._max_priority == 1.0


def test_per_buffer_add(per_buffer):
    """Test adding transitions to the buffer."""
    # 添加一个时间步的数据 (num_envs=2)
    per_buffer.add(
        obs=torch.zeros(2, 1),
        action=torch.zeros(2, 1),
        reward=torch.zeros(2),
        done=torch.zeros(2, dtype=torch.bool),
        truncated=torch.zeros(2, dtype=torch.bool),
        next_obs=torch.zeros(2, 1),
    )

    assert per_buffer.current_size == 1
    assert per_buffer.total_current_size == 2

    # 新添加的两个样本，优先级都应为 max_priority (1.0)
    # SumTree 的 data_pointer 现在是 2
    assert per_buffer.sum_tree.data_pointer == 2
    assert np.isclose(per_buffer.sum_tree.total_priority, 2.0)

    # 再添加一个时间步
    per_buffer.add(
        obs=torch.ones(2, 1),
        action=torch.ones(2, 1),
        reward=torch.ones(2),
        done=torch.zeros(2, dtype=torch.bool),
        truncated=torch.zeros(2, dtype=torch.bool),
        next_obs=torch.ones(2, 1),
    )

    assert per_buffer.current_size == 2
    assert per_buffer.total_current_size == 4
    assert np.isclose(per_buffer.sum_tree.total_priority, 4.0)

    # 检查数据是否被正确存储
    assert torch.all(per_buffer._observations[1, 0] == 1.0)
    assert torch.all(per_buffer._actions[1, 1] == 1.0)


def test_per_buffer_get(per_buffer):
    """Test sampling from the buffer and IS weights calculation."""
    # 填充 buffer
    for i in range(4):
        per_buffer.add(
            obs=torch.full((2, 1), float(i)),
            action=torch.zeros(2, 1),
            reward=torch.zeros(2),
            done=torch.zeros(2, dtype=torch.bool),
            truncated=torch.zeros(2, dtype=torch.bool),
            next_obs=torch.zeros(2, 1),
        )

    # 此时所有8个样本的优先级都是1.0，总和为8.0
    assert per_buffer.total_current_size == 8
    assert np.isclose(per_buffer.sum_tree.total_priority, 8.0)

    # 此时采样应该是均匀的
    batch = per_buffer.get(batch_size=4)
    assert isinstance(batch, PrioritizedReplayBatch)
    assert batch.observations.shape == (4, 1)

    # 检查 IS 权重
    # 当 p 均匀时, P(i) = 1/N, w = (N * P(i))^-beta = (N * 1/N)^-1 = 1^-1 = 1
    expected_weights = torch.ones(4)
    assert torch.allclose(batch.weights, expected_weights)


def test_per_buffer_update_priorities_and_get(per_buffer):
    """Test the full cycle: add -> update priorities -> get with new priorities."""
    # 1. 填充 buffer
    for i in range(4):
        per_buffer.add(
            obs=torch.full((2, 1), float(i)),
            action=torch.zeros(2, 1),
            reward=torch.zeros(2),
            done=torch.zeros(2, dtype=torch.bool),
            truncated=torch.zeros(2, dtype=torch.bool),
            next_obs=torch.zeros(2, 1),
        )

    # 2. 模拟第一次采样和训练，然后更新优先级
    # 假设我们采样了索引为 [0, 2, 5, 7] 的样本
    indices_to_update = torch.tensor([0, 2, 5, 7])
    # 假设计算出的 TD-errors 如下
    td_errors = torch.tensor([0.1, 0.2, 1.0, 2.0])  # 样本5和7的error很大

    per_buffer.update_priorities(indices_to_update, td_errors)

    # 3. 检查 SumTree 是否被正确更新
    # 原始优先级都是1.0。新的总和应该是:
    # (8 - 4) + (0.1 + 0.2 + 1.0 + 2.0) = 4 + 3.3 = 7.3
    assert np.isclose(per_buffer.sum_tree.total_priority, 7.3)
    assert np.isclose(per_buffer._max_priority, 2.0)

    # 4. 再次采样，这次应该是有偏的
    # 由于样本 5 和 7 的优先级很高，多次采样它们被选中的概率应该远大于其他样本
    sample_counts = {i: 0 for i in range(8)}
    num_samples = 1000
    for _ in range(num_samples):
        batch = per_buffer.get(batch_size=1)
        sampled_idx = batch.indices.item()
        sample_counts[sampled_idx] += 1

    # 断言高优先级样本被采样的次数远多于低优先级样本
    assert sample_counts[5] > sample_counts[0] * 5
    assert sample_counts[7] > sample_counts[0] * 10

    # 5. 检查新采样批次的 IS 权重
    batch = per_buffer.get(batch_size=4)
    assert isinstance(batch, PrioritizedReplayBatch)

    # 权重不应该再是1.0了
    assert not torch.allclose(batch.weights, torch.ones(4))

    # 手动验证一个权重
    # P(i) = p_i / total_p
    # w_i = (N * P(i))^-beta = (8 * p_i / 7.3)^-1.0
    # 权重应该被归一化
    # 这是一个复杂的计算，测试它不全为1已经足够了


def test_weight_ratios_are_correct(per_buffer):
    """
    This is the definitive test for weights. It verifies the core mathematical
    relationship between priorities and weights, which is invariant to normalization
    and sampling randomness.
    The relationship is: w_i / w_j = (p_j / p_i)^beta
    """
    # 1. 设置一个清晰、无歧义的场景
    buffer = PrioritizedReplayBuffer(
        buffer_size=4,
        num_envs=1,  # 使用 num_envs=1 让索引更简单
        obs_shape=(1,),
        action_shape=(1,),
        stats_tracker=MockStatsTracker(),  # type: ignore
        alpha=0.7,
        beta=0.5,  # 使用非1.0的alpha和beta，测试更通用
        epsilon=1e-6,
    )

    # 2. 添加4个样本
    for i in range(4):
        buffer.add(
            obs=torch.zeros(1, 1),
            action=torch.zeros(1, 1),
            reward=torch.zeros(1),
            done=torch.zeros(1, dtype=torch.bool),
            truncated=torch.zeros(1, dtype=torch.bool),
            next_obs=torch.zeros(1, 1),
        )

    # 3. 设置已知的、不同的优先级
    all_indices = torch.tensor([0, 1, 2, 3])
    # 使用浮点数TD-error模拟真实场景
    td_errors = torch.tensor([0.1, 0.5, 1.0, 2.0])
    buffer.update_priorities(all_indices, td_errors)

    # 4. 采样整个缓冲区以获得所有样本和权重
    # 即使有重复，我们的逻辑也能处理
    batch = buffer.get(batch_size=4)

    # 5. 验证核心数学关系
    # 创建从索引到优先级和权重的映射
    priorities = (torch.abs(td_errors) + buffer.epsilon) ** buffer.alpha
    idx_to_priority = {i: p.item() for i, p in enumerate(priorities)}
    idx_to_weight = {
        idx.item(): w.item() for idx, w in zip(batch.indices, batch.weights)
    }

    # 获取批次中唯一的索引，以处理可能的重复采样
    unique_indices_in_batch = sorted(list(idx_to_weight.keys()))

    # 遍历所有唯一的索引对
    for i in range(len(unique_indices_in_batch)):
        for j in range(i + 1, len(unique_indices_in_batch)):
            idx1 = unique_indices_in_batch[i]
            idx2 = unique_indices_in_batch[j]

            p1 = idx_to_priority[int(idx1)]
            p2 = idx_to_priority[int(idx2)]

            w1 = idx_to_weight[idx1]
            w2 = idx_to_weight[idx2]

            # 预期比率: (p2 / p1)^beta
            expected_ratio = (p2 / p1) ** buffer.beta
            # 实际比率: w1 / w2 (因为权重与优先级成反比)
            actual_ratio = w1 / w2

            assert np.isclose(actual_ratio, expected_ratio), (
                f"Ratio mismatch for indices {idx1} and {idx2}. "
                f"Actual: {actual_ratio:.4f}, Expected: {expected_ratio:.4f}"
            )

    # 6. 额外检查 beta=0 的情况
    batch_beta0 = buffer.get(batch_size=4, beta=0.0)
    assert torch.allclose(
        batch_beta0.weights, torch.ones(4)
    ), "Weights should all be 1.0 when beta=0.0"
