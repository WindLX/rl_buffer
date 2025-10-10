import pytest
import numpy as np

from rl_buffer.utils.sum_tree import SumTree


def test_sum_tree_init():
    """Test the initialization of the SumTree."""
    tree = SumTree(capacity=8)
    assert tree.capacity == 8
    # 树的总节点数应为 2*capacity - 1
    assert len(tree.tree) == 15
    assert np.all(tree.tree == 0)
    assert tree.data_pointer == 0


def test_sum_tree_add_and_total_priority():
    """Test adding priorities and checking the total sum."""
    tree = SumTree(capacity=4)

    # 第一次添加
    tree.add(priority=3.0)
    assert tree.data_pointer == 1
    assert np.isclose(tree.total_priority, 3.0)

    # 继续添加
    tree.add(priority=5.0)
    tree.add(priority=2.0)
    assert tree.data_pointer == 3
    assert np.isclose(tree.total_priority, 10.0)  # 3 + 5 + 2

    # 添加第四个，填满
    tree.add(priority=1.0)
    assert tree.data_pointer == 0  # data_pointer 循环回到起点
    assert np.isclose(tree.total_priority, 11.0)  # 3 + 5 + 2 + 1

    # 覆盖第一个
    tree.add(priority=4.0)
    assert tree.data_pointer == 1
    # 新的总和应为 4 (新) + 5 + 2 + 1 = 12
    assert np.isclose(tree.total_priority, 12.0)


def test_sum_tree_update():
    """Test updating an existing priority."""
    tree = SumTree(capacity=4)
    tree.add(3.0)
    tree.add(5.0)
    tree.add(2.0)  # total = 10.0

    # 更新索引为 1 的元素 (值为5.0) 为 1.0
    tree.update(idx=1, priority=1.0)
    # 新的总和应为 3 + 1 + 2 = 6
    assert np.isclose(tree.total_priority, 6.0)

    # 检查树内部结构 (可选，但有助于调试)
    # 叶子节点: tree[3] to tree[6] -> [3, 1, 2, 0]
    assert np.isclose(tree.tree[3 + 1], 1.0)

    # 更新索引为 0 的元素 (值为3.0) 为 7.0
    tree.update(idx=0, priority=7.0)
    # 新的总和应为 7 + 1 + 2 = 10
    assert np.isclose(tree.total_priority, 10.0)


def test_sum_tree_get():
    """Test retrieving an item based on a sample value."""
    tree = SumTree(capacity=4)
    priorities = [1.0, 2.0, 3.0, 4.0]
    for p in priorities:
        tree.add(p)

    # 总和是 10.0
    assert np.isclose(tree.total_priority, 10.0)

    # --- 测试采样边界 ---
    # s < 1.0 应该返回第一个元素
    idx, p = tree.get(s=0.5)
    assert idx == 0
    assert np.isclose(p, 1.0)

    # 1.0 <= s < 3.0 (1+2) 应该返回第二个元素
    idx, p = tree.get(s=1.5)
    assert idx == 1
    assert np.isclose(p, 2.0)

    # 3.0 <= s < 6.0 (1+2+3) 应该返回第三个元素
    idx, p = tree.get(s=4.0)
    assert idx == 2
    assert np.isclose(p, 3.0)

    # 6.0 <= s < 10.0 (1+2+3+4) 应该返回第四个元素
    idx, p = tree.get(s=8.0)
    assert idx == 3
    assert np.isclose(p, 4.0)


def test_sum_tree_large_capacity():
    """Test with a larger capacity to ensure correctness."""
    capacity = 1000
    tree = SumTree(capacity=capacity)

    priorities = np.random.rand(capacity) + 0.1
    for p in priorities:
        tree.add(p)

    assert np.isclose(tree.total_priority, np.sum(priorities))

    # 更新一个随机索引
    update_idx = np.random.randint(0, capacity)
    new_priority = 5.0
    expected_total = tree.total_priority - priorities[update_idx] + new_priority
    priorities[update_idx] = new_priority

    tree.update(update_idx, new_priority)

    assert np.isclose(tree.total_priority, expected_total)
    assert np.isclose(tree.total_priority, np.sum(priorities))
