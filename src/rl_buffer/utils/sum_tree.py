import numpy as np


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        # 树的节点总数是 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        # 数据（优先级）存储在树的后半部分（叶子节点）
        self.data_pointer = 0

    def _propagate(self, idx: int, change: float):
        """从叶子节点向上更新父节点的值"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx: int, priority: float):
        """更新指定索引的叶子节点的优先级"""
        # data_pointer 指向的是数据存储的实际物理索引，而这里的 idx 是逻辑索引
        tree_idx = idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(self, priority: float):
        """添加一个新的优先级，如果满了则从头覆盖"""
        # data_pointer 是下一个要插入数据的位置
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(self.data_pointer, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def get(self, s: float) -> tuple[int, float]:
        """根据一个值 s 在树中查找对应的叶子节点索引和优先级"""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

        data_idx = idx - (self.capacity - 1)
        return data_idx, self.tree[idx]

    @property
    def total_priority(self) -> float:
        """返回所有优先级的总和（树的根节点）"""
        return self.tree[0]
