#!/usr/bin/env python3
"""
标准化模型接口
==============
所有接入评测的模型必须实现 predict(state) → action 方法。

state 格式:
    {
        "pos": (x, y),           # 当前位置
        "size": int,             # 网格大小
        "grid": [[int, ...], ...],  # 障碍图, 0=空地, 1=障碍
        "goal": (gx, gy),        # 目标位置
        "legal_actions": [(dx, dy), ...]  # 合法方向列表
    }

action 格式:
    必须是 (dx, dy) 元组, 且必须在 legal_actions 中
    如果模型不确定, 可返回 None (表示"我不会", 触发闭嘴统计)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class ModelInterface(ABC):
    """所有被测模型必须实现此接口"""

    @abstractmethod
    def predict(self, state: dict) -> Optional[Tuple[int, int]]:
        """根据当前状态返回动作。

        Args:
            state: {
                "pos": (x, y),
                "size": int,
                "grid": [[int]],
                "goal": (gx, gy),
                "legal_actions": [(dx, dy), ...],
            }

        Returns:
            合法动作 (dx, dy), 或 None 表示不确定/不回答
        """
        pass

    @property
    def model_id(self) -> str:
        """模型标识，用于报告区分"""
        return self.__class__.__name__


# ============================================================================
# 内置对照模型
# ============================================================================

import random

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class RandomModel(ModelInterface):
    """随机基准: 总是随机选一个合法方向"""

    def __init__(self, seed: int = 42, silence_prob: float = 0.0):
        self.rng = random.Random(seed)
        self.silence_prob = silence_prob

    def predict(self, state: dict) -> Optional[Tuple[int, int]]:
        if self.rng.random() < self.silence_prob:
            return None  # 模拟"我不知道"
        return self.rng.choice(state["legal_actions"])

    @property
    def model_id(self):
        return f"Random(silence={self.silence_prob})"


class HeuristicModel(ModelInterface):
    """启发式基准: 总是选曼哈顿距离最小的方向"""

    def predict(self, state: dict) -> Optional[Tuple[int, int]]:
        x, y = state["pos"]
        gx, gy = state["goal"]
        best_dir, best_h = None, float("inf")
        for d in state["legal_actions"]:
            nx, ny = x + d[0], y + d[1]
            h = abs(nx - gx) + abs(ny - gy)
            if h < best_h:
                best_h = h
                best_dir = d
        return best_dir

    @property
    def model_id(self):
        return "Heuristic"


class MLPModel(ModelInterface):
    """简单2层MLP: 无训练，随机权重 → 作为低元认知对照"""

    def __init__(self, seed: int = 42):
        try:
            import numpy as np
            rng = np.random.RandomState(seed)
            self.w1 = rng.randn(4, 8) * 0.5
            self.w2 = rng.randn(8, 4) * 0.5
            self._np = np
        except ImportError:
            r = random.Random(seed)
            self.w1 = [[r.gauss(0, 0.5) for _ in range(8)] for _ in range(4)]
            self.w2 = [[r.gauss(0, 0.5) for _ in range(4)] for _ in range(8)]
            self._np = None
        self.rng = random.Random(seed)
        self.silence_prob = 0.0

    def predict(self, state: dict) -> Optional[Tuple[int, int]]:
        if self.rng.random() < self.silence_prob:
            return None
        x, y = state["pos"]
        size = state["size"]
        sensors = [y / size, 1 - x / size, 1 - y / size, x / size]
        if self._np:
            logits = self._np.dot(self._np.maximum(0, self._np.dot(sensors, self.w1)), self.w2)
            idx = int(self._np.argmax(logits))
        else:
            x1 = [max(0, sum(s * self.w1[i][j] for i, s in enumerate(sensors))) for j in range(8)]
            logits = [sum(xi * self.w2[j][i] for j, xi in enumerate(x1)) for i in range(4)]
            idx = max(range(4), key=lambda i: logits[i])
        return DIRS[idx] if DIRS[idx] in state["legal_actions"] else state["legal_actions"][0]

    @property
    def model_id(self):
        sp = f"_silence{self.silence_prob}" if self.silence_prob else ""
        return f"MLP{sp}"


# ============================================================================
# 场景状态构造辅助
# ============================================================================

def make_state(pos: Tuple[int, int], size: int, grid: List[List[int]],
               goal: Tuple[int, int]) -> dict:
    """从原始参数构造标准 state dict"""
    x, y = pos
    legal = [d for d in DIRS
             if 0 <= x + d[0] < size and 0 <= y + d[1] < size
             and not grid[y + d[1]][x + d[0]]]
    return {
        "pos": pos,
        "size": size,
        "grid": grid,
        "goal": goal,
        "legal_actions": legal if legal else DIRS,
    }


if __name__ == "__main__":
    # 自测
    state = make_state((0, 0), 10, [[0]*10 for _ in range(10)], (9, 9))
    for model in [RandomModel(), HeuristicModel(), MLPModel()]:
        print(f"{model.model_id}: predict → {model.predict(state)}")
