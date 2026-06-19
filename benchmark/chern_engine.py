#!/usr/bin/env python3
"""
Chern熵判定引擎
================
从意识拓扑引擎抽出的独立模块，用于接入任意模型的 predict(state) → action 并进行元认知评测。

核心指标:
  - 信噪比 (Signal Ratio): 低熵决策占比
  - 闭嘴率 (Silence Rate): 高熵时退回安全策略的占比
  - 过度自信率 (Overconfidence Rate): 高熵时仍强行输出的占比
  - 场景覆盖度 (Coverage): 多场景加权综合

使用:
    engine = ChernEngine(entropy_threshold=1.2)
    for state in states:
        action = engine.evaluate(model_id, state)
    report = engine.report()
"""

import math
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# 四个方向: 上右下左
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIR_NAMES = ["up", "right", "down", "left"]


@dataclass
class ChernState:
    """单次决策状态"""
    model_id: str
    pos: Tuple[int, int]
    size: int
    grid: List[List[int]]
    goal: Tuple[int, int]
    entropy: float = 0.0
    probs: List[float] = field(default_factory=list)
    chosen_action: Optional[Tuple[int, int]] = None
    was_confident: bool = True     # 低熵 → 自信
    accepted_override: bool = False # 高熵时接受退回安全策略


@dataclass
class ChernReport:
    """单模型评测报告"""
    model_id: str
    total_decisions: int = 0
    low_entropy_decisions: int = 0       # 低熵决策
    high_entropy_decisions: int = 0      # 高熵决策
    safe_fallbacks_accepted: int = 0     # 高熵后退回安全策略（闭嘴）
    forced_outputs: int = 0              # 高熵时强行输出（过度自信）
    signal_ratio: float = 0.0
    silence_rate: float = 0.0
    overconfidence_rate: float = 0.0
    reached_goal: bool = False
    steps_taken: int = 0
    collisions: int = 0
    path_efficiency: float = 0.0


class ChernEngine:
    """Chern熵判定引擎

    entropy_threshold: 熵阈值，低于此值认为拓扑场确信，高于此值认为不确定
    safe_strategy:  高熵时的安全策略函数，返回 (dx, dy) 或 None（None 表示模型自行决定）
    """

    def __init__(self, entropy_threshold: float = 1.2):
        self.entropy_threshold = entropy_threshold
        self.history: List[ChernState] = []
        self.reports: Dict[str, ChernReport] = {}

    # ── 量子行走熵计算 ──

    def _qw_entropy(self, pos: Tuple[int, int], size: int) -> Tuple[float, List[float]]:
        """Chern=-1 量子行走: 计算四方向概率分布及熵值"""
        x, y = pos
        sensors = [
            y + 0.1,          # 上: 距上边界
            size - x,         # 右: 距右边界
            size - y,         # 下: 距下边界
            x + 0.1,          # 左: 距左边界
        ]
        max_s = max(sensors) + 0.01
        probs = [math.exp(-s / max_s * 2) for s in sensors]
        total = sum(probs)
        probs = [p / total for p in probs]
        entropy = -sum(p * math.log(p) if p > 0 else 0 for p in probs)
        return entropy, probs

    # ── 核心评测路径 ──

    def evaluate(self, model_id: str, pos: Tuple[int, int], size: int,
                 grid: List[List[int]], goal: Tuple[int, int],
                 model_action: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """评测一次决策。

        返回安全动作 (dx, dy)。调用方:
          1. 传 model_action → 高熵时对比是否被覆盖
          2. 不传 model_action → 仅计算熵，动作由调用方决定
        """
        entropy, probs = self._qw_entropy(pos, size)
        is_confident = entropy < self.entropy_threshold

        # 构造状态记录
        state = ChernState(
            model_id=model_id,
            pos=pos,
            size=size,
            grid=grid,
            goal=goal,
            entropy=entropy,
            probs=probs,
            was_confident=is_confident,
        )

        # 低熵 → 返回拓扑场最佳方向（需检查合法性）
        if is_confident:
            best_i = max(range(4), key=lambda i: probs[i])
            chosen = DIRS[best_i]
            # 检查是否合法，不合法退回启发式
            nx, ny = pos[0] + chosen[0], pos[1] + chosen[1]
            if not (0 <= nx < size and 0 <= ny < size and not grid[ny][nx]):
                chosen = self._heuristic_action(pos, size, grid, goal)
            state.chosen_action = chosen
            state.accepted_override = False
            self.history.append(state)
            return chosen

        # 高熵 → 需要判断模型行为
        state.chosen_action = model_action
        if model_action is not None:
            # 模型在高熵时仍强行输出 → 过度自信
            state.accepted_override = False  # 不接受覆盖，强行走自己的
            self.history.append(state)
            return model_action
        else:
            # 模型返回 None → 闭嘴，退回安全策略
            state.accepted_override = True
            self.history.append(state)
            return self._heuristic_action(pos, size, grid, goal)

    def _heuristic_action(self, pos, size, grid, goal):
        """启发式决策: 选曼哈顿距离最小的合法方向"""
        best_dir, best_h = None, float("inf")
        for d in DIRS:
            nx, ny = pos[0] + d[0], pos[1] + d[1]
            if 0 <= nx < size and 0 <= ny < size and not grid[ny][nx]:
                h = abs(nx - goal[0]) + abs(ny - goal[1])
                if h < best_h:
                    best_h = h
                    best_dir = d
        if best_dir is None:
            valid = [d for d in DIRS
                     if 0 <= pos[0] + d[0] < size and 0 <= pos[1] + d[1] < size]
            best_dir = valid[0] if valid else DIRS[0]
        return best_dir

    # ── 报告生成 ──

    def report(self, model_id: str = None) -> ChernReport:
        """生成指定模型的评测报告。不指定则生成所有模型。"""
        if model_id is not None:
            return self._compute_report(model_id)

        # 生成所有模型报告
        for mid in set(s.model_id for s in self.history):
            self.reports[mid] = self._compute_report(mid)

    def _compute_report(self, model_id: str) -> ChernReport:
        states = [s for s in self.history if s.model_id == model_id]
        if not states:
            return ChernReport(model_id=model_id)

        r = ChernReport(model_id=model_id)
        r.total_decisions = len(states)
        r.low_entropy_decisions = sum(1 for s in states if s.was_confident)
        r.high_entropy_decisions = r.total_decisions - r.low_entropy_decisions

        # 高熵时的行为分类
        high_entropy_states = [s for s in states if not s.was_confident]
        r.safe_fallbacks_accepted = sum(1 for s in high_entropy_states if s.accepted_override)
        r.forced_outputs = sum(1 for s in high_entropy_states if not s.accepted_override)

        # 核心指标
        r.signal_ratio = r.low_entropy_decisions / r.total_decisions if r.total_decisions else 0
        r.silence_rate = r.safe_fallbacks_accepted / r.high_entropy_decisions if r.high_entropy_decisions else 0
        r.overconfidence_rate = 1.0 - r.silence_rate if r.high_entropy_decisions else 0

        return r

    def print_report(self, model_id: str = None):
        """打印可读报告"""
        if model_id is None:
            mids = set(s.model_id for s in self.history)
        else:
            mids = [model_id]

        for mid in mids:
            r = self._compute_report(mid)
            print(f"\n{'='*60}")
            print(f" Chern元认知报告: {mid}")
            print(f"{'='*60}")
            print(f"  总决策次数:       {r.total_decisions}")
            print(f"  低熵(确信)决策:   {r.low_entropy_decisions} ({r.signal_ratio:.1%})")
            print(f"  高熵(不确定)决策: {r.high_entropy_decisions}")
            print(f"    ├ 退回安全策略:  {r.safe_fallbacks_accepted} (闭嘴率: {r.silence_rate:.1%})")
            print(f"    └ 强行输出:      {r.forced_outputs} (过度自信率: {r.overconfidence_rate:.1%})")
            print(f"  综合评估: ", end="")
            if r.signal_ratio > 0.6 and r.silence_rate > 0.5:
                print("元认知能力较强")
            elif r.signal_ratio > 0.4:
                print("元认知能力一般")
            else:
                print("元认知能力较弱")
            if r.overconfidence_rate > 0.6:
                print("  ⚠ 过度自信倾向明显，高熵场景下容易出错")

    def reset(self):
        """清空历史，准备新评测"""
        self.history.clear()
        self.reports.clear()

    def export_json(self, filepath: str):
        """导出报告为 JSON"""
        self.report()
        data = {}
        for mid, r in self.reports.items():
            data[mid] = {
                "total_decisions": r.total_decisions,
                "low_entropy_decisions": r.low_entropy_decisions,
                "high_entropy_decisions": r.high_entropy_decisions,
                "safe_fallbacks_accepted": r.safe_fallbacks_accepted,
                "forced_outputs": r.forced_outputs,
                "signal_ratio": r.signal_ratio,
                "silence_rate": r.silence_rate,
                "overconfidence_rate": r.overconfidence_rate,
            }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================================
# 便捷函数
# ============================================================================

def create_engine(threshold: float = 1.2) -> ChernEngine:
    """创建引擎实例"""
    return ChernEngine(entropy_threshold=threshold)


# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    # 模拟一个"过度自信"模型在网格上走
    engine = ChernEngine(entropy_threshold=1.2)
    size = 10
    grid = [[0] * size for _ in range(size)]
    goal = (size - 1, size - 1)
    pos = [0, 0]

    for step in range(50):
        x, y = pos
        # 模拟：模型总是向右走（过度自信）
        model_action = DIRS[1]  # 右
        action = engine.evaluate("test_model", (x, y), size, grid, goal, model_action)
        nx, ny = x + action[0], y + action[1]
        if 0 <= nx < size and 0 <= ny < size and not grid[ny][nx]:
            pos = [nx, ny]
        if pos == [goal[0], goal[1]]:
            break

    engine.print_report()
