#!/usr/bin/env python3
"""
意识拓扑引擎 — 独立基准测试脚本
==================================
五策略网格导航对比: Random / MLP / Topological / MLP+Topo / Topo+MLP

使用方式:
    python benchmark.py                  # 默认: 全部条件, 80% 规模
    python benchmark.py --quick          # 快速模式: 仅 10x10, 30% 规模
    python benchmark.py --grid 50x50     # 仅跑 50x50

依赖: numpy (仅 Topo+MLP / MLP+Topo 需要)
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter

# ── 可选依赖 ──
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


# ============================================================================
# 配置
# ============================================================================

DEFAULT_GRIDS = ["10x10", "20x20", "50x50"]
DEFAULT_DENSITIES = [0.02, 0.05, 0.10, 0.20, 0.30]
DEFAULT_MAPS_PER_CONDITION = 10
DEFAULT_ROUNDS_PER_MAP = 30
DEFAULT_MAX_STEPS_FACTOR = 10  # max_steps = grid_size * factor
DEFAULT_SEED = 42
OUTPUT_FILE = "results.json"


# ============================================================================
# 网格与地图生成
# ============================================================================

def generate_map(size, density, rng):
    """生成 size x size 网格，随机放置密度为 density 的障碍。
    保证 (0,0) 和 (size-1,size-1) 无障碍。
    """
    grid = [[0] * size for _ in range(size)]
    num_obstacles = int(size * size * density)
    candidates = [(x, y) for x in range(size) for y in range(size)
                  if not (x == 0 and y == 0) and not (x == size - 1 and y == size - 1)]
    rng.shuffle(candidates)
    for x, y in candidates[:num_obstacles]:
        grid[y][x] = 1
    return grid


# ============================================================================
# 工具函数
# ============================================================================

def heuristic(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # 上右下左


def heuristic_action(pos, size, grid, goal):
    """启发式决策：从合法方向中选曼哈顿距离最小的。无合法方向时随机选。"""
    best_dir, best_h = None, float("inf")
    for d in DIRS:
        nx, ny = pos[0] + d[0], pos[1] + d[1]
        if 0 <= nx < size and 0 <= ny < size and not grid[ny][nx]:
            h = heuristic((nx, ny), goal)
            if h < best_h:
                best_h = h
                best_dir = d
    if best_dir is None:
        # 所有方向受阻，随机选一个不撞墙的
        valid = [d for d in DIRS
                 if 0 <= pos[0] + d[0] < size and 0 <= pos[1] + d[1] < size]
        best_dir = valid[0] if valid else DIRS[0]
    return best_dir


# ============================================================================
# 策略 1: Random — 启发式 + 30% 随机覆盖
# ============================================================================

class RandomStrategy:
    def __init__(self, rng, random_rate=0.30):
        self.rng = rng
        self.random_rate = random_rate

    def decide(self, pos, size, grid, goal):
        if self.rng.random() < self.random_rate:
            valid = [d for d in DIRS
                     if 0 <= pos[0] + d[0] < size and 0 <= pos[1] + d[1] < size]
            return valid[self.rng.randint(0, len(valid) - 1)] if valid else DIRS[0]
        return heuristic_action(pos, size, grid, goal)


# ============================================================================
# 策略 2: MLP — 启发式 + 20% 神经网络覆盖 (2 层)
# ============================================================================

class MLPStrategy:
    def __init__(self, rng, noise_rate=0.20):
        self.rng = rng
        self.noise_rate = noise_rate
        # 随机初始化 2 层 MLP: 输入4维传感器 → 隐藏8 → 输出4
        if HAS_NUMPY:
            self.w1 = np.random.randn(4, 8) * 0.5
            self.w2 = np.random.randn(8, 4) * 0.5
        else:
            self.w1 = [[random.gauss(0, 0.5) for _ in range(8)] for _ in range(4)]
            self.w2 = [[random.gauss(0, 0.5) for _ in range(4)] for _ in range(8)]

    def _mlp_forward(self, sensors):
        """4 维传感器 → 4 维方向 logits"""
        if HAS_NUMPY:
            x = np.maximum(0, np.dot(sensors, self.w1))
            return np.dot(x, self.w2)
        else:
            x = [max(0, sum(s * self.w1[i][j] for i, s in enumerate(sensors))) for j in range(8)]
            return [sum(xi * self.w2[j][i] for j, xi in enumerate(x)) for i in range(4)]

    def decide(self, pos, size, grid, goal):
        if self.rng.random() < self.noise_rate:
            # 传感器: 到各边界的归一化距离
            sensors = [pos[1] / size,
                       1 - pos[0] / size,
                       1 - pos[1] / size,
                       pos[0] / size]
            logits = self._mlp_forward(sensors)
            idx = max(range(4), key=lambda i: logits[i])
            d = DIRS[idx]
            nx, ny = pos[0] + d[0], pos[1] + d[1]
            if 0 <= nx < size and 0 <= ny < size and not grid[ny][nx]:
                return d
        return heuristic_action(pos, size, grid, goal)


# ============================================================================
# 策略 3: Topological — Chern=-1 QW + 熵门控
# ============================================================================

class TopologicalStrategy:
    def __init__(self, rng, entropy_threshold=1.2):
        self.rng = rng
        self.entropy_threshold = entropy_threshold

    def _qw_entropy(self, pos, size):
        """Chern=-1 量子行走: 计算四方向概率分布的熵值"""
        sensors = [pos[1] + 0.1,
                   size - pos[0],
                   size - pos[1],
                   pos[0] + 0.1]
        max_s = max(sensors) + 0.01
        probs = [math.exp(-s / max_s * 2) for s in sensors]
        total = sum(probs)
        probs = [p / total for p in probs]
        entropy = -sum(p * math.log(p) if p > 0 else 0 for p in probs)
        return entropy, probs

    def decide(self, pos, size, grid, goal):
        entropy, probs = self._qw_entropy(pos, size)
        if entropy < self.entropy_threshold:
            # 低熵 → 拓扑场确信，直接走最佳方向
            best_i = max(range(4), key=lambda i: probs[i])
            d = DIRS[best_i]
            nx, ny = pos[0] + d[0], pos[1] + d[1]
            if 0 <= nx < size and 0 <= ny < size and not grid[ny][nx]:
                return d
        return heuristic_action(pos, size, grid, goal)


# ============================================================================
# 策略 4: MLP+Topo — MLP 主引擎 + 拓扑覆盖层
# ============================================================================

class MLPTopoStrategy:
    def __init__(self, rng, entropy_threshold=1.0):
        self.rng = rng
        self.entropy_threshold = entropy_threshold
        self.mlp = MLPStrategy(rng, noise_rate=0.20)
        self.topo = TopologicalStrategy(rng, entropy_threshold)

    def decide(self, pos, size, grid, goal):
        # 先做拓扑场分析
        entropy, probs = self.topo._qw_entropy(pos, size)
        # 低熵 → 拓扑场覆盖 MLP
        if entropy < self.entropy_threshold:
            best_i = max(range(4), key=lambda i: probs[i])
            d = DIRS[best_i]
            nx, ny = pos[0] + d[0], pos[1] + d[1]
            if 0 <= nx < size and 0 <= ny < size and not grid[ny][nx]:
                return d
        # 否则退回 MLP
        return self.mlp.decide(pos, size, grid, goal)


# ============================================================================
# 策略 5: Topo+MLP — 拓扑主引擎 + MLP 覆盖层
# ============================================================================

class TopoMLPStrategy:
    def __init__(self, rng, entropy_threshold=1.4):
        self.rng = rng
        self.entropy_threshold = entropy_threshold
        self.mlp = MLPStrategy(rng, noise_rate=0.20)
        self.topo = TopologicalStrategy(rng, entropy_threshold)

    def decide(self, pos, size, grid, goal):
        entropy, probs = self.topo._qw_entropy(pos, size)
        # 低熵 → 拓扑场主引擎决策
        if entropy < self.entropy_threshold:
            best_i = max(range(4), key=lambda i: probs[i])
            d = DIRS[best_i]
            nx, ny = pos[0] + d[0], pos[1] + d[1]
            if 0 <= nx < size and 0 <= ny < size and not grid[ny][nx]:
                return d
        # 高熵 → 退回 MLP
        return self.mlp.decide(pos, size, grid, goal)


# ============================================================================
# 策略工厂
# ============================================================================

STRATEGIES = {
    "Random": RandomStrategy,
    "MLP": MLPStrategy,
    "Topological": TopologicalStrategy,
    "MLP+Topo": MLPTopoStrategy,
    "Topo+MLP": TopoMLPStrategy,
}


# ============================================================================
# 模拟引擎
# ============================================================================

def run_simulation(strategy, grid, size, start, goal, max_steps):
    """返回 (reached, steps, collisions, path_efficiency)"""
    pos = list(start)
    steps = 0
    collisions = 0
    heuristic_start = heuristic(start, goal)

    for _ in range(max_steps):
        if pos[0] == goal[0] and pos[1] == goal[1]:
            eff = heuristic_start / max(steps, 1)
            return True, steps, collisions, min(eff, 1.0)

        d = strategy.decide(pos, size, grid, goal)
        nx, ny = pos[0] + d[0], pos[1] + d[1]

        if not (0 <= nx < size and 0 <= ny < size) or grid[ny][nx]:
            collisions += 1
            steps += 1
            continue

        pos[0], pos[1] = nx, ny
        steps += 1

    # 超时，未到达
    eff = heuristic_start / max(steps, 1)
    return False, steps, collisions, min(eff, 1.0)


# ============================================================================
# 单条件测试
# ============================================================================

def run_condition(strategy_cls, size, density, n_maps, n_rounds, max_steps, seed):
    """返回该条件下某策略的汇总统计"""
    rng = random.Random(seed)
    all_reached = []
    all_steps = []
    all_collisions = []
    all_efficiency = []

    for map_i in range(n_maps):
        map_seed = seed + map_i * 1000
        map_rng = random.Random(map_seed)
        grid = generate_map(size, density, map_rng)
        start = (0, 0)
        goal = (size - 1, size - 1)

        for round_i in range(n_rounds):
            s = strategy_cls(rng)
            reached, steps, collisions, eff = run_simulation(
                s, grid, size, start, goal, max_steps
            )
            all_reached.append(reached)
            all_steps.append(steps)
            all_collisions.append(collisions)
            all_efficiency.append(eff)

    n = len(all_reached)
    return {
        "reach_rate": round(sum(all_reached) / n, 4),
        "avg_steps": round(sum(all_steps) / n, 1),
        "avg_collisions": round(sum(all_collisions) / n, 4),
        "avg_efficiency": round(sum(all_efficiency) / n, 4),
    }


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="意识拓扑引擎基准测试")
    parser.add_argument("--grid", type=str, default=None,
                        help="网格大小，如 50x50。不传则跑全部")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：仅 10x10, 30%% 规模")
    parser.add_argument("--density", type=float, default=None,
                        help="障碍密度, 如 0.30")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"随机种子 (默认 {DEFAULT_SEED})")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                        help=f"输出文件 (默认 {OUTPUT_FILE})")
    args = parser.parse_args()

    grids = [args.grid] if args.grid else (
        ["10x10"] if args.quick else DEFAULT_GRIDS
    )
    densities = [args.density] if args.density else DEFAULT_DENSITIES

    if args.quick:
        n_maps = max(1, int(DEFAULT_MAPS_PER_CONDITION * 0.3))
        n_rounds = max(1, int(DEFAULT_ROUNDS_PER_MAP * 0.3))
    else:
        n_maps = DEFAULT_MAPS_PER_CONDITION
        n_rounds = DEFAULT_ROUNDS_PER_MAP

    results = {}
    total_conditions = len(grids) * len(densities) * len(STRATEGIES)
    completed = 0
    t_start = time.time()

    for grid_str in grids:
        size = int(grid_str.split("x")[0])
        max_steps = size * DEFAULT_MAX_STEPS_FACTOR

        for density in densities:
            cond_key = f"{grid_str}_d{int(density * 100):02d}"
            results[cond_key] = {}

            for name, strategy_cls in STRATEGIES.items():
                summary = run_condition(
                    strategy_cls, size, density, n_maps, n_rounds,
                    max_steps, args.seed
                )
                results[cond_key][name] = {"summary": summary}
                completed += 1
                elapsed = time.time() - t_start
                print(f"[{completed}/{total_conditions}] {cond_key} {name} | "
                      f"reach={summary['reach_rate']:.3f} "
                      f"eff={summary['avg_efficiency']:.3f} "
                      f"({elapsed:.0f}s)")

    # 写入结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"完成 {completed} 个条件, 耗时 {total_time:.0f}s")
    print(f"结果已保存至 {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
