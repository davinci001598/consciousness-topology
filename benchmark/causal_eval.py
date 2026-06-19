#!/usr/bin/env python3
"""
推理深度评测器 v2
==============
修复：
  1. steps_completed: 追踪完成的因果对（钥+门）数量
  2. BFS模型：内部BFS支持拾取钥匙
  3. 场景生成：门周围加隔离墙，强制穿过门
"""

import os, sys, random
from typing import Tuple, List, Set, Optional
from dataclasses import dataclass
from collections import deque

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from causal_scene import (generate_causal_scene, DIRS,
                          KEY_START, DOOR_START, ONEWAY_START, TELEPORT)

DIR_NAMES = ["up", "right", "down", "left"]


# ═══════════════════════════════════════════════════
# 状态工具
# ═══════════════════════════════════════════════════

def make_causal_state(pos, scene, inventory, step):
    """step = 已完成的因果对数量 (0..chain_length)"""
    x, y = pos
    legal = []
    for d in DIRS:
        nx, ny = x + d[0], y + d[1]
        if not (0 <= nx < scene.size and 0 <= ny < scene.size):
            continue
        cell = scene.grid[ny][nx]
        if cell == 1:
            continue
        if DOOR_START <= cell < ONEWAY_START:
            door_idx = cell - DOOR_START
            if KEY_START + door_idx not in inventory:
                continue
        if ONEWAY_START <= cell < TELEPORT:
            if d != DIRS[cell - ONEWAY_START]:
                continue
        legal.append(d)

    return {
        "pos": pos,
        "scene": scene,
        "inventory": set(inventory),
        "step": step,
        "chain_length": len(scene.chain),
        "goal": scene.goal,
        "legal_actions": legal if legal else DIRS,
        "nearby_cells": {
            d: (0 if (KEY_START <= scene.grid[y + d[1]][x + d[0]] < DOOR_START
                      and scene.grid[y + d[1]][x + d[0]] in inventory)
                  else scene.grid[y + d[1]][x + d[0]])
            for d in DIRS
            if 0 <= x + d[0] < scene.size and 0 <= y + d[1] < scene.size
        },
    }


def bfs_optimal_steps(grid, size, start, goal):
    """BFS 计算无因果链约束下的最短路径步数（下界参考）"""
    visited = set()
    q = deque([(start[0], start[1], 0)])
    visited.add(start)
    while q:
        x, y, d = q.popleft()
        if (x, y) == goal:
            return d
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and grid[ny][nx] != 1:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny, d + 1))
    return size * size


# ═══════════════════════════════════════════════════
# 评测数据 & 指标
# ═══════════════════════════════════════════════════

@dataclass
class CausalResult:
    chain_length: int
    size: int
    steps_completed: int      # 完成的因果对数量 (0..chain_length)
    total_chain_steps: int
    model_steps: int
    optimal_steps: int
    door_attempts_no_key: int
    door_attempts_total: int
    reached_goal: bool
    terminated: bool = False


def compute_metrics(results):
    if not results:
        return {"error": "no results"}
    n = len(results)
    total_chain = sum(r.total_chain_steps for r in results)
    completed_chain = sum(r.steps_completed for r in results)
    total_no_key = sum(r.door_attempts_no_key for r in results)
    total_door = sum(r.door_attempts_total for r in results)
    reached = sum(1 for r in results if r.reached_goal)

    complete_rate = completed_chain / total_chain if total_chain else 0
    skip_rate = total_no_key / total_door if total_door else 0
    goal_rate = reached / n

    efficiencies = []
    for r in results:
        if r.reached_goal and r.model_steps > 0:
            eff = min(1.0, r.optimal_steps / max(1, r.model_steps))
        else:
            eff = 0.0  # 未到达目标 → 步数效率为 0
        efficiencies.append(eff)

    composite = (0.40 * complete_rate + 0.25 * (1 - skip_rate) +
                 0.20 * (sum(efficiencies)/n) + 0.15 * goal_rate)

    return {
        "complete_rate": round(complete_rate, 4),
        "skip_rate": round(skip_rate, 4),
        "step_efficiency": round(sum(efficiencies)/n, 4),
        "goal_rate": round(goal_rate, 4),
        "composite": round(composite, 4),
        "total_scenes": n,
        "rating": "A" if composite >= 0.7 else "B" if composite >= 0.5 else "C" if composite >= 0.3 else "D",
    }


# ═══════════════════════════════════════════════════
# 评测器
# ═══════════════════════════════════════════════════

class CausalEvaluator:
    def __init__(self, max_steps=500):
        self.max_steps = max_steps

    def evaluate(self, model, chain_lengths=None, sizes=None,
                 trials_per_config=3, wall_openings=0, dead_ends=0,
                 alcove_keys=0):
        if chain_lengths is None:
            chain_lengths = [2, 3, 4, 5]
        if sizes is None:
            sizes = [30]

        results = []
        for cl in chain_lengths:
            for size in sizes:
                for trial in range(trials_per_config):
                    seed = 42 + cl * 100 + size * 10 + trial
                    scene = generate_causal_scene(size, cl, wall_openings=wall_openings,
                                                  dead_ends=dead_ends,
                                                  alcove_keys=alcove_keys, seed=seed)
                    r = self._run_one(model, scene)
                    r.chain_length = cl
                    r.size = size
                    results.append(r)
                    print(f"  chain={cl} trial={trial} | "
                          f"steps={r.steps_completed}/{cl} "
                          f"model_steps={r.model_steps} "
                          f"skip={r.door_attempts_no_key} "
                          f"goal={r.reached_goal}")
        return results

    def _run_one(self, model, scene):
        pos = list(scene.start)
        inventory = set()
        step = 0  # 已完成的因果对数量
        model_steps = 0
        door_no_key = 0
        door_total = 0
        optimal = bfs_optimal_steps(scene.grid, scene.size, scene.start, scene.goal)
        chain_len = len(scene.chain)

        for _ in range(self.max_steps):
            model_steps += 1

            # 检查当前位置 → 拾取钥匙
            cell = scene.grid[pos[1]][pos[0]]
            if KEY_START <= cell < DOOR_START:
                inventory.add(cell)

            # 到达终点？
            if tuple(pos) == scene.goal:
                return CausalResult(
                    chain_length=chain_len, size=scene.size,
                    steps_completed=step, total_chain_steps=chain_len,
                    model_steps=model_steps, optimal_steps=optimal,
                    door_attempts_no_key=door_no_key,
                    door_attempts_total=door_total,
                    reached_goal=True)

            # 模型决策
            state = make_causal_state(tuple(pos), scene, inventory, step)
            action = model.predict(state)
            if action is None:
                break

            nx, ny = pos[0] + action[0], pos[1] + action[1]

            # 检查目标格是否是门 → 统计跳步
            if 0 <= nx < scene.size and 0 <= ny < scene.size:
                tgt = scene.grid[ny][nx]
                if DOOR_START <= tgt < ONEWAY_START:
                    door_total += 1
                    if KEY_START + (tgt - DOOR_START) not in inventory:
                        door_no_key += 1

            # 执行移动
            if action in state["legal_actions"]:
                old_pos = tuple(pos)
                pos[0], pos[1] = nx, ny
                cell_after = scene.grid[pos[1]][pos[0]]

                # 通过门 → 检查是否完成当前因果对
                if DOOR_START <= cell_after < ONEWAY_START:
                    door_idx = cell_after - DOOR_START
                    # 当前正在处理的因果对 = step（第 step 个）
                    # 对应 door_idx == step 且持有 key step
                    if door_idx == step and KEY_START + door_idx in inventory:
                        step += 1  # 完成当前因果对

        # 超时
        return CausalResult(
            chain_length=chain_len, size=scene.size,
            steps_completed=step, total_chain_steps=chain_len,
            model_steps=model_steps, optimal_steps=optimal,
            door_attempts_no_key=door_no_key,
            door_attempts_total=door_total,
            reached_goal=False, terminated=True)


# ═══════════════════════════════════════════════════
# 基准模型
# ═══════════════════════════════════════════════════

class RandomModel:
    def __init__(self, seed=42, silence=0.0):
        self.rng = random.Random(seed)
        self.silence = silence

    def predict(self, state):
        if self.rng.random() < self.silence:
            return None
        return self.rng.choice(state["legal_actions"])

    @property
    def model_id(self):
        return f"Random(sil={self.silence})"


class GreedyModel:
    """贪婪向终点，无视因果链"""
    def predict(self, state):
        x, y = state["pos"]
        gx, gy = state["goal"]
        best, best_h = None, float("inf")
        for d in state["legal_actions"]:
            h = abs(x+d[0]-gx) + abs(y+d[1]-gy)
            if h < best_h:
                best_h = h
                best = d
        return best

    @property
    def model_id(self):
        return "Greedy"


class BFSModel:
    """BFS 带钥匙拾取 → 理想上限"""
    def predict(self, state):
        pos = state["pos"]
        goal = state["goal"]
        scene = state["scene"]
        inventory = frozenset(state["inventory"])

        path = self._bfs_with_pickup(scene, pos, goal, inventory)
        if path and len(path) >= 2:
            nx, ny = path[1]
            return (nx - pos[0], ny - pos[1])
        if state["legal_actions"]:
            return state["legal_actions"][0]
        return None

    def _bfs_with_pickup(self, scene, start, goal, inventory):
        """
        BFS 内部允许拾取钥匙：
        状态 = (x, y, frozenset_of_keys)
        遇到钥匙格自动拾取，遇到门检查钥匙。
        """
        visited = set()
        q = deque()
        init = (start[0], start[1], inventory)
        q.append((start[0], start[1], set(inventory), [start]))
        visited.add(init)

        while q:
            x, y, inv, path = q.popleft()
            if (x, y) == goal:
                return path

            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < scene.size and 0 <= ny < scene.size):
                    continue
                cell = scene.grid[ny][nx]
                if cell == 1:
                    continue
                # 门检查
                if DOOR_START <= cell < ONEWAY_START:
                    if KEY_START + (cell - DOOR_START) not in inv:
                        continue
                # 单向通道
                if ONEWAY_START <= cell < TELEPORT:
                    if (dx, dy) != DIRS[cell - ONEWAY_START]:
                        continue

                new_inv = set(inv)
                # 自动拾取钥匙
                if KEY_START <= cell < DOOR_START:
                    new_inv.add(cell)

                state_key = (nx, ny, frozenset(new_inv))
                if state_key not in visited:
                    visited.add(state_key)
                    q.append((nx, ny, new_inv, path + [(nx, ny)]))
        return None

    @property
    def model_id(self):
        return "BFS_upperbound"


# ═══════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    evaluator = CausalEvaluator(max_steps=300)

    models = [
        RandomModel(silence=0.0),
        RandomModel(silence=0.3),
        GreedyModel(),
        BFSModel(),
    ]

    # 场景 A：纯直线走廊（baseline 对照）
    print("\n" + "="*60)
    print(" 场景 A: 纯直线走廊 (wall_openings=0, dead_ends=0, alcove_keys=0)")
    print("="*60)
    for model in models:
        print(f"\n 模型: {model.model_id}")
        results = evaluator.evaluate(model, chain_lengths=[2, 3, 4],
                                     trials_per_config=4,
                                     wall_openings=0, dead_ends=0,
                                     alcove_keys=0)
        m = compute_metrics(results)
        print(f" 完整推理率: {m['complete_rate']:.1%}")
        print(f" 跳步率:     {m['skip_rate']:.1%}")
        print(f" 步数效率:   {m['step_efficiency']:.1%}")
        print(f" 到达率:     {m['goal_rate']:.1%}")
        print(f" 综合分:     {m['composite']:.3f} ({m['rating']})")

    # 场景 B：凹室钥匙 + 死胡同 + 墙壁开口（复杂场景，区分推理能力）
    print("\n" + "="*60)
    print(" 场景 B: 复杂迷宫 (wall_openings=3, dead_ends=3, alcove_keys=2)")
    print("="*60)
    for model in models:
        print(f"\n 模型: {model.model_id}")
        results = evaluator.evaluate(model, chain_lengths=[2, 3, 4],
                                     trials_per_config=4,
                                     wall_openings=3, dead_ends=3,
                                     alcove_keys=2)
        m = compute_metrics(results)
        print(f" 完整推理率: {m['complete_rate']:.1%}")
        print(f" 跳步率:     {m['skip_rate']:.1%}")
        print(f" 步数效率:   {m['step_efficiency']:.1%}")
        print(f" 到达率:     {m['goal_rate']:.1%}")
        print(f" 综合分:     {m['composite']:.3f} ({m['rating']})")
