#!/usr/bin/env python3
"""
推理深度场景生成器 v2
=====================
线性走廊设计：确保钥匙和门按序强制经过。

布局（chain_length=3 示例）：
  ########################
  #S..k0......d0..k1...d1..k2...d2..G#
  ########################
  - k0-d0, k1-d1, k2-d2 各对之间有走廊
  - 只有一条通道，必须按序通过所有钥匙和门
  - 钥匙放在走廊中，门作为走廊的一部分

变量：
  - corridor_width: 相邻 key/door 对之间的走廊格数
  - wall_openings: 是否在走廊墙壁上随机开口（增加干扰路径）
"""

import random
from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

KEY_START = 10
DOOR_START = 20
MAX_KEYS = 10
ONEWAY_START = 30
TELEPORT = 40


@dataclass
class CausalScene:
    size: int
    grid: List[List[int]]
    start: Tuple[int, int]
    goal: Tuple[int, int]
    key_positions: Dict[int, Tuple[int, int]]
    door_positions: Dict[int, Tuple[int, int]]
    chain: List[Tuple[int, int]]  # [(key_id, door_id), ...]
    teleports: List[Tuple[int, int]]
    description: str = ""


def generate_causal_scene(size: int = 30, chain_length: int = 4,
                          corridor_width: int = 4, wall_openings: int = 1,
                          dead_ends: int = 0, alcove_keys: int = 0,
                          seed: int = 42) -> CausalScene:
    """
    线性走廊因果链场景。

    走廊布局（水平走向）:
    - y=1: 墙壁
    - y=2: 走廊（钥匙/门/空地）
    - y=3: 墙壁
    - 起始 x=2，终点 x=size-3

    每对 key/door 之间 spacing = corridor_width 格。
    钥匙放在走廊格上（特殊标记），门放在走廊格上。
    墙壁上有 wall_openings 个随机开口（增加干扰，让模型需要规划而非走直线）。
    """
    rng = random.Random(seed)
    chain_length = min(chain_length, MAX_KEYS)

    # 初始化全障碍
    grid = [[1] * size for _ in range(size)]

    # y=2 整行清空作为走廊
    corridor_y = 2
    for x in range(size):
        grid[corridor_y][x] = 0  # 所有列先清空

    # 墙上开口（干扰路径）：在 y=1 和 y=3 随机开口
    # 策略：将开口放在 door 前方 1 格处，制造"表面捷径"——
    # 贪心模型会试图穿过开口绕开门，但对面没有钥匙，仍无法通过
    openings = set()
    rng.seed(42)  # 重新固定种子确保位置可控
    candidate_openings = list(range(3, size-3))
    rng.shuffle(candidate_openings)
    if wall_openings > 0:
        for x in candidate_openings[:wall_openings]:
            openings.add(x)
            grid[1][x] = 0
            grid[3][x] = 0

    # 陷阱开口：在每扇门正前方 y=2 开一个向下/向上的开口
    # 贪心看到"门前有洞"会试图钻过去绕关门，但绕过去后发现对面还是障碍墙
    # BFS 不会被迷惑，因为 BFS 计算时会发现绕路后还要回来拿钥匙
    trap_openings = set()
    rng.seed(42 + chain_length * 100)

    # 死胡同分支（迷惑贪心模型）：从走廊向上/下扩展短分支，尽头是死路
    if dead_ends > 0:
        branch_candidates = list(range(3, size-3))
        rng.shuffle(branch_candidates)
        for x in branch_candidates[:dead_ends]:
            # 随机向上或向下挖 1~2 格
            dy = rng.choice([-1, 1])
            length = rng.randint(1, 2)
            for d in range(1, length + 1):
                ny = corridor_y + dy * d
                if 0 <= ny < size and grid[ny][x] == 1:
                    grid[ny][x] = 0

    # 起点
    start_x = 2
    start = (start_x, corridor_y)
    grid[corridor_y][start_x] = 2  # S

    # 计算最小所需宽度并调整
    total_segments = chain_length * 2
    min_width = start_x + total_segments * 2 + 4  # 每 segment 至少 2 格，+终点+边距
    if size < min_width:
        size = min_width + 2  # 留一点余量
    usable_width = (size - 4) - start_x  # 终点在 size-3
    segment_spacing = max(2, usable_width // (total_segments + 1))

    # 放置钥匙和门
    chain = []
    key_positions = {}
    door_positions = {}

    current_x = start_x + segment_spacing

    for i in range(chain_length):
        kid = KEY_START + i
        did = DOOR_START + i

        # 钥匙位置
        kx = current_x
        ky = corridor_y

        # 凹室钥匙：每间隔一个钥匙放在侧边凹室，贪心模型会径直走过
        # alcove_keys 控制从第几个钥匙开始变为凹室（如 alcove_keys=2 表示从第 2 把起）
        if alcove_keys > 0 and i >= alcove_keys - 1:
            # 奇数索引向上，偶数向下
            offset_y = -1 if i % 2 == 0 else 1
            alcove_ky = ky + offset_y
            if 0 <= alcove_ky < size:
                grid[alcove_ky][kx] = 0  # 挖开凹室入口
                openings.add(kx)  # 防止后续墙壁封闭覆盖此入口
                ky = alcove_ky
        current_x += segment_spacing

        # 门位置（在钥匙之后）
        dx_pos = current_x
        dy_pos = corridor_y
        current_x += segment_spacing

        # 确保不越界
        if dx_pos >= size - 2:
            kx = min(kx, size - 4 - segment_spacing)
            dx_pos = kx + segment_spacing

        grid[ky][kx] = kid
        grid[dy_pos][dx_pos] = did
        key_positions[kid] = (kx, ky)
        door_positions[did] = (dx_pos, dy_pos)
        chain.append((kid, did))

    # 终点
    goal_x = min(current_x + segment_spacing, size - 3)
    goal = (goal_x, corridor_y)
    grid[corridor_y][goal_x] = 3  # G

    # 墙壁封闭：y=1 和 y=3（除了开口）
    for x in range(size):
        if x not in openings:
            grid[1][x] = 1
            grid[3][x] = 1
    # 确保走廊上方是封闭的（y=1整行，除开口）
    grid[0][:] = [1] * size  # 顶部墙壁

    # 终点后封闭
    for x in range(goal_x + 1, size):
        grid[corridor_y][x] = 1

    # 起点前封闭
    for x in range(start_x):
        grid[corridor_y][x] = 1

    # 开口处保留空地（不覆盖已有钥匙/门标记）
    for ox in openings:
        if not (KEY_START <= grid[1][ox] < ONEWAY_START):
            grid[1][ox] = 0
        if not (KEY_START <= grid[3][ox] < ONEWAY_START):
            grid[3][ox] = 0
        # 开口向外延伸一格
        if ox > 0:
            grid[0][ox] = 0 if ox in openings else 1

    # 验证（最多重试 5 次，每次减少开口数）
    for attempt in range(5):
        if _verify_solvable(grid, size, start, goal, chain, key_positions):
            break
        # 更保守的参数重试
        grid = [[1] * size for _ in range(size)]
        for x in range(size):
            grid[corridor_y][x] = 0
        openings = set()
        for x in candidate_openings[:max(0, wall_openings - attempt - 1)]:
            openings.add(x)
            grid[1][x] = 0
            grid[3][x] = 0
        current_x = start_x + segment_spacing
        for i in range(chain_length):
            kid, did = KEY_START + i, DOOR_START + i
            kx = current_x; current_x += segment_spacing
            dx_pos = current_x; current_x += segment_spacing
            grid[corridor_y][kx] = kid
            grid[corridor_y][dx_pos] = did
            key_positions[kid] = (kx, corridor_y)
            door_positions[did] = (dx_pos, corridor_y)
        goal_x = min(current_x + segment_spacing, size - 3)
        grid[corridor_y][goal_x] = 3
        for x in range(size):
            if x not in openings:
                grid[1][x] = 1; grid[3][x] = 1
        grid[0][:] = [1] * size
        for x in range(goal_x + 1, size): grid[corridor_y][x] = 1
        for x in range(start_x): grid[corridor_y][x] = 1
        for ox in openings: grid[1][ox] = 0; grid[3][ox] = 0
    else:
        # 5 次都失败，用最保守参数强制生成
        pass  # 最终生成的场景作为兜底

    return CausalScene(
        size=size, grid=grid, start=start, goal=goal,
        key_positions=key_positions, door_positions=door_positions,
        chain=chain, teleports=[],
        description=f"线性走廊, {chain_length}步因果链, {wall_openings}个开口")


def _verify_solvable(grid, size, start, goal, chain, key_positions):
    """BFS 验证：遵循因果链顺序能否到达终点"""
    from collections import deque

    visited = set()
    q = deque()
    init_state = (start[0], start[1], 0, frozenset())
    visited.add(init_state)
    q.append((start[0], start[1], 0, set()))

    steps_needed = len(chain)

    while q:
        x, y, step, inventory = q.popleft()
        if (x, y) == goal and step == steps_needed:
            return True

        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < size and 0 <= ny < size):
                continue

            cell = grid[ny][nx]
            if cell == 1:
                continue

            new_step = step
            new_inv = set(inventory)

            if KEY_START <= cell < DOOR_START:
                new_inv.add(cell)
            elif DOOR_START <= cell < ONEWAY_START:
                door_idx = cell - DOOR_START
                if KEY_START + door_idx not in new_inv:
                    continue
                if door_idx == step:
                    new_step = step + 1
            elif ONEWAY_START <= cell < TELEPORT:
                if (dx, dy) != DIRS[cell - ONEWAY_START]:
                    continue

            new_state = (nx, ny, new_step, frozenset(new_inv))
            if new_state not in visited:
                visited.add(new_state)
                q.append((nx, ny, new_step, new_inv))

    return False


# ═══════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════

def print_grid(grid, size):
    CHARS = {0: '.', 1: '#', 2: 'S', 3: 'G'}
    for y in range(size):
        line = ""
        for x in range(size):
            c = grid[y][x]
            if c in CHARS:
                line += CHARS[c]
            elif KEY_START <= c < DOOR_START:
                line += chr(ord('a') + c - KEY_START)
            elif DOOR_START <= c < ONEWAY_START:
                line += chr(ord('A') + c - DOOR_START)
            elif ONEWAY_START <= c < TELEPORT:
                line += '><^v'[c - ONEWAY_START]
            elif c == TELEPORT:
                line += '@'
            else:
                line += '?'
        print(line)


# ═══════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    for cl in [2, 3, 4, 5]:
        for openings in [0, 2]:
            scene = generate_causal_scene(30, cl, 4, openings, seed=42 + cl * 10 + openings)
            ok = _verify_solvable(scene.grid, scene.size, scene.start,
                                   scene.goal, scene.chain, scene.key_positions)
            print(f"\n=== chain={cl} openings={openings} solvable={ok} ===")
            print_grid(scene.grid, scene.size)
            chain_labels = [(k - KEY_START, d - DOOR_START) for k, d in scene.chain]
            print(f"Chain: {chain_labels}")
