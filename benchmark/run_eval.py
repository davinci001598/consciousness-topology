#!/usr/bin/env python3
"""
评测执行器
==========
将模型接入Chern引擎，在网格导航场景上跑完整评测，输出报告。

使用:
    python run_eval.py                              # 跑内置模型
    python run_eval.py --model openai gpt-4o-mini   # 跑OpenAI模型
    python run_eval.py --model ollama llama3        # 跑Ollama本地模型
"""

import argparse
import json
import os
import sys
import time
from typing import List, Tuple

# 确保能导入同级和父级模块
bench_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, bench_dir)
sys.path.insert(0, os.path.dirname(bench_dir))  # 父目录, 用于 metrics.*

from chern_engine import ChernEngine, ChernReport
from api import ModelInterface, make_state, RandomModel, HeuristicModel, MLPModel
from metrics.scorer import score_model, generate_markdown_report, export_json

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


# ============================================================================
# 地图生成
# ============================================================================

def generate_map(size: int, density: float, seed: int = 42) -> list:
    """生成带障碍的网格地图"""
    import random
    rng = random.Random(seed)
    grid = [[0] * size for _ in range(size)]
    num_obs = int(size * size * density)
    candidates = [(x, y) for x in range(size) for y in range(size)
                  if not (x == 0 and y == 0) and not (x == size - 1 and y == size - 1)]
    rng.shuffle(candidates)
    for x, y in candidates[:num_obs]:
        grid[y][x] = 1
    return grid


# ============================================================================
# 模拟运行
# ============================================================================

def run_single_simulation(
    model: ModelInterface,
    engine: ChernEngine,
    size: int,
    density: float,
    max_steps: int,
    seed: int,
    verbose: bool = False,
) -> Tuple[bool, int, int, ChernReport]:
    """在单张地图上跑一次完整导航。

    Returns:
        (是否到达, 步数, 碰撞次数, Chern报告)
    """
    grid = generate_map(size, density, seed)
    pos = [0, 0]
    goal = (size - 1, size - 1)
    steps = 0
    collisions = 0
    model_id = model.model_id

    for _ in range(max_steps):
        if pos[0] == goal[0] and pos[1] == goal[1]:
            break

        # 构造状态
        state = make_state(tuple(pos), size, grid, goal)

        # 模型决策
        model_action = model.predict(state)

        # Chern引擎评测
        action = engine.evaluate(model_id, tuple(pos), size, grid, goal, model_action)

        # 如果模型返回 None，引擎已经回了启发式动作
        nx, ny = pos[0] + action[0], pos[1] + action[1]

        if not (0 <= nx < size and 0 <= ny < size) or grid[ny][nx]:
            collisions += 1
            steps += 1
            continue

        pos[0], pos[1] = nx, ny
        steps += 1

    reached = pos[0] == goal[0] and pos[1] == goal[1]
    report = engine.report(model_id)
    if report:
        report.reached_goal = reached
        report.steps_taken = steps
        report.collisions = collisions

    if verbose:
        status = "✓" if reached else "✗"
        print(f"  [{status}] {model_id} grid={size}x{size} d={density:.2f} "
              f"steps={steps} coll={collisions}")

    return reached, steps, collisions, report


# ============================================================================
# 批量评测
# ============================================================================

def run_benchmark(
    models: List[ModelInterface],
    grid_sizes: List[int] = [10, 20],
    densities: List[float] = [0.10, 0.20],
    maps_per_condition: int = 5,
    max_steps_factor: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """批量评测多个模型"""
    import random
    rng = random.Random(seed)

    results = {}
    total = len(models) * len(grid_sizes) * len(densities) * maps_per_condition
    completed = 0
    t0 = time.time()

    for model in models:
        engine = ChernEngine(entropy_threshold=1.2)
        engine.reset()
        model_reports = []

        for size in grid_sizes:
            for density in densities:
                max_steps = size * max_steps_factor

                for map_i in range(maps_per_condition):
                    map_seed = rng.randint(1, 100000)
                    reached, steps, coll, report = run_single_simulation(
                        model, engine, size, density, max_steps, map_seed, verbose=verbose
                    )
                    if report:
                        model_reports.append(report)
                    completed += 1

        # 评分
        scored = score_model(model_reports, coverage=0.25)  # 4场景，目前跑了1个
        results[model.model_id] = {
            "scored": scored,
            "reports": model_reports,
        }

        if verbose:
            print(f"\n  {model.model_id}: score={scored.metacognition_score} "
                  f"sig={scored.signal_ratio:.1%} sil={scored.silence_rate:.1%} "
                  f"over={scored.overconfidence_rate:.1%}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n{'='*60}")
        print(f"评测完成: {completed} 次模拟, 耗时 {elapsed:.0f}s")

    return results


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="AGI元认知评测执行器")
    parser.add_argument("--model", nargs=2, default=None, metavar=("TYPE", "NAME"),
                        help="接入外部模型, 如: --model openai gpt-4o-mini")
    parser.add_argument("--grid", type=int, nargs="+", default=[10, 20],
                        help="网格大小列表 (默认 10 20)")
    parser.add_argument("--density", type=float, nargs="+", default=[0.10, 0.20],
                        help="障碍密度列表 (默认 0.10 0.20)")
    parser.add_argument("--maps", type=int, default=5,
                        help="每条件地图数 (默认 5)")
    parser.add_argument("--output", type=str, default="output/report",
                        help="报告输出前缀 (默认 output/report)")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # 构建模型列表
    if args.model:
        model_type, model_name = args.model
        if model_type == "openai":
            from adapters.openai_adapter import OpenAIAdapter
            models = [OpenAIAdapter(model=model_name)]
        elif model_type == "ollama":
            from adapters.openai_adapter import OllamaAdapter
            models = [OllamaAdapter(model=model_name)]
        elif model_type == "deepseek":
            from adapters.openai_adapter import DeepSeekAdapter
            models = [DeepSeekAdapter(model=model_name)]
        else:
            print(f"未知模型类型: {model_type}")
            sys.exit(1)
    else:
        # 默认跑内置模型
        models = [
            RandomModel(silence_prob=0.0),
            RandomModel(silence_prob=0.5),
            HeuristicModel(),
            MLPModel(),
        ]

    verbose = not args.quiet

    # 跑评测
    results = run_benchmark(
        models=models,
        grid_sizes=args.grid,
        densities=args.density,
        maps_per_condition=args.maps,
        verbose=verbose,
    )

    # 生成报告
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    scored_list = [r["scored"] for r in results.values()]

    md_path = generate_markdown_report(scored_list, f"{args.output}.md")
    json_path = f"{args.output}.json"
    export_json(scored_list, json_path)

    print(f"\n报告已生成:")
    print(f"  Markdown: {os.path.abspath(md_path)}")
    print(f"  JSON:     {os.path.abspath(json_path)}")


if __name__ == "__main__":
    main()
