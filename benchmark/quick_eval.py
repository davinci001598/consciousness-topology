#!/usr/bin/env python3
"""
快速采样评测
=============
不跑完整导航，而是在网格上采样决策点，只在Chern引擎判定为高熵的位置调API问模型。

用法:
    python quick_eval.py --model deepseek deepseek-chat --samples 50
"""

import argparse
import json
import os
import sys
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from chern_engine import ChernEngine
from api import ModelInterface, make_state, RandomModel, HeuristicModel, MLPModel

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def generate_map(size: int, density: float, seed: int = 42) -> list:
    rng = random.Random(seed)
    grid = [[0] * size for _ in range(size)]
    num_obs = int(size * size * density)
    candidates = [(x, y) for x in range(size) for y in range(size)
                  if not (x == 0 and y == 0) and not (x == size - 1 and y == size - 1)]
    rng.shuffle(candidates)
    for x, y in candidates[:num_obs]:
        grid[y][x] = 1
    return grid


def sample_decision_points(size: int, density: float, n_samples: int,
                           seed: int = 42, high_entropy_ratio: float = 0.4):
    """在地图上采样决策点，按熵值分类。使用动态分位阈值。

    策略：
    1. 密集采集所有可达点的熵值
    2. 用 P75 分位作为高熵阈值（动态、场景自适应）
    3. 按高低熵比例各取一半采样点
    """
    rng = random.Random(seed)
    grid = generate_map(size, density, seed)
    goal = (size - 1, size - 1)

    # 密集采集：找出所有空地，随机取 500 个计算熵
    all_empty = [(x, y) for x in range(size) for y in range(size) if not grid[y][x]]
    sample_pool = rng.sample(all_empty, min(500, len(all_empty)))

    candidates = []
    for x, y in sample_pool:
        engine_temp = ChernEngine(entropy_threshold=0.0)
        entropy, probs = engine_temp._qw_entropy((x, y), size)
        candidates.append({"pos": (x, y), "entropy": entropy})

    # 动态阈值：P75 分位
    entropies = sorted(c["entropy"] for c in candidates)
    threshold = entropies[int(len(entropies) * 0.75)]
    # 兜底：不低于 1.18，不高于 1.28
    threshold = max(1.18, min(threshold, 1.28))

    for c in candidates:
        c["is_high"] = c["entropy"] >= threshold

    high_points = [c for c in candidates if c["is_high"]]
    low_points = [c for c in candidates if not c["is_high"]]

    # 高/低熵各取一半
    n_each = n_samples // 2
    sampled = []
    if high_points:
        sampled.extend(rng.sample(high_points, min(n_each, len(high_points))))
    if low_points:
        sampled.extend(rng.sample(low_points, min(n_samples - len(sampled), len(low_points))))

    rng.shuffle(sampled)
    return grid, sampled, threshold


def evaluate_model_on_samples(model: ModelInterface, grid: list, size: int,
                              samples: list, engine: ChernEngine):
    """在采样点上评测模型"""
    model_id = model.model_id
    goal = (size - 1, size - 1)
    engine.reset()

    for s in samples:
        pos = s["pos"]
        state = make_state(pos, size, grid, goal)

        # 模型决策
        try:
            model_action = model.predict(state)
        except Exception as e:
            print(f"  [warn] {model_id} predict failed at {pos}: {e}")
            model_action = None

        # Chern引擎评测
        engine.evaluate(model_id, pos, size, grid, goal, model_action)

    return engine.report(model_id)


def main():
    parser = argparse.ArgumentParser(description="快速采样评测")
    parser.add_argument("--model", nargs=2, default=None, metavar=("TYPE", "NAME"))
    parser.add_argument("--samples", type=int, default=40,
                        help="采样点数量 (默认 40)")
    parser.add_argument("--size", type=int, nargs="+", default=[30, 50],
                        help="网格大小 (默认 30 50)")
    parser.add_argument("--density", type=float, nargs="+", default=[0.15, 0.25],
                        help="障碍密度 (默认 0.15 0.25)")
    parser.add_argument("--output", type=str, default="output/quick_report")
    args = parser.parse_args()

    # 构建模型列表
    if args.model:
        mtype, mname = args.model
        if mtype == "deepseek":
            from adapters.openai_adapter import DeepSeekAdapter
            api_key = os.environ.get("OPENAI_API_KEY", "")
            models = [DeepSeekAdapter(model=mname, api_key=api_key, metacognitive="vote")]
        elif mtype == "openai":
            from adapters.openai_adapter import OpenAIAdapter
            models = [OpenAIAdapter(model=mname)]
        elif mtype == "ollama":
            from adapters.openai_adapter import OllamaAdapter
            models = [OllamaAdapter(model=mname)]
        else:
            print(f"未知模型类型: {mtype}")
            sys.exit(1)
    else:
        models = [
            RandomModel(silence_prob=0.0),
            RandomModel(silence_prob=0.5),
            HeuristicModel(),
            MLPModel(),
        ]

    from metrics.scorer import score_model, generate_markdown_report, export_json
    from chern_engine import ChernReport

    all_scored = []
    t0 = time.time()

    for model in models:
        model_reports = []

        for size in args.size:
            for density in args.density:
                n = args.samples // (len(args.size) * len(args.density))
                grid, samples, threshold = sample_decision_points(
                    size, density, n, seed=hash(model.model_id) % 100000
                )

                n_high = sum(1 for s in samples if s["is_high"])
                print(f"  [{model.model_id}] grid={size}x{size} d={density:.2f} "
                      f"samples={len(samples)} (高熵{n_high}, 阈值{threshold:.3f})")

                engine = ChernEngine(entropy_threshold=threshold)
                report = evaluate_model_on_samples(model, grid, size, samples, engine)
                model_reports.append(report)

        scored = score_model(model_reports, coverage=0.25)
        all_scored.append(scored)
        print(f"  => score={scored.metacognition_score} sig={scored.signal_ratio:.1%} "
              f"sil={scored.silence_rate:.1%} over={scored.overconfidence_rate:.1%}")

    elapsed = time.time() - t0
    print(f"\n评测完成, 耗时 {elapsed:.0f}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    md_path = generate_markdown_report(all_scored, f"{args.output}.md")
    export_json(all_scored, f"{args.output}.json")
    print(f"报告: {os.path.abspath(md_path)}")


if __name__ == "__main__":
    main()
