#!/usr/bin/env python3
"""
因果链迷宫 —— LLM 评测运行脚本
================================
用法:
    # 用环境变量 OPENAI_API_KEY
    python run_llm_causal_eval.py --model gpt-4o-mini

    # 显式传 key
    python run_llm_causal_eval.py --model deepseek-chat --api-key sk-xxx

    # 指定场景复杂度
    python run_llm_causal_eval.py --model gpt-4o-mini --complexity b

    # 对比多个模型
    python run_llm_causal_eval.py --model gpt-4o-mini --model deepseek-chat

    # 本地 Ollama
    python run_llm_causal_eval.py --model qwen2.5 --base-url http://localhost:11434/v1 --api-key ollama
"""

import argparse
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from causal_eval import CausalEvaluator, compute_metrics, GreedyModel, BFSModel
from adapters.causal_llm_adapter import CausalLLMAdapter


# ═══════════════════════════════════════════════
# 预设场景
# ═══════════════════════════════════════════════

SCENES = {
    "a": {  # 纯直线走廊（baseline）
        "desc": "场景A: 纯直线走廊",
        "wall_openings": 0, "dead_ends": 0, "alcove_keys": 0,
    },
    "b": {  # 复杂迷宫（区分推理能力）
        "desc": "场景B: 复杂迷宫 (开口3/死胡同3/凹室钥匙2)",
        "wall_openings": 3, "dead_ends": 3, "alcove_keys": 2,
    },
}


def run_model(model, scene_name, scene_cfg, max_steps=300,
              chain_lengths=None, trials=4):
    """运行单个模型在单个场景上的评测"""
    if chain_lengths is None:
        chain_lengths = [2, 3, 4]

    evaluator = CausalEvaluator(max_steps=max_steps)
    print(f"\n  {scene_cfg['desc']}")
    results = evaluator.evaluate(
        model,
        chain_lengths=chain_lengths,
        trials_per_config=trials,
        wall_openings=scene_cfg["wall_openings"],
        dead_ends=scene_cfg["dead_ends"],
        alcove_keys=scene_cfg["alcove_keys"],
    )
    return results


def print_metrics(model_id, metrics):
    """打印评测指标"""
    print(f"  完整推理率: {metrics['complete_rate']:.1%}")
    print(f"  跳步率:     {metrics['skip_rate']:.1%}")
    print(f"  步数效率:   {metrics['step_efficiency']:.1%}")
    print(f"  到达率:     {metrics['goal_rate']:.1%}")
    print(f"  综合分:     {metrics['composite']:.3f} ({metrics['rating']})")


def main():
    parser = argparse.ArgumentParser(description="因果链迷宫 LLM 评测")
    parser.add_argument("--model", "-m", action="append", dest="models",
                        help="被测模型名，可多次指定 (如 -m gpt-4o-mini -m deepseek-chat)")
    parser.add_argument("--api-key", default=None,
                        help="API Key (也可通过环境变量 OPENAI_API_KEY 传入)")
    parser.add_argument("--base-url", default=None,
                        help="自定义 API Base URL (deepseek 自动使用官方地址)")
    parser.add_argument("--temperature", "-t", type=float, default=0.0,
                        help="采样温度，默认 0.0")
    parser.add_argument("--complexity", "-c", choices=["a", "b", "both"], default="both",
                        help="场景复杂度: a=简单直线, b=复杂迷宫, both=两者 (默认)")
    parser.add_argument("--chain-lengths", nargs="+", type=int, default=[2, 3, 4],
                        help="因果链长度列表 (默认 2 3 4)")
    parser.add_argument("--trials", type=int, default=4,
                        help="每组配置重复次数 (默认 4)")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="单次最大步数 (默认 300)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="不跑 baseline (Greedy/BFS)")
    parser.add_argument("--concise", action="store_true", default=True,
                        help="使用精简 prompt (默认)")
    parser.add_argument("--full-prompt", dest="concise", action="store_false",
                        help="使用全量 prompt (包含完整因果链描述)")

    args = parser.parse_args()

    if not args.models:
        parser.error("至少指定一个 --model")

    # 场景选择
    if args.complexity == "both":
        scene_names = ["a", "b"]
    else:
        scene_names = [args.complexity]

    # ── Baseline ──
    if not args.no_baseline:
        print("=" * 60)
        print(" BASELINE 对照")
        print("=" * 60)
        baselines = [GreedyModel(), BFSModel()]
        for bl in baselines:
            print(f"\n── {bl.model_id} ──")
            for sn in scene_names:
                cfg = SCENES[sn]
                results = run_model(bl, sn, cfg, args.max_steps,
                                    args.chain_lengths, args.trials)
                m = compute_metrics(results)
                print(f"  {cfg['desc']}")
                print_metrics(bl.model_id, m)

    # ── LLM 模型 ──
    for model_name in args.models:
        print("\n" + "=" * 60)
        print(f" 被测模型: {model_name}")
        print("=" * 60)

        adapter = CausalLLMAdapter(
            model=model_name,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            concise=args.concise,
        )

        for sn in scene_names:
            cfg = SCENES[sn]
            results = run_model(adapter, sn, cfg, args.max_steps,
                                args.chain_lengths, args.trials)
            m = compute_metrics(results)
            print_metrics(model_name, m)

    print("\n评测完成。")


if __name__ == "__main__":
    main()
