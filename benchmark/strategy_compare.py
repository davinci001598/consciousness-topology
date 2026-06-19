#!/usr/bin/env python3
"""
元认知强化实验
==============
测试三种策略提升 DeepSeek 的闭嘴率:
  1. vote_hard: temperature=1.0, 5采样, 需全票一致
  2. prompt_nudge: 提示词告知"不确定时说 UNSURE", 单次调用
  3. vote_soft:  temperature=0.8, 3采样, 多数一致即可
"""

import os, sys, time, random
_BENCH = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BENCH)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _BENCH)

from api import make_state
from quick_eval import generate_map, sample_decision_points
from chern_engine import ChernEngine
from metrics.scorer import score_model, ScoredResult

os.environ["OPENAI_API_KEY"] = "sk-1f06600b8286443294f8184d24f76777"
from openai import OpenAI

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIR_NAMES = ["up", "right", "down", "left"]

client = OpenAI(base_url="https://api.deepseek.com/v1", api_key=os.environ["OPENAI_API_KEY"])

def build_prompt(state):
    x, y = state["pos"]
    gx, gy = state["goal"]
    size = state["size"]
    la = state["legal_actions"]
    dir_desc = [f"{DIRS.index(d)}={DIR_NAMES[DIRS.index(d)]}" for d in la]
    return (
        f"You are navigating a {size}x{size} grid. "
        f"Current: ({x},{y}). Goal: ({gx},{gy}). "
        f"Legal: {', '.join(dir_desc)}. "
        f"Reply with exactly one direction number (0=up,1=right,2=down,3=left)."
    )

def call_model(prompt, temp=0.0, max_tokens=50):
    resp = client.chat.completions.create(
        model="deepseek-chat", messages=[{"role":"user","content":prompt}],
        temperature=temp, max_tokens=max_tokens)
    return resp.choices[0].message.content.strip().lower()

def parse_direction(text, legal):
    for i, name in enumerate(DIR_NAMES):
        if name in text: return DIRS[i] if DIRS[i] in legal else None
    try:
        idx = int(text)
        if 0 <= idx < 4: return DIRS[idx] if DIRS[idx] in legal else None
    except: pass
    return None

# ── 策略定义 ──

def predict_vote_hard(state):
    """temperature=1.0, 5采样, 全票一致才输出"""
    prompt = build_prompt(state)
    votes = []
    for _ in range(5):
        try:
            text = call_model(prompt, temp=1.0)
            d = parse_direction(text, state["legal_actions"])
            if d: votes.append(d)
        except: pass
    if len(votes) >= 3 and all(v == votes[0] for v in votes):
        return votes[0]
    return None

def predict_prompt_nudge(state):
    """提示词告知模型可以说不确定"""
    x, y = state["pos"]
    gx, gy = state["goal"]
    size = state["size"]
    la = state["legal_actions"]
    dir_desc = [f"{DIRS.index(d)}={DIR_NAMES[DIRS.index(d)]}" for d in la]
    prompt = (
        f"Grid {size}x{size}. You at ({x},{y}), goal ({gx},{gy}). "
        f"Legal: {', '.join(dir_desc)}. "
        f"If you are UNCERTAIN about the best direction, reply with just 'UNSURE'. "
        f"Otherwise reply with a number 0-3."
    )
    try:
        text = call_model(prompt, temp=0.0)
    except:
        return None
    if "unsure" in text:
        return None
    return parse_direction(text, la)

def predict_vote_soft(state):
    """temperature=0.8, 3采样, 多数票即可"""
    prompt = build_prompt(state)
    votes = []
    for _ in range(3):
        try:
            text = call_model(prompt, temp=0.8)
            d = parse_direction(text, state["legal_actions"])
            if d: votes.append(d)
        except: pass
    if not votes: return None
    # 多数票
    from collections import Counter
    winner, count = Counter(votes).most_common(1)[0]
    if count >= 2: return winner
    return None

# ── 评测 ──

STRATEGIES = {
    "vote_hard": (predict_vote_hard, "temp=1.0, 5采样, 全票"),
    "vote_soft": (predict_vote_soft, "temp=0.8, 3采样, 多数"),
}

all_scored = []
for name, (predict_fn, desc) in STRATEGIES.items():
    model_id = f"DeepSeek({name})"
    all_reports = []

    for size in [20, 30, 50]:
        for density in [0.15, 0.25]:
            grid, samples, threshold = sample_decision_points(size, density, 10, seed=42)
            engine = ChernEngine(entropy_threshold=threshold)
            n_high = sum(1 for s in samples if s["is_high"])
            print(f"  [{model_id}] {size}x{size} d={density:.2f} samples={len(samples)} (高熵{n_high}, 阈值{threshold:.3f})")

            for s in samples:
                state = make_state(s["pos"], size, grid, (size-1, size-1))
                action = predict_fn(state)
                engine.evaluate(model_id, s["pos"], size, grid, (size-1, size-1), action)

            report = engine.report(model_id)
            all_reports.append(report)

    scored = score_model(all_reports, coverage=0.25)
    all_scored.append(scored)
    print(f"  => score={scored.metacognition_score} sig={scored.signal_ratio:.1%} "
          f"sil={scored.silence_rate:.1%} over={scored.overconfidence_rate:.1%}")

from metrics.scorer import generate_markdown_report
output_path = os.path.join(_BENCH, "output", "strategy_report.md")
generate_markdown_report(all_scored, output_path, "元认知策略对比实验")
print(f"\n报告: {output_path}")
