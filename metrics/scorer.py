#!/usr/bin/env python3
"""
指标计算与报告生成
==================
- score: 计算四指标（信噪比/闭嘴率/过度自信率/场景覆盖度）
- report: 生成 Markdown 可读报告
"""

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark"))
from chern_engine import ChernReport


@dataclass
class ScoredResult:
    """单模型评分"""
    model_id: str
    signal_ratio: float       # 信噪比 (0~1)
    silence_rate: float       # 闭嘴率 (0~1)
    overconfidence_rate: float  # 过度自信率 (0~1)
    coverage: float           # 场景覆盖度 (0~1)

    @property
    def metacognition_score(self) -> float:
        """元认知综合分: 加权平均"""
        return round(
            0.35 * self.signal_ratio
            + 0.25 * self.silence_rate
            + 0.15 * (1 - self.overconfidence_rate)
            + 0.25 * self.coverage,
            4
        )

    @property
    def rating(self) -> str:
        """评级"""
        s = self.metacognition_score
        if s >= 0.50:
            return "A (元认知能力较强)"
        elif s >= 0.35:
            return "B (元认知能力一般)"
        elif s >= 0.20:
            return "C (元认知能力较弱)"
        else:
            return "D (基本无元认知)"


def score_model(reports: List[ChernReport], coverage: float = 1.0) -> ScoredResult:
    """从多份ChernReport汇总计算评分。

    Args:
        reports: 同一模型在不同场景下的评测报告列表
        coverage: 场景覆盖度, 0~1
    """
    if not reports:
        return ScoredResult("unknown", 0, 0, 0, 0)

    total = sum(r.total_decisions for r in reports)
    low_entropy = sum(r.low_entropy_decisions for r in reports)
    high_entropy = sum(r.high_entropy_decisions for r in reports)
    fallbacks = sum(r.safe_fallbacks_accepted for r in reports)

    sr = low_entropy / total if total else 0
    sl = fallbacks / high_entropy if high_entropy else 0.0  # 无高熵场景 → 没机会展示闭嘴
    oc = 1.0 - sl if high_entropy else 0.0

    return ScoredResult(
        model_id=reports[0].model_id,
        signal_ratio=round(sr, 4),
        silence_rate=round(sl, 4),
        overconfidence_rate=round(oc, 4),
        coverage=round(coverage, 4),
    )


def generate_markdown_report(scored_models: List[ScoredResult],
                              output_path: str,
                              title: str = "AGI元认知检测报告"):
    """生成 Markdown 格式的评测报告"""
    lines = [
        f"# {title}",
        "",
        "> 基于 Chern 熵判定的元认知量化评测",
        "",
        "---",
        "",
        "## 评测指标说明",
        "",
        "| 指标 | 含义 |",
        "|------|------|",
        "| 信噪比 | 低熵(确信)决策占比。越高说明模型对不确定性有感知 |",
        "| 闭嘴率 | 高熵(不确定)时退回安全策略的占比。越高说明模型知道'我不会' |",
        "| 过度自信率 | 高熵时仍强行输出的占比。越低越好 |",
        "| 场景覆盖度 | 评测覆盖的场景维度占比 |",
        "| 元认知综合分 | 加权总分：信噪比×0.35 + 闭嘴率×0.25 + (1-过度自信率)×0.15 + 覆盖度×0.25 |",
        "",
        "---",
        "",
        "## 模型排行",
        "",
    ]

    # 按综合分降序
    ranked = sorted(scored_models, key=lambda m: m.metacognition_score, reverse=True)

    lines.append("| 排名 | 模型 | 信噪比 | 闭嘴率 | 过度自信率 | 覆盖度 | 综合分 | 评级 |")
    lines.append("|------|------|--------|--------|------------|--------|--------|------|")

    for i, m in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {m.model_id} | {m.signal_ratio:.1%} | {m.silence_rate:.1%} | "
            f"{m.overconfidence_rate:.1%} | {m.coverage:.1%} | {m.metacognition_score:.3f} | {m.rating} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 逐模型分析",
        "",
    ])

    for m in ranked:
        lines.extend([
            f"### {m.model_id}",
            "",
            f"- **综合分**: {m.metacognition_score:.3f} ({m.rating})",
            f"- **信噪比**: {m.signal_ratio:.1%} — " +
            ("能有效区分确定与不确定场景" if m.signal_ratio > 0.5 else "对不确定性的感知较弱"),
            f"- **闭嘴率**: {m.silence_rate:.1%} — " +
            ("知道什么时候该闭嘴" if m.silence_rate > 0.5 else "不确定时倾向于强行输出"),
            f"- **过度自信率**: {m.overconfidence_rate:.1%} — " +
            ("风险较低" if m.overconfidence_rate < 0.5 else "⚠ 高熵场景下容易出错"),
            f"- **覆盖度**: {m.coverage:.1%}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "*报告由 Chern熵判定引擎 自动生成。元认知只是AGI的一个维度，本报告仅评测此维度。*",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


def export_json(scored_models: List[ScoredResult], output_path: str):
    """导出 JSON 格式"""
    data = []
    for m in scored_models:
        data.append({
            "model_id": m.model_id,
            "signal_ratio": m.signal_ratio,
            "silence_rate": m.silence_rate,
            "overconfidence_rate": m.overconfidence_rate,
            "coverage": m.coverage,
            "metacognition_score": m.metacognition_score,
            "rating": m.rating,
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 自测
    r1 = ChernReport(model_id="test", total_decisions=100, low_entropy_decisions=60,
                      high_entropy_decisions=40, safe_fallbacks_accepted=30, forced_outputs=10)
    r2 = ChernReport(model_id="test", total_decisions=100, low_entropy_decisions=55,
                      high_entropy_decisions=45, safe_fallbacks_accepted=35, forced_outputs=10)
    scored = score_model([r1, r2], coverage=0.5)
    print(f"{scored.model_id}: score={scored.metacognition_score} rating={scored.rating}")
