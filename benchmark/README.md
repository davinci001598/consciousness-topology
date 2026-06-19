---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: bc2aabcf7eae94020af33ebb4c3679e4_3935b9a96b9111f1a99c5254007bceed
    ReservedCode1: ytYzjeLV6b2GfosqK5Vk563cpK7W8DOyk4R7QVY78PpUwhHt6cBdD48XqXg7Ny+H9SYCcH/EhJrX6QCptuXKxuxMyMavOtQWHhTNEH3XDCLdktnCEUNHJJOkzLY5drArbdyGPvERE5x7m967MnUoAK3EMovn7k/L2Fgy+m2PURTacmahxke9E3JfgSg=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: bc2aabcf7eae94020af33ebb4c3679e4_3935b9a96b9111f1a99c5254007bceed
    ReservedCode2: ytYzjeLV6b2GfosqK5Vk563cpK7W8DOyk4R7QVY78PpUwhHt6cBdD48XqXg7Ny+H9SYCcH/EhJrX6QCptuXKxuxMyMavOtQWHhTNEH3XDCLdktnCEUNHJJOkzLY5drArbdyGPvERE5x7m967MnUoAK3EMovn7k/L2Fgy+m2PURTacmahxke9E3JfgSg=
---

# Benchmark: 意识拓扑引擎复现指南

## 子项目

| 项目 | 文件 | 说明 |
|------|------|------|
| **路径规划** | `benchmark.py` | 拓扑场 vs 感知机路径规划基准 |
| **因果链迷宫** | `causal_eval.py` | 因果推理能力评测（钥匙→门因果链） |

---

## 一、路径规划基准

### 快速开始

```bash
pip install numpy           # 仅 MLP+Topo / Topo+MLP 需要
python benchmark.py         # 完整基准测试 (10x10/20x20/50x50, 全部密度)
python benchmark.py --quick # 快速验证 (仅 10x10, 30% 规模, ~5分钟)
python benchmark.py --grid 50x50 --density 0.30  # 单条件测试
```

## 实验设计

| 参数 | 取值 |
|------|------|
| 网格大小 | 10x10, 20x20, 50x50 |
| 障碍密度 | 2%, 5%, 10%, 20%, 30% |
| 每条件地图数 | 10（固定种子保证可复现） |
| 每地图轮数 | 30 |
| 策略 | Random, MLP, Topological, MLP+Topo, Topo+MLP |
| 指标 | 到达率, 路径效率, 平均步数, 碰撞率 |

## 输出格式

`results.json` 结构:

```json
{
  "10x10_d02": {
    "Random": { "summary": { "reach_rate": 1.0, "avg_efficiency": 0.652, ... } },
    "MLP":    { "summary": { "reach_rate": 1.0, "avg_efficiency": 0.798, ... } },
    ...
  }
}
```

## 策略简介

| 策略 | 核心机制 |
|------|---------|
| **Random** | 启发式 + 30% 随机覆盖 |
| **MLP** | 2 层感知机噪声投票，20% 覆盖 |
| **Topological** | Chern=-1 量子行走 → 熵门控 → 低熵采纳 / 高熵回退启发式 |
| **MLP+Topo** | MLP 做主引擎，拓扑场低熵时覆盖 |
| **Topo+MLP** | 拓扑场做主引擎，高熵时退回 MLP |

## 完整结果

`results.json` 已包含全部 15 条件 x 5 策略的汇总数据。

---

## 二、因果链迷宫评测

### 核心概念

因果链迷宫（Causal Chain Maze）评测模型的**多步因果推理**能力：场景中包含若干钥匙和对应的门，模型必须按正确顺序拾取钥匙才能打开门到达终点。每一步因果推理形成一条链接 — 链越长、推理深度要求越高。

### 评分体系

| 指标 | 含义 | 权重 |
|------|------|------|
| 完整推理率 | 完整因果链 / 总因果链 | 40% |
| 跳步率 | 违规开门次数 / 总门次数 | 25% |
| 步数效率 | BFS 最优步数 / 实际步数 | 25% |
| 到达率 | 成功到达终点比例 | 10% |

等级：A (0.85+) / B (0.70-0.85) / C (0.50-0.70) / D (0.30-0.50) / E (<0.30)

### 场景

| 场景 | 描述 | 迷宫 | 钥匙凹室 | 死胡同 | 开门次数 |
|------|------|------|----------|--------|----------|
| A | 简单走廊 | 15×15 | 2 | 2 | 2 |
| B | 复杂迷宫 | 20×20 | 2 | 3 | 3 |

### 快速开始

```bash
# 安装依赖
pip install openai numpy

# 运行 Greedy + BFS 基线（无需 API）
python causal_eval.py

# 运行 LLM 评测（需要 DeepSeek API Key）
python run_llm_causal_eval.py -m deepseek-chat -c b --chain-lengths 2,3,4 --trials 2
python run_llm_causal_eval.py -m deepseek-chat -c a --chain-lengths 2 --trials 2 --full-prompt

# 传 API Key（推荐，避免环境变量失效）
python run_llm_causal_eval.py -m deepseek-chat -c b --api-key sk-xxx
```

### 评测结果：场景B

| 因果链长度 | Greedy | BFS (上限) | DeepSeek-chat |
|-----------|--------|-----------|---------------|
| chain=2 (2 对钥匙-门) | C — 0.410 | A — 0.972 | **A — 0.972** |
| chain=3 (3 对钥匙-门) | C — 0.410 | A — 0.972 | **A — 0.972** |
| chain=4 (4 对钥匙-门) | C — 0.350 | A — 0.948 | **A — 0.948** |

**结论**：DeepSeek-chat 在 chain=2/3/4 全部达到 BFS 最优上限，27 步到达终点，完整推理率 100%、跳步率 0%。多步因果推理能力与最优搜索算法持平。

### 文件说明

| 文件 | 作用 |
|------|------|
| `causal_scene.py` | 因果链迷宫生成（钥匙/门/凹室布局） |
| `causal_eval.py` | Greedy + BFS 基线评测 |
| `run_llm_causal_eval.py` | LLM 评测脚本入口 |
| `adapters/causal_llm_adapter.py` | LLM 适配器（因果状态 → prompt） |
| `adapters/openai_adapter.py` | OpenAI/DeepSeek API 封装 |
| `run_eval.py` | 批量场景/模型/链长调度 |
| `strategy_compare.py` | 多策略对比汇总 |
| `merge_report.py` | 合并多轮评测报告 |
| `quick_eval.py` | 快速单条件评测 |
*（内容由AI生成，仅供参考）*
