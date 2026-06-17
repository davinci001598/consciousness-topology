# Benchmark: 意识拓扑引擎复现指南

## 快速开始

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
