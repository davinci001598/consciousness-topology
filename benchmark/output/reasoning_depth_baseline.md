---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: bc2aabcf7eae94020af33ebb4c3679e4_13e3469b6b3111f1a99c5254007bceed
    ReservedCode1: FBwgl24fyJuyDqo+wOuLJSfFgMNG3MrrlE0dOpzmIeAmErCMgwgZ3df78JBFrcxw1Q8MrLxIrtpYPN5KOFPduqdXaBD9jtpYBqG0EE2NREByHX9T/mO7bnT3AN1oQzVm3CEeNObuOXL8/UZEFRjRCViUx48Ho2yUieH4H+h6tNage+GWL2sTOvCl4Ew=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: bc2aabcf7eae94020af33ebb4c3679e4_13e3469b6b3111f1a99c5254007bceed
    ReservedCode2: FBwgl24fyJuyDqo+wOuLJSfFgMNG3MrrlE0dOpzmIeAmErCMgwgZ3df78JBFrcxw1Q8MrLxIrtpYPN5KOFPduqdXaBD9jtpYBqG0EE2NREByHX9T/mO7bnT3AN1oQzVm3CEeNObuOXL8/UZEFRjRCViUx48Ho2yUieH4H+h6tNage+GWL2sTOvCl4Ew=
---

# 推理深度评测 - 基线报告

## 场景设计

**因果链迷宫**：线性走廊中按序放置 N 对钥匙和门，必须拾取钥匙 i 才能通过门 i，最终到达终点。

```
grid 编码: 0=空地, 1=障碍, 2=起点, 3=终点, 10-19=钥匙, 20-29=门
```

## 评价指标

| 指标 | 权重 | 含义 |
|------|------|------|
| 完整推理率 | 0.40 | 完成的因果对 / 总因果对 |
| (1-跳步率) | 0.25 | 尝试过未解锁门次数 / 总门交互 |
| 步数效率 | 0.20 | 未达目标=0；达目标=最优步数/模型步数 |
| 到达率 | 0.15 | 到达终点的场景占比 |

## 基线结果（wall_openings=0，直线走廊）

| 模型 | 完整推理率 | 跳步率 | 步数效率 | 到达率 | 综合分 | 评级 |
|------|-----------|--------|---------|--------|--------|------|
| BFS 上限 | 100.0% | 0.0% | 95.8% | 100.0% | 0.992 | A |
| Greedy | 100.0% | 0.0% | 95.8% | 100.0% | 0.992 | A |
| Random | 83.3% | 0.0% | 3.7% | 33.3% | 0.641 | B |
| Random+沉默30% | 0.0% | 0.0% | 0.0% | 0.0% | 0.250 | D |

## 文件结构

```
benchmark/
  causal_scene.py   - 场景生成器（线性走廊 + 钥匙/门机制）
  causal_eval.py    - 评测器（4 指标 + 综合评分 + 内置基准模型）
  output/
    reasoning_depth_baseline.md  - 本报告
```
*（内容由AI生成，仅供参考）*
