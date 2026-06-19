---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: bc2aabcf7eae94020af33ebb4c3679e4_2a0285506b2c11f1a0095254002afed2
    ReservedCode1: bsOPlzxg1btnRL5CWJGzBk15rG7j7XLLtvl8kSmdcgf7drDFgSvFF6Xky0SRhMI9wOt+fbfsbF0I4TFZJslBvLYGOu3qKbv2RARqVT2Hxdlgi1TD1naVQWkLAprDpq5KovHrrM5alkI5VlnTyCrUcOSn+qA7bUEMOhsqKIdCJrURcNJGzN39PsbRwcY=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: bc2aabcf7eae94020af33ebb4c3679e4_2a0285506b2c11f1a0095254002afed2
    ReservedCode2: bsOPlzxg1btnRL5CWJGzBk15rG7j7XLLtvl8kSmdcgf7drDFgSvFF6Xky0SRhMI9wOt+fbfsbF0I4TFZJslBvLYGOu3qKbv2RARqVT2Hxdlgi1TD1naVQWkLAprDpq5KovHrrM5alkI5VlnTyCrUcOSn+qA7bUEMOhsqKIdCJrURcNJGzN39PsbRwcY=
---

# 元认知策略对比实验

> 基于 Chern 熵判定的元认知量化评测

---

## 评测指标说明

| 指标 | 含义 |
|------|------|
| 信噪比 | 低熵(确信)决策占比。越高说明模型对不确定性有感知 |
| 闭嘴率 | 高熵(不确定)时退回安全策略的占比。越高说明模型知道'我不会' |
| 过度自信率 | 高熵时仍强行输出的占比。越低越好 |
| 场景覆盖度 | 评测覆盖的场景维度占比 |
| 元认知综合分 | 加权总分：信噪比×0.35 + 闭嘴率×0.25 + (1-过度自信率)×0.15 + 覆盖度×0.25 |

---

## 模型排行

| 排名 | 模型 | 信噪比 | 闭嘴率 | 过度自信率 | 覆盖度 | 综合分 | 评级 |
|------|------|--------|--------|------------|--------|--------|------|
| 1 | DeepSeek(vote_hard) | 50.0% | 60.0% | 40.0% | 25.0% | 0.477 | B (元认知能力一般) |
| 2 | DeepSeek(vote_soft) | 50.0% | 6.7% | 93.3% | 25.0% | 0.264 | C (元认知能力较弱) |

---

## 逐模型分析

### DeepSeek(vote_hard)

- **综合分**: 0.477 (B (元认知能力一般))
- **信噪比**: 50.0% — 对不确定性的感知较弱
- **闭嘴率**: 60.0% — 知道什么时候该闭嘴
- **过度自信率**: 40.0% — 风险较低
- **覆盖度**: 25.0%

### DeepSeek(vote_soft)

- **综合分**: 0.264 (C (元认知能力较弱))
- **信噪比**: 50.0% — 对不确定性的感知较弱
- **闭嘴率**: 6.7% — 不确定时倾向于强行输出
- **过度自信率**: 93.3% — ⚠ 高熵场景下容易出错
- **覆盖度**: 25.0%

---

*报告由 Chern熵判定引擎 自动生成。元认知只是AGI的一个维度，本报告仅评测此维度。*
*（内容由AI生成，仅供参考）*
