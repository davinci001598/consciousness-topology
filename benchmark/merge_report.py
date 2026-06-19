"""合并内置模型（from JSON）+ DeepSeek最新结果 → 统一报告"""
import json, os, sys
_BENCH = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BENCH)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _BENCH)
from metrics.scorer import score_model, generate_markdown_report, ScoredResult

# 读取内置模型结果
json_path = os.path.join(_BENCH, "output", "quick_report.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

scored_models = []
for d in data:
    s = ScoredResult(
        model_id=d["model_id"],
        signal_ratio=d["signal_ratio"],
        silence_rate=d["silence_rate"],
        overconfidence_rate=d["overconfidence_rate"],
        coverage=d.get("coverage", 0.25),
    )
    scored_models.append(s)

# 加入 DeepSeek 结果
from api import make_state
os.environ["OPENAI_API_KEY"] = "sk-1f06600b8286443294f8184d24f76777"
from adapters.openai_adapter import DeepSeekAdapter
from quick_eval import generate_map, sample_decision_points
from chern_engine import ChernEngine

model = DeepSeekAdapter(model="deepseek-chat", metacognitive="vote")
all_reports = []

for size in [20, 30, 50]:
    for density in [0.15, 0.25]:
        grid, samples, threshold = sample_decision_points(size, density, 10, seed=42)
        engine = ChernEngine(entropy_threshold=threshold)
        for s in samples:
            state = make_state(s["pos"], size, grid, (size-1, size-1))
            action = model.predict(state)
            engine.evaluate(model.model_id, s["pos"], size, grid, (size-1, size-1), action)
        report = engine.report(model.model_id)
        all_reports.append(report)

from metrics.scorer import ScoredResult
ds_scored = score_model(all_reports, coverage=0.25)
print(f"DeepSeek(vote): score={ds_scored.metacognition_score} sig={ds_scored.signal_ratio:.1%} "
      f"sil={ds_scored.silence_rate:.1%} over={ds_scored.overconfidence_rate:.1%}")

scored_models.append(ds_scored)
scored_models.sort(key=lambda s: s.metacognition_score, reverse=True)

output_path = os.path.join(_BENCH, "output", "full_report.md")
md = generate_markdown_report(scored_models, output_path)
print(f"报告: {os.path.abspath(md)}")
