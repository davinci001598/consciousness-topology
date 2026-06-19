"""诊断 DeepSeek 自评置信度分布"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import make_state
from quick_eval import generate_map, sample_decision_points

os.environ["OPENAI_API_KEY"] = "sk-1f06600b8286443294f8184d24f76777"
from adapters.openai_adapter import DeepSeekAdapter

model = DeepSeekAdapter(model="deepseek-chat", metacognitive=True, confidence_threshold=99)

import random
results = []
for size, den in [(30, 0.15), (50, 0.25)]:
    grid, samples, th = sample_decision_points(size, den, 10, seed=42)
    for s in samples:
        x, y = s["pos"]
        is_high = s["is_high"]
        state = make_state((x,y), size, grid, (size-1,size-1))
        client = model._get_client()
        prompt = model._build_assessment_prompt(state)
        try:
            resp = client.chat.completions.create(model=model.model, messages=[{"role":"user","content":prompt}],
                                                   temperature=0.0, max_tokens=10)
            text = resp.choices[0].message.content.strip()
        except Exception as e:
            text = f"ERROR: {e}"
        results.append(f"  [{size}x{size}] high={is_high} ent={s['entropy']:.2f} → '{text}'")
        print(results[-1])

print(f"\n总计 {len(results)} 次")
