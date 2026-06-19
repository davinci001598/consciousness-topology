"""调试：直接用 DeepSeek API 看看返回了什么"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causal_scene import generate_causal_scene
from causal_eval import make_causal_state, KEY_START, DOOR_START
from adapters.causal_llm_adapter import CausalLLMAdapter

adapter = CausalLLMAdapter(model="deepseek-chat", concise=True)

# 场景 B, chain=2, trial=0
scene = generate_causal_scene(30, 2, wall_openings=3, dead_ends=3, alcove_keys=2,
                              seed=42 + 2*100 + 0)  # seed from evaluator
state = make_causal_state(scene.start, scene, set(), 0)

print("=== 状态 ===")
print(f"pos={state['pos']}, goal={state['goal']}")
print(f"inventory={state['inventory']}, step={state['step']}")
print(f"legal_actions={state['legal_actions']}")
print(f"nearby={state['nearby_cells']}")
print(f"nearby labels:")
for d, cell in state['nearby_cells'].items():
    print(f"  {d}: {cell}")

print("\n=== 精简 prompt ===")
msg = adapter._build_user_prompt(state)
print(msg)

print("\n=== 全量 prompt ===")
adapter_full = CausalLLMAdapter(model="deepseek-chat", concise=False)
msg_full = adapter_full._build_user_prompt(state)
print(msg_full)

print("\n=== 实际 API 调用 ===")
import openai
client = openai.OpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
)

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": adapter._SYSTEM_CONCISE},
        {"role": "user", "content": msg},
    ],
    temperature=0.0,
    max_tokens=20,
)
text = resp.choices[0].message.content
print(f"RAW response: [{text}]")

# 解析
from causal_llm_adapter import DIRS, DIR_NAMES
action = adapter._parse_direction(text.strip().lower(), state["legal_actions"])
print(f"parsed action: {action}")
