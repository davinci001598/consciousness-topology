#!/usr/bin/env python3
"""
因果链迷宫 LLM 适配器
=====================
将 causal_eval.make_causal_state() 产出的状态转换为文本 prompt，
调用任意 OpenAI 兼容 API，解析回复为移动方向。

可接入模型:
  - gpt-4o / gpt-4o-mini / gpt-3.5-turbo (OpenAI)
  - deepseek-chat / deepseek-reasoner (DeepSeek)
  - claude-3-5-sonnet 等 (通过兼容代理)
  - Ollama 本地模型 (llama3/qwen2.5 等)
  - LM Studio 本地模型

用法:
    from causal_llm_adapter import CausalLLMAdapter
    model = CausalLLMAdapter(model="gpt-4o-mini", api_key="sk-...")
    action = model.predict(state)  # → (dx, dy) or None
"""

import os
import sys
from typing import Optional, Tuple

# 确保 benchmark 目录在 path 中
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from api import ModelInterface

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIR_NAMES = ["up", "right", "down", "left"]

# 因果链场景的 grid 编码
KEY_START = 10
DOOR_START = 20
MAX_KEYS = 10

_CELL_SYMBOLS = {
    0: "空地", 1: "墙", 2: "起点", 3: "终点",
}


def _cell_label(cell: int) -> str:
    """将 grid 数字转成人类可读标签"""
    if cell in _CELL_SYMBOLS:
        return _CELL_SYMBOLS[cell]
    if KEY_START <= cell < DOOR_START:
        return f"密钥{cell - KEY_START}"
    if DOOR_START <= cell < DOOR_START + MAX_KEYS:
        return f"门{cell - DOOR_START}"
    return f"未知({cell})"


class CausalLLMAdapter(ModelInterface):
    """因果链迷宫 → LLM 适配器

    参数:
      - model:         模型名 (gpt-4o-mini / deepseek-chat / ...)
      - base_url:      API 地址，默认 openai 官方
      - api_key:       API Key，也可通过环境变量 OPENAI_API_KEY / DEEPSEEK_API_KEY 传入
      - temperature:   采样温度，默认 0.0
      - concise:       提示词精简模式（省略全局场景描述，仅给局部信息），默认 True
      - system_prompt: 自定义 system 消息，None 则用内置因果推理引导
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = None,
        api_key: str = None,
        temperature: float = 0.0,
        concise: bool = True,
        system_prompt: str = None,
    ):
        self.model = model
        self.temperature = temperature
        self.concise = concise
        self.system_prompt = system_prompt

        # 自动推断 base_url
        if base_url:
            self.base_url = base_url
        elif "deepseek" in model.lower():
            self.base_url = "https://api.deepseek.com/v1"
        else:
            self.base_url = "https://api.openai.com/v1"

        # 自动推断 api_key
        if api_key:
            self.api_key = api_key
        elif "deepseek" in model.lower():
            self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")

        self._client = None
        self._history = []  # 最近的位置历史，用于避免回头

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client

    # ── predict ──────────────────────────────────────────

    def predict(self, state: dict) -> Optional[Tuple[int, int]]:
        """根据因果链状态返回下一步动作"""
        # 记录位置历史
        pos = state["pos"]
        self._history.append(tuple(pos))
        if len(self._history) > 8:
            self._history.pop(0)

        messages = self._build_messages(state)

        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=20,
            )
            text = resp.choices[0].message.content.strip().lower()
        except Exception:
            return None

        return self._parse_direction(text, state["legal_actions"])

    # ── prompt 构造 ──────────────────────────────────────

    _SYSTEM_FULL = (
        "你在一个因果链迷宫中导航。迷宫中有若干对密钥和门，"
        "密钥N必须先拾取才能通过门N。通过门N后，才能继续找密钥N+1。"
        "你必须严格按 密钥0→门0→密钥1→门1→...→终点的顺序前进。"
        "规则：不要回头走已走过的路。总是向目标方向前进。"
        "每次只需回复一个方向词: up / right / down / left。不回复其他内容。"
    )

    _SYSTEM_CONCISE = (
        "迷宫导航。密钥N→门N，按序前进。不回头。只回复 up / right / down / left。"
    )

    def _build_messages(self, state: dict) -> list:
        """构建 messages 列表"""
        system = self.system_prompt
        if system is None:
            system = self._SYSTEM_CONCISE if self.concise else self._SYSTEM_FULL

        user = self._build_user_prompt(state)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _build_user_prompt(self, state: dict) -> str:
        """根据 causal state 构建用户提示词"""
        pos = state["pos"]
        goal = state["goal"]
        inventory = state.get("inventory", set())
        step = state.get("step", 0)
        chain_len = state.get("chain_length", 0)
        nearby = state.get("nearby_cells", {})
        legal = state.get("legal_actions", [])

        # 方向描述
        dir_map = {d: n for n, d in zip(DIR_NAMES, DIRS)}
        legal_desc = ", ".join(dir_map[d] for d in legal if d in dir_map)

        # 周围格子描述
        nearby_lines = []
        for d in DIRS:
            if d in nearby:
                cell = nearby[d]
                name = dir_map.get(d, "?")
                label = _cell_label(cell)
                nearby_lines.append(f"  {name}: {label}")
        nearby_str = "\n".join(nearby_lines)

        # 已持密钥
        held = []
        for k in sorted(inventory):
            if KEY_START <= k < KEY_START + MAX_KEYS:
                held.append(str(k - KEY_START))
        held_str = ", ".join(held) if held else "无"

        # 进度
        progress = f"已通过{step}/{chain_len}对"

        if self.concise:
            return (
                f"位置({pos[0]},{pos[1]}) → 终点({goal[0]},{goal[1]}) | {progress}\n"
                f"已持密钥: {held_str}\n"
                f"周围:\n{nearby_str}\n"
                f"可选方向: {legal_desc}"
            )
        else:
            # 全量提示 —— 把 chain 完整列出来
            chain_info = self._describe_chain(state)
            target_hint = self._build_target_hint(state)
            history_str = self._build_history()
            return (
                f"当前位置: ({pos[0]}, {pos[1]})\n"
                f"终点位置: ({goal[0]}, {goal[1]})\n"
                f"进度: 已完成 {step}/{chain_len} 对密钥-门\n"
                f"当前持有密钥: {held_str}\n"
                f"因果链顺序: {chain_info}\n"
                f"目标指引: {target_hint}\n"
                f"{history_str}"
                f"周围格子:\n{nearby_str}\n"
                f"当前可选移动方向: {legal_desc}\n"
                f"请选择下一步方向 (仅回复一个词)。"
            )

    def _describe_chain(self, state: dict) -> str:
        """描述完整因果链"""
        scene = state.get("scene")
        if scene is None:
            return "未知"
        parts = []
        for kid, did in scene.chain:
            k_idx = kid - KEY_START
            d_idx = did - DOOR_START
            kp = scene.key_positions.get(kid, ("?", "?"))
            dp = scene.door_positions.get(did, ("?", "?"))
            parts.append(f"密钥{k_idx}({kp})→门{d_idx}({dp})")
        return " → ".join(parts) + " → 终点"

    def _build_target_hint(self, state: dict) -> str:
        """生成当前子目标指引"""
        pos = state["pos"]
        step = state.get("step", 0)
        chain_len = state.get("chain_length", 0)
        inventory = state.get("inventory", set())
        scene = state.get("scene")

        if step >= chain_len or scene is None:
            return "前往终点"

        kid = KEY_START + step
        did = DOOR_START + step

        if kid in inventory:
            # 已有钥匙 → 目标是门
            dp = scene.door_positions.get(did, ("?", "?"))
            dx = dp[0] - pos[0]
            dy = dp[1] - pos[1]
            dir_hint = []
            if dx > 0: dir_hint.append("右")
            elif dx < 0: dir_hint.append("左")
            if dy > 0: dir_hint.append("下")
            elif dy < 0: dir_hint.append("上")
            return f"已持有密钥{step}，请前往门{step}({dp[0]},{dp[1]})，门在你的{'·'.join(dir_hint)}方"
        else:
            # 缺钥匙 → 目标是钥匙
            kp = scene.key_positions.get(kid, ("?", "?"))
            dx = kp[0] - pos[0]
            dy = kp[1] - pos[1]
            dir_hint = []
            if dx > 0: dir_hint.append("右")
            elif dx < 0: dir_hint.append("左")
            if dy > 0: dir_hint.append("下")
            elif dy < 0: dir_hint.append("上")
            return f"缺少密钥{step}，请前往密钥{step}({kp[0]},{kp[1]})，钥匙在你的{'·'.join(dir_hint)}方"

    def _build_history(self) -> str:
        """构建最近走过的位置历史"""
        if len(self._history) < 2:
            return ""
        # 去重连续相同位置，取最近几个
        unique = []
        for p in self._history:
            if not unique or p != unique[-1]:
                unique.append(p)
        if len(unique) < 2:
            return ""
        recent = unique[-6:]  # 最近6个不重复位置
        recent_str = " → ".join(f"({x},{y})" for x, y in recent)
        # 检测震荡
        if len(recent) >= 3:
            oscillation = f"\n警告：你在 ({recent[-1][0]},{recent[-1][1]}) 和 ({recent[-2][0]},{recent[-2][1]}) 之间反复震荡！请果断向目标方向前进，不要回头。" if (
                len(recent) >= 4 and recent[-1] == recent[-3] and recent[-2] == recent[-4]
            ) else ""
        else:
            oscillation = ""
        return f"最近路径: {recent_str}{oscillation}\n"

    # ── 解析 ────────────────────────────────────────────

    def _parse_direction(self, text: str, legal: list) -> Optional[Tuple[int, int]]:
        """从 LLM 回复中提取方向"""
        # 1. 方向词匹配
        for name, d in zip(DIR_NAMES, DIRS):
            if name in text and d in legal:
                return d

        # 2. 数字匹配 (0=up, 1=right, 2=down, 3=left)
        import re
        numbers = re.findall(r'\d+', text)
        for num_str in numbers:
            idx = int(num_str)
            if 0 <= idx < 4 and DIRS[idx] in legal:
                return DIRS[idx]

        # 3. dx,dy 格式
        for d in DIRS:
            if f"dx={d[0]}" in text or f"dy={d[1]}" in text:
                if d in legal:
                    return d

        return None

    @property
    def model_id(self):
        return self.model


# ═══════════════════════════════════════════════
# 便捷工厂
# ═══════════════════════════════════════════════

def make_gpt4o(api_key: str = None) -> CausalLLMAdapter:
    return CausalLLMAdapter(model="gpt-4o", api_key=api_key, temperature=0.0)

def make_gpt4o_mini(api_key: str = None) -> CausalLLMAdapter:
    return CausalLLMAdapter(model="gpt-4o-mini", api_key=api_key, temperature=0.0)

def make_deepseek(api_key: str = None) -> CausalLLMAdapter:
    return CausalLLMAdapter(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.0,
    )

def make_deepseek_reasoner(api_key: str = None) -> CausalLLMAdapter:
    return CausalLLMAdapter(
        model="deepseek-reasoner",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.0,
    )

def make_ollama(model: str = "llama3", base_url: str = "http://localhost:11434/v1") -> CausalLLMAdapter:
    return CausalLLMAdapter(model=model, base_url=base_url, api_key="ollama", temperature=0.0)


# ═══════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from causal_scene import generate_causal_scene
    from causal_eval import make_causal_state

    scene = generate_causal_scene(size=30, chain_length=3, wall_openings=1,
                                  dead_ends=2, alcove_keys=1, seed=99)

    print(f"场景: {scene.description}")
    print(f"因果链: {[(k-KEY_START, d-DOOR_START) for k,d in scene.chain]}")
    print(f"钥匙位置: {scene.key_positions}")
    print(f"门位置:   {scene.door_positions}")

    # 用简洁模式构造 state 并检查 prompt
    state = make_causal_state(scene.start, scene, set(), 0)
    adapter = CausalLLMAdapter(model="gpt-4o-mini", concise=True)
    prompt = adapter._build_user_prompt(state)

    print(f"\n── 简洁 prompt ──")
    print(prompt)

    # 全量模式
    adapter_full = CausalLLMAdapter(model="gpt-4o-mini", concise=False)
    prompt_full = adapter_full._build_user_prompt(state)
    print(f"\n── 全量 prompt ──")
    print(prompt_full)
