#!/usr/bin/env python3
"""
OpenAI API 适配器
================
将 OpenAI 兼容 API 接入 Chern 熵判定引擎进行元认知评测。

使用前设置环境变量:
    set OPENAI_API_KEY=sk-xxx
    set OPENAI_BASE_URL=http://localhost:11434/v1  (可选, 默认 OpenAI)

使用:
    adapter = OpenAIAdapter(model="gpt-4o-mini")
    state = make_state((0,0), 10, grid, (9,9))
    action = adapter.predict(state)
"""

import os
import json
from typing import Optional, Tuple

from api import ModelInterface

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIR_NAMES = ["up", "right", "down", "left"]

# 系统提示: 教模型理解网格导航任务
SYSTEM_PROMPT = """你是一个网格导航代理。你会收到当前状态，需要选择一个方向移动。

状态格式:
{
  "pos": [x, y],         # 当前位置 (0-indexed)
  "size": 网格大小,
  "grid": 二维数组,       # 0=空地, 1=障碍
  "goal": [gx, gy],      # 目标位置
  "legal_actions": ["up", "right", "down", "left"]  # 当前可行的方向
}

规则:
1. 你只能选择 legal_actions 中的方向
2. 不能穿越障碍物
3. 目标是从起点到达终点
4. 如果你不确定该往哪走，回复 {"action": null}
5. 确定时回复 {"action": "right"}

只回复 JSON，不要其他文字。"""


class OpenAIAdapter(ModelInterface):
    """OpenAI API 适配器

    Args:
        model: 模型名称, 如 gpt-4o-mini, gpt-4, llama3 等
        base_url: API 地址, 默认 https://api.openai.com/v1
        api_key: API Key, 默认从 OPENAI_API_KEY 环境变量读取
        temperature: 采样温度, 默认 0 (确定性)
        silence_keywords: 模型输出中包含这些词时视为"不确定", 返回 None
    """

    def __init__(self, model: str = "gpt-4o-mini",
                 base_url: str = None,
                 api_key: str = None,
                 temperature: float = 0.0,
                 silence_keywords: tuple = ("null", "不确定", "unknown", "i don't know")):
        self._model = model
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._temperature = temperature
        self._silence_keywords = silence_keywords

        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
            self._available = True
        except ImportError:
            self._available = False
            print("[OpenAIAdapter] openai 包未安装。pip install openai")

    def predict(self, state: dict) -> Optional[Tuple[int, int]]:
        if not self._available:
            return self._local_fallback(state)

        # 构建合法方向描述
        legal_names = []
        for d in state["legal_actions"]:
            for i, dir_tuple in enumerate(DIRS):
                if d == dir_tuple:
                    legal_names.append(DIR_NAMES[i])
                    break

        # 构建消息
        user_msg = json.dumps({
            "pos": list(state["pos"]),
            "size": state["size"],
            "grid": state["grid"],
            "goal": list(state["goal"]),
            "legal_actions": legal_names,
        }, ensure_ascii=False)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self._temperature,
                max_tokens=50,
            )
            text = response.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"[OpenAIAdapter] API 调用失败: {e}")
            return None

        # 解析响应
        return self._parse_response(text, state["legal_actions"])

    def _parse_response(self, text: str, legal_actions: list) -> Optional[Tuple[int, int]]:
        """解析模型输出"""
        # 检查沉默关键词
        for kw in self._silence_keywords:
            if kw in text:
                return None

        # 尝试 JSON 解析
        try:
            data = json.loads(text)
            action_name = data.get("action")
        except json.JSONDecodeError:
            # 尝试从文本中提取方向词
            action_name = text.strip().strip('"').strip("'")

        if action_name is None or action_name == "null":
            return None

        # 映射方向名到元组
        name_to_dir = dict(zip(DIR_NAMES, DIRS))
        action = name_to_dir.get(action_name)
        if action is None:
            return None
        if action not in legal_actions:
            return None

        return action

    def _local_fallback(self, state: dict) -> Optional[Tuple[int, int]]:
        """本地离线回落: 纯启发式"""
        x, y = state["pos"]
        gx, gy = state["goal"]
        best_dir, best_h = None, float("inf")
        for d in state["legal_actions"]:
            nx, ny = x + d[0], y + d[1]
            h = abs(nx - gx) + abs(ny - gy)
            if h < best_h:
                best_h = h
                best_dir = d
        return best_dir

    @property
    def model_id(self):
        return self._model


class OllamaAdapter(OpenAIAdapter):
    """Ollama 适配器 (OpenAI 兼容 API)"""

    def __init__(self, model: str = "llama3", **kwargs):
        super().__init__(
            model=model,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            **kwargs,
        )


class LMStudioAdapter(OpenAIAdapter):
    """LM Studio 适配器 (OpenAI 兼容 API)"""

    def __init__(self, model: str = "local-model", **kwargs):
        super().__init__(
            model=model,
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            **kwargs,
        )


class DeepSeekAdapter(OpenAIAdapter):
    """DeepSeek 适配器 (OpenAI 兼容 API)"""

    def __init__(self, model: str = "deepseek-chat", **kwargs):
        super().__init__(
            model=model,
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            **kwargs,
        )


# ============================================================================
# 命令行自测
# ============================================================================

if __name__ == "__main__":
    from api import make_state

    state = make_state((0, 0), 5, [[0]*5 for _ in range(5)], (4, 4))
    print(f"state: pos={state['pos']}, legal={state['legal_actions']}")

    adapter = OpenAIAdapter(model="gpt-4o-mini")
    action = adapter.predict(state)
    print(f"predicted action: {action}")
