#!/usr/bin/env python3
"""
OpenAI 兼容适配器 + DeepSeek自评扩展
=====================================
支持通过 OpenAI 兼容 API 接入任意在线模型。
DeepSeekAdapter 额外支持 metacognitive 模式：先自评置信度，低置信自动闭嘴。
"""

import os
import json
import random
from typing import Optional, Tuple
from api import ModelInterface

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIR_NAMES = ["up", "right", "down", "left"]


class OpenAIAdapter(ModelInterface):
    """通用 OpenAI 兼容适配器"""

    def __init__(self, model: str = "gpt-3.5-turbo", base_url: str = None,
                 api_key: str = None, temperature: float = 0.0):
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client

    def predict(self, state: dict) -> Optional[Tuple[int, int]]:
        prompt = self._build_prompt(state)
        client = self._get_client()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=50,
            )
            text = resp.choices[0].message.content.strip().lower()
        except Exception:
            return None

        # 解析方向
        for name, d in zip(DIR_NAMES, DIRS):
            if name in text:
                return d if d in state["legal_actions"] else None
        # 尝试解析数字
        try:
            idx = int(text.strip())
            if 0 <= idx < 4:
                return DIRS[idx] if DIRS[idx] in state["legal_actions"] else None
        except ValueError:
            pass
        return None

    def _build_prompt(self, state: dict) -> str:
        x, y = state["pos"]
        gx, gy = state["goal"]
        size = state["size"]
        la = state["legal_actions"]
        dir_desc = []
        for d in la:
            idx = DIRS.index(d)
            dir_desc.append(f"{idx}={DIR_NAMES[idx]}(dx={d[0]},dy={d[1]})")
        return (
            f"You are navigating a {size}x{size} grid. "
            f"Current position: ({x},{y}). Goal: ({gx},{gy}). "
            f"Legal moves: {', '.join(dir_desc)}. "
            f"Reply with exactly one direction number (0=up,1=right,2=down,3=left). "
            f"Only reply with a number."
        )

    @property
    def model_id(self):
        return self.model


class DeepSeekAdapter(OpenAIAdapter):
    """DeepSeek 适配器，支持元认知自评模式。

    metacognitive 策略:
      - "self_report": 先自评置信度（1-10），≤threshold 返回 None
      - "vote": temperature=0.7 采样3次，若3次不一致则返回 None（通过预测不稳定性度量不确定性）
      - None/False: 直接预测
    """

    def __init__(self, model: str = "deepseek-chat",
                 api_key: str = None,
                 metacognitive: str = "vote",
                 confidence_threshold: int = 3):
        super().__init__(
            model=model,
            base_url="https://api.deepseek.com/v1",
            api_key=api_key,
            temperature=0.0,
        )
        self.metacognitive = metacognitive
        self.confidence_threshold = confidence_threshold

    def predict(self, state: dict) -> Optional[Tuple[int, int]]:
        if not self.metacognitive or self.metacognitive is False:
            return super().predict(state)

        if self.metacognitive == "self_report":
            return self._predict_self_report(state)
        elif self.metacognitive == "vote":
            return self._predict_vote(state)
        return super().predict(state)

    def _predict_self_report(self, state: dict) -> Optional[Tuple[int, int]]:
        """自评模式：先问置信度"""
        client = self._get_client()
        assessment_prompt = self._build_assessment_prompt(state)
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": assessment_prompt}],
                temperature=0.0, max_tokens=10,
            )
            text = resp.choices[0].message.content.strip()
            confidence = int(text) if text.isdigit() else 5
        except Exception:
            confidence = 5
        if confidence <= self.confidence_threshold:
            return None
        return super().predict(state)

    def _predict_vote(self, state: dict) -> Optional[Tuple[int, int]]:
        """投票模式：temperature=0.7 采样3次，不一致→闭嘴"""
        client = self._get_client()
        prompt = self._build_prompt(state)
        predictions = []
        for _ in range(3):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7, max_tokens=50,
                )
                text = resp.choices[0].message.content.strip().lower()
            except Exception:
                predictions.append(None)
                continue

            found = None
            for name, d in zip(DIR_NAMES, DIRS):
                if name in text:
                    found = d if d in state["legal_actions"] else None
                    break
            if found is None:
                try:
                    idx = int(text)
                    if 0 <= idx < 4:
                        found = DIRS[idx] if DIRS[idx] in state["legal_actions"] else None
                except ValueError:
                    pass
            predictions.append(found)

        # 检查一致性
        valid = [p for p in predictions if p is not None]
        if len(valid) < 2:
            return None
        # 所有有效预测必须一致
        if all(p == valid[0] for p in valid):
            return valid[0]
        return None  # 意见不一 → 闭嘴

    def _build_assessment_prompt(self, state: dict) -> str:
        x, y = state["pos"]
        gx, gy = state["goal"]
        size = state["size"]
        la = state["legal_actions"]
        n_legal = len(la)
        return (
            f"Grid {size}x{size}. You are at ({x},{y}), goal at ({gx},{gy}). "
            f"There are {n_legal} legal moves available. "
            f"Rate your confidence in picking the optimal direction (1-10, 1=completely unsure, 10=certain). "
            f"Reply with ONLY a single number 1-10."
        )

    @property
    def model_id(self):
        suffix = "_meta" if self.metacognitive else ""
        return f"deepseek{suffix}"


class OllamaAdapter(OpenAIAdapter):
    """Ollama 本地模型适配器"""

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434/v1"):
        super().__init__(model=model, base_url=base_url, api_key="ollama")


class LMStudioAdapter(OpenAIAdapter):
    """LM Studio 本地模型适配器"""

    def __init__(self, model: str = "local-model", base_url: str = "http://localhost:1234/v1"):
        super().__init__(model=model, base_url=base_url, api_key="lm-studio")
