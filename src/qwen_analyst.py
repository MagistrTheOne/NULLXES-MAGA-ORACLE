"""LLM presenter on top of FACTS + analysis_payload.

The model never runs Monte Carlo and never invents probabilities.
Default backend is 'facts' (no weight download).
Transformers / OpenAI-compatible backends are opt-in for Colab.
"""

from __future__ import annotations

import json
from typing import Mapping

from assumptions import DISCLAIMER
from facts import facts_block, present_without_llm

SYSTEM_BOARD = """Ты аналитический слой NULLXES MAGA ORACLE (BOARD MODE).

Ты НЕ изменяешь результаты Monte Carlo.
Ты НЕ придумываешь вероятности.
Ты НЕ выдаёшь assumptions за факты.
Ты НЕ прогнозируешь реальные политические, военные или экономические события.

Тебе дан блок FACTS и JSON. Используй ТОЛЬКО числа из них.
Если поля нет — пиши «данных недостаточно».

Структура ответа:
1. Главный результат (survival, cash breach, C90, ES5).
2. Ключевые причины риска по FACTS (runway, checkpoint, realized deal/bank).
3. Какие параметры сильнее всего влияют на survival (sensitivity_top).
4. Escape solver (bridge / max burn) или «данных недостаточно».
5. Сравнение политик только по данным симуляции.
6. Явно раздели: known inputs / assumptions / simulation outputs.

Язык: сухой CFO/CEO, русский, без метафор.
В конце одна строка дисклеймера из JSON.disclaimer."""

SYSTEM_MAGA = """Ты аналитический слой NULLXES MAGA ORACLE (MAGA MODE).

Ты НЕ изменяешь результаты Monte Carlo.
Ты НЕ придумываешь вероятности.
Ты НЕ выдаёшь assumptions за факты.
Ты НЕ прогнозируешь реальные политические, военные или экономические события.

Тебе дан блок FACTS и JSON. Используй ТОЛЬКО числа из них.
Если поля нет — пиши «данных недостаточно».

Структура ответа:
1. Одна жёсткая фраза главного результата + AI label из JSON.
2. Где именно ломается касса (runway vs bank checkpoint vs deal fail) — по FACTS.
3. Sensitivity: что сильнее двигает P(SURVIVAL).
4. Escape solver: сколько моста/какой burn для целевого survival, если числа есть.
5. Политики: кто лучше на тех же мирах, в п.п., без советов «как жить».
6. Known vs ASSUMPTION vs OUTPUT — явно.

Можно быть человеческим, но каждое число — из FACTS/JSON.
В конце дисклеймер из JSON.disclaimer."""


def build_messages(
    payload: Mapping,
    mode: str = "board",
    delta: Mapping | None = None,
) -> list[dict[str, str]]:
    system = SYSTEM_MAGA if mode == "maga" else SYSTEM_BOARD
    user = (
        facts_block(payload, delta)
        + "\n\nJSON\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
    )
    if delta:
        user += "\n\nDELTA_JSON\n" + json.dumps(delta, ensure_ascii=False, default=str)
    user += (
        "\n\nНапиши briefing строго по FACTS. Не добавляй вероятности, которых нет в FACTS."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_briefing(
    payload: Mapping,
    mode: str = "board",
    delta: Mapping | None = None,
    backend: str = "facts",
    model_id: str = "Qwen/Qwen3-1.7B",
    max_new_tokens: int = 700,
) -> dict:
    """Return {mode, backend, model_id, text, used_llm}."""
    mode = "maga" if mode == "maga" else "board"
    backend = (backend or "facts").lower()
    if backend in {"facts", "none", "off", "rule"}:
        text = present_without_llm(payload, mode=mode, delta=delta)
        return {
            "mode": mode,
            "backend": "facts",
            "model_id": None,
            "used_llm": False,
            "text": text,
            "disclaimer": DISCLAIMER,
        }
    messages = build_messages(payload, mode=mode, delta=delta)
    if backend in {"transformers", "hf", "local"}:
        text = _generate_transformers(messages, model_id, max_new_tokens)
    elif backend in {"openai", "openai_compat", "api"}:
        text = _generate_openai_compat(messages, model_id, max_new_tokens)
    else:
        raise ValueError(f"unknown analyst backend: {backend}")
    return {
        "mode": mode,
        "backend": backend,
        "model_id": model_id,
        "used_llm": True,
        "text": text.strip(),
        "disclaimer": DISCLAIMER,
    }


def _generate_transformers(messages: list[dict], model_id: str, max_new_tokens: int) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        prompt = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True)


def _generate_openai_compat(messages: list[dict], model_id: str, max_new_tokens: int) -> str:
    """HF router / OpenAI-compatible endpoint. Requires env:

    OPENAI_BASE_URL  e.g. https://router.huggingface.co/v1
    OPENAI_API_KEY   or HF_TOKEN
    """
    import os
    from urllib.request import Request, urlopen

    base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("HF_OPENAI_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN")
    if not base or not key:
        raise RuntimeError(
            "openai backend needs OPENAI_BASE_URL and OPENAI_API_KEY or HF_TOKEN; "
            "weights are not downloaded locally."
        )
    body = json.dumps(
        {
            "model": model_id,
            "messages": messages,
            "temperature": 0,
            "max_tokens": int(max_new_tokens),
        }
    ).encode("utf-8")
    req = Request(
        base.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]
