"""LLM presenter on top of AUTHORITATIVE FACTS + analysis_payload.

Monte Carlo is the only source of numbers. Qwen is a press secretary.
Default backend is 'facts' (no weight download).
`--analyst-backend qwen` is opt-in (Colab / transformers).
HF token is read from env or Colab secret only and is never printed.
"""

from __future__ import annotations

import json
from typing import Mapping

from assumptions import DISCLAIMER
from facts import facts_block, present_without_llm
from fidelity import attach_fidelity_footer, check_numeric_fidelity
from hf_session import huggingface_from_pretrained_kwargs, token_status

SYSTEM_SHARED = """Ты аналитический слой NULLXES MAGA ORACLE.

Ты НЕ изменяешь результаты Monte Carlo.
Ты НЕ пересчитываешь survival, ES, bridge, вероятности и ranking политик.
Ты НЕ придумываешь вероятности.
Ты НЕ выдаёшь ASSUMPTION / PLACEHOLDER за факты.
Ты НЕ прогнозируешь реальные политические, военные или экономические события.

Источник истины — блок AUTHORITATIVE FACTS. Копируй эти числа дословно.
Если поля нет или написано «данных недостаточно» — так и пиши.
Каждый ASSUMPTION, который упоминаешь, помечай словами ASSUMPTION / PLACEHOLDER.
"""

SYSTEM_BOARD = (
    SYSTEM_SHARED
    + """
Режим BOARD: сухой CFO/CEO, русский, без метафор.

Структура:
1. Главный результат (survival, cash breach, C90, ES5) — числа из FACTS.
2. Ключевые причины риска по FACTS (runway, checkpoint, deal/bank realized).
3. Sensitivity top — только из FACTS.
4. Escape solver или «данных недостаточно».
5. Политики только по данным симуляции.
6. Явно: known inputs / ASSUMPTION / simulation outputs.

В конце одна строка дисклеймера из JSON.disclaimer.
"""
)

SYSTEM_SWEEP = (
    SYSTEM_SHARED
    + """
Режим CASH SWEEP: человеческий тон, числа только из AUTHORITATIVE FACTS.

Ответь на 5 пунктов:
1. Первый Cash0 на сетке, где P(SURVIVAL) материально растёт (cliff survival≥5%).
2. Первый Cash0, где P(CASH BREACH) < 50%.
3. Первый Cash0, где обычно доходят до bank checkpoint без inflow.
4. Первый Cash0, где финансирование перестаёт быть единственным условием выживания.
5. Что арифметика runway, а что interaction со shocks и timing deal/bank.

P(DEAL CLOSE), P(BANK APPROVAL), AI не меняются с Cash0. Если в FACTS они одинаковые — так и скажи.
Не интерполируй между точками. Пропуск = «данных недостаточно».
CLIFF строки копируй дословно.
"""
)

SYSTEM_PIZDEC = (
    SYSTEM_SHARED
    + """
Режим PIZDEC HEATMAP: календарь 28.09–28.11. Числа только из AUTHORITATIVE FACTS.

Ответь:
1. HOTTEST_WEEK — копируй id, даты, pizdec_index, P(AI>=80).
2. По неделям: где выше P(EVENT) FIN / MACRO / GEO. Не усредняй соседние недели.
3. Checkpoint и CBR_MEETING только как даты из FACTS, без выдуманного решения ставки.
4. HOLD vs BEST. Если Cash0 в модели 500 — это арифметика кассы NULLXES, не ВВП РФ.
5. PUBLIC_INPUTS (NWF, резервы, KEY_RATE) не переводи в P(SURVIVAL).

Не интерполируй. Пропуск = «данных недостаточно».
"""
)

SYSTEM_FRONTIER = (
    SYSTEM_SHARED
    + """
Режим SURVIVAL FRONTIER: Cash0 × Burn. Числа только из AUTHORITATIVE FACTS.

Ответь:
1. При каких burn на этой сетке dead_zone (никакая политика не спасает).
2. Для каждого burn минимальный Cash0 с best P(SURVIVAL) ≥ 50% и ≥ 80%.
3. Где HOLD мёртв, а best policy ещё жива — копируй best_policy и best_minus_HOLD_pp.
4. BRIDGE@80% по клеткам; infeasible так и пиши.
5. Арифметика runway vs interaction.

Не интерполируй. Пропуск = «данных недостаточно».
"""
)

SYSTEM_MAGA = (
    SYSTEM_SHARED
    + """
Режим MAGA: человеческий тон, но каждое число — из AUTHORITATIVE FACTS.

Структура:
1. Жёсткая фраза результата + AI label из FACTS.
2. Где ломается касса (runway vs checkpoint vs deal) — по FACTS.
3. Sensitivity.
4. Escape solver, если числа есть.
5. Политики в п.п., без жизненных советов.
6. Known vs ASSUMPTION / PLACEHOLDER vs OUTPUT.

В конце дисклеймер из JSON.disclaimer.
"""
)

_FACTS_BACKENDS = {"facts", "none", "off", "rule"}
_QWEN_BACKENDS = {"qwen", "transformers", "hf", "local"}
_API_BACKENDS = {"openai", "openai_compat", "api"}


def build_messages(
    payload: Mapping,
    mode: str = "board",
    delta: Mapping | None = None,
) -> list[dict[str, str]]:
    if payload.get("experiment") == "cash_sweep":
        from cash_sweep import authoritative_sweep_block, sweep_view_from_payload

        system = SYSTEM_SWEEP
        user = authoritative_sweep_block(sweep_view_from_payload(payload))
    elif payload.get("experiment") == "frontier_sweep":
        from frontier_sweep import authoritative_frontier_block, frontier_view_from_payload

        system = SYSTEM_FRONTIER
        user = authoritative_frontier_block(frontier_view_from_payload(payload))
    elif payload.get("experiment") == "pizdec_heatmap":
        from pizdec_heatmap import authoritative_pizdec_block, pizdec_view_from_payload

        system = SYSTEM_PIZDEC
        user = authoritative_pizdec_block(pizdec_view_from_payload(payload))
    else:
        system = SYSTEM_MAGA if mode == "maga" else SYSTEM_BOARD
        user = facts_block(payload, delta)
    user += "\n\nJSON (context only; if it conflicts with AUTHORITATIVE FACTS, FACTS win)\n"
    user += json.dumps(payload, ensure_ascii=False, default=str)
    if delta:
        user += "\n\nDELTA_JSON\n" + json.dumps(delta, ensure_ascii=False, default=str)
    user += (
        "\n\nНапиши briefing. Не добавляй вероятности, которых нет в AUTHORITATIVE FACTS. "
        "Не округляй вероятность так, чтобы получилось другое число."
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
    mode = "maga" if mode == "maga" else "board"
    backend = (backend or "facts").lower()
    if payload.get("experiment") in {"cash_sweep", "frontier_sweep", "pizdec_heatmap"} and max_new_tokens <= 700:
        max_new_tokens = 1400
    if backend in _FACTS_BACKENDS:
        if payload.get("experiment") == "cash_sweep":
            from cash_sweep import sweep_presenter, sweep_view_from_payload

            text = sweep_presenter(sweep_view_from_payload(payload), mode=mode)
        elif payload.get("experiment") == "frontier_sweep":
            from frontier_sweep import frontier_presenter, frontier_view_from_payload

            text = frontier_presenter(frontier_view_from_payload(payload), mode=mode)
        elif payload.get("experiment") == "pizdec_heatmap":
            from pizdec_heatmap import pizdec_presenter, pizdec_view_from_payload

            text = pizdec_presenter(pizdec_view_from_payload(payload), mode=mode)
        else:
            text = present_without_llm(payload, mode=mode, delta=delta)
        rail = check_numeric_fidelity(text, payload, delta)
        text = attach_fidelity_footer(text, rail)
        return {
            "mode": mode,
            "backend": "facts",
            "model_id": None,
            "used_llm": False,
            "hf_token": token_status(),
            "fidelity": rail,
            "text": text,
            "disclaimer": DISCLAIMER,
        }
    messages = build_messages(payload, mode=mode, delta=delta)
    if backend in _QWEN_BACKENDS:
        text = _generate_transformers(messages, model_id, max_new_tokens)
        backend_name = "qwen"
    elif backend in _API_BACKENDS:
        text = _generate_openai_compat(messages, model_id, max_new_tokens)
        backend_name = "openai"
    else:
        raise ValueError(f"unknown analyst backend: {backend}")
    rail = check_numeric_fidelity(text, payload, delta)
    text = attach_fidelity_footer(text.strip(), rail)
    return {
        "mode": mode,
        "backend": backend_name,
        "model_id": model_id,
        "used_llm": True,
        "hf_token": token_status(),
        "fidelity": rail,
        "text": text,
        "disclaimer": DISCLAIMER,
    }


def _generate_transformers(messages: list[dict], model_id: str, max_new_tokens: int) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok_kwargs = huggingface_from_pretrained_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        **tok_kwargs,
    )
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        prompt = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
    }
    if tokenizer.pad_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True)


def _generate_openai_compat(messages: list[dict], model_id: str, max_new_tokens: int) -> str:
    """OpenAI-compatible endpoint. Token from env/Colab secret only."""
    import os
    from urllib.request import Request, urlopen

    from hf_session import session_hf_token

    base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("HF_OPENAI_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY") or session_hf_token()
    if not base or not key:
        raise RuntimeError(
            "openai backend needs OPENAI_BASE_URL and a session token "
            "(OPENAI_API_KEY or HF_TOKEN / Colab secret). Token is not logged."
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
