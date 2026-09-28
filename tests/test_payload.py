"""Payload and facts stay honest: no invented numbers."""

from __future__ import annotations

from assumptions import load_config
from facts import extract_facts, present_without_llm
from payload import build_analysis_payload, build_delta_payload
from qwen_analyst import generate_briefing
from simulation import run_oracle


def test_payload_uses_simulation_survival():
    cfg = load_config("baseline.yaml")
    r = run_oracle(cfg, n_worlds=24, seed=3, policies=["HOLD", "CAPITAL_FIRST"])
    p = build_analysis_payload(cfg, r, escape={}, sensitivity_rows=[], policy="HOLD")
    assert p["llm_must_not_invent_numbers"] is True
    assert abs(p["metrics"]["survival_probability"] - r.metrics["HOLD"]["P(SURVIVAL)"]) < 1e-12
    assert p["known_inputs"]["cash_initial"] == 500.0
    assert p["assumptions"]["tag"].startswith("ASSUMPTION")
    facts = extract_facts(p)
    assert any("OUTPUT P(SURVIVAL)" in f for f in facts)
    text = present_without_llm(p, mode="board")
    assert "BOARD MODE" in text
    out = generate_briefing(p, mode="maga", backend="facts")
    assert out["used_llm"] is False
    assert "MAGA MODE" in out["text"]


def test_delta_payload_sign():
    a = {
        "scenario": "baseline",
        "policy": "HOLD",
        "metrics": {
            "survival_probability": 0.40,
            "cash_breach_probability": 0.60,
            "median_cash_d90": -100.0,
            "expected_shortfall_5": -500.0,
        },
        "sensitivity_top": [["Burn", -0.2]],
    }
    b = {
        "scenario": "monday",
        "policy": "HOLD",
        "metrics": {
            "survival_probability": 0.10,
            "cash_breach_probability": 0.90,
            "median_cash_d90": -3100.0,
            "expected_shortfall_5": -8000.0,
        },
        "sensitivity_top": [["P(bank approval)", 0.18]],
    }
    d = build_delta_payload(a, b)
    assert abs(d["survival_delta_pp"] - (-30.0)) < 1e-9
    assert abs(d["cash_breach_delta_pp"] - 30.0) < 1e-9
    assert d["main_driver_from_b_sensitivity"] == "P(bank approval)"
