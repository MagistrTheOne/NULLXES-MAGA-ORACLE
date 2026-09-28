"""Rails: facts backend needs no transformers; Qwen cannot rewrite Monte Carlo numbers."""

from __future__ import annotations

import builtins
import json

from fidelity import check_numeric_fidelity
from qwen_analyst import generate_briefing


def _payload():
    return {
        "disclaimer": "test disclaimer",
        "scenario": "baseline",
        "policy": "HOLD",
        "known_inputs": {"cash_initial": 500.0},
        "assumptions": {
            "tag": "ASSUMPTION / PLACEHOLDER",
            "burn_monthly": 200000.0,
            "bank_approval_probability": 0.22,
            "deal_close_probability": 0.12,
            "bridge_capital": 0.0,
        },
        "derived": {"daily_burn": 6666.67, "runway_days_if_no_inflows": 0.075, "p_breach_before_checkpoint": 0.99},
        "metrics": {
            "survival_probability": 0.001,
            "cash_breach_probability": 0.999,
            "financing_before_breach": 0.0,
            "deal_close_probability_realized": 0.05,
            "bank_approval_probability_realized": 0.1,
            "median_cash_d90": -240000.0,
            "p5_cash_d90": -1800000.0,
            "expected_shortfall_5": -3200000.0,
            "p_fail_class": 0.999,
            "p_escape_class": 0.0,
            "p_trap_class": 0.0,
        },
        "policy_comparison": {"best_policy_by_survival": "CAPITAL_FIRST", "best_minus_HOLD_pp": 19.2},
        "escape_solver": {
            "bridge_for_80pct_survival": 4300000.0,
            "max_burn_for_80pct_survival": 370000.0,
        },
        "sensitivity_top": [["bridge_capital", 0.42]],
        "absurdity": {"median_max_ai": 96.4, "label": "ПОНЕДЕЛЬНИК"},
        "simulation": {"worlds": 1000, "seed": 1, "days": 90},
    }


def test_facts_backend_does_not_import_transformers(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers" or name.startswith("transformers."):
            raise AssertionError("facts backend must not import transformers")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)
    out = generate_briefing(_payload(), mode="board", backend="facts")
    assert out["used_llm"] is False
    assert out["backend"] == "facts"
    assert "P(SURVIVAL)" in out["text"]
    assert out["hf_token"] in {"present", "absent"}
    assert "hf_" not in out["text"]


def test_qwen_backend_alias_is_recognized():
    from qwen_analyst import _QWEN_BACKENDS

    assert "qwen" in _QWEN_BACKENDS


def test_fidelity_rejects_invented_survival():
    p = _payload()
    rail = check_numeric_fidelity("P(SURVIVAL): 0.73 the company is fine", p)
    assert rail["ok"] is False
    assert any("P(SURVIVAL)" in x for x in rail["issues"])


def test_fidelity_accepts_authoritative_survival():
    p = _payload()
    rail = check_numeric_fidelity("P(SURVIVAL): 0.001000 and P(CASH BREACH): 0.999000", p)
    assert rail["ok"] is True


def test_fidelity_accepts_percent_form():
    p = _payload()
    rail = check_numeric_fidelity("P(SURVIVAL): 0.1%", p)
    assert rail["ok"] is True


def test_generate_facts_rail_does_not_rewrite_payload_numbers():
    p = _payload()
    out = generate_briefing(p, backend="facts", mode="maga")
    assert "0.001" in out["text"] or "0.1%" in out["text"]
    assert p["metrics"]["survival_probability"] == 0.001
    assert out["fidelity"]["ok"] is True


def test_qwen_backend_cannot_modify_numeric_facts(monkeypatch):
    p = _payload()
    snap = json.dumps(p, sort_keys=True)

    def fake_llm(messages, model_id, max_new_tokens):
        assert "transformers" not in str(type(messages))
        return (
            "P(SURVIVAL): 0.73 we are saved. "
            "P(CASH BREACH): 0.01. "
            "BRIDGE@80%: 1"
        )

    monkeypatch.setattr("qwen_analyst._generate_transformers", fake_llm)
    out = generate_briefing(p, backend="qwen", mode="board")
    assert out["used_llm"] is True
    assert out["backend"] == "qwen"
    assert json.dumps(p, sort_keys=True) == snap
    assert p["metrics"]["survival_probability"] == 0.001
    assert out["fidelity"]["ok"] is False
    assert any("P(SURVIVAL)" in x for x in out["fidelity"]["issues"])
    assert "hf_" not in out["text"]


def test_qwen_backend_passes_when_it_copies_facts(monkeypatch):
    p = _payload()

    def fake_llm(messages, model_id, max_new_tokens):
        user = messages[1]["content"]
        assert "AUTHORITATIVE FACTS" in user
        assert "0.001" in user
        return (
            "BOARD\n"
            "P(SURVIVAL): 0.001000\n"
            "P(CASH BREACH): 0.999000\n"
            "BRIDGE@80%: 4300000\n"
            "ES5: -3200000\n"
            "C90_MEDIAN: -240000\n"
            "ASSUMPTION / PLACEHOLDER burn is tagged.\n"
        )

    monkeypatch.setattr("qwen_analyst._generate_transformers", fake_llm)
    out = generate_briefing(p, backend="qwen", mode="board")
    assert out["fidelity"]["ok"] is True
    assert p["metrics"]["survival_probability"] == 0.001


def test_briefing_never_echoes_session_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_this_is_a_fake_test_token_not_real")
    out = generate_briefing(_payload(), backend="facts")
    assert out["hf_token"] == "present"
    assert "hf_this_is_a_fake_test_token_not_real" not in out["text"]
    assert "hf_this_is_a_fake_test_token_not_real" not in json.dumps(out)
