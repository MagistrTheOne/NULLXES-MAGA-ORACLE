"""Compact analysis_payload.json — the only object an LLM is allowed to see.

No raw worlds. No xlsx. Numbers come from Monte Carlo, never from the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from assumptions import DISCLAIMER, checkpoint_day, daily_burn
from briefing import p_breach_by_day
from sensitivity import tornado_table
from simulation import OracleResult


def _num(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def _pct(x: Any) -> float | None:
    v = _num(x)
    return v


def build_analysis_payload(
    cfg: Mapping,
    result: OracleResult,
    escape: Mapping | None = None,
    sensitivity_rows: list[dict] | None = None,
    policy: str = "HOLD",
    scenario: str | None = None,
) -> dict:
    escape = escape or {}
    if policy not in result.metrics:
        policy = next(iter(result.metrics))
    m = result.metrics[policy]
    hold = result.metrics.get("HOLD", m)
    chk = checkpoint_day(cfg)
    fb = result.cash[policy].first_breach_day
    burn = daily_burn(cfg)
    cash0 = float(cfg["company"]["cash_initial"])
    bridge0 = float(cfg["company"].get("bridge_capital", 0.0))

    torn = tornado_table(sensitivity_rows or [])
    sensitivity_top = [
        [r["parameter"], _num(r.get("delta_survival"))]
        for r in torn[:8]
        if r.get("parameter") is not None
    ]

    bridge = escape.get("min_bridge_for_target") or {}
    max_burn = escape.get("max_burn_for_target") or {}
    ranking = escape.get("policy_ranking") or [
        {
            "policy": name,
            "P(SURVIVAL)": mm["P(SURVIVAL)"],
            "P(CASH BREACH)": mm["P(CASH BREACH)"],
        }
        for name, mm in result.metrics.items()
    ]

    policies = {}
    for name, mm in result.metrics.items():
        policies[name] = {
            "survival_probability": _pct(mm["P(SURVIVAL)"]),
            "cash_breach_probability": _pct(mm["P(CASH BREACH)"]),
            "median_cash_d90": _num(mm["c90_median"]),
            "expected_shortfall_5": _num(mm["es5"]),
            "p_escape_class": _pct(mm.get("P(ESCAPE)")),
            "p_fail_class": _pct(mm.get("P(FAIL)")),
        }

    best = escape.get("best_policy_by_survival") or ranking[0]["policy"]
    best_s = policies.get(best, {}).get("survival_probability")
    hold_s = policies.get("HOLD", {}).get("survival_probability")
    policy_lift_pp = None
    if best_s is not None and hold_s is not None:
        policy_lift_pp = (best_s - hold_s) * 100.0

    live = list((result.extras or {}).get("live_events") or [])

    payload = {
        "disclaimer": DISCLAIMER,
        "engine": "NULLXES-MAGA-ORACLE Monte Carlo",
        "llm_must_not_invent_numbers": True,
        "scenario": scenario or (cfg.get("meta") or {}).get("name") or "baseline",
        "policy": policy,
        "simulation": {
            "worlds": int(m["n_worlds"]),
            "days": int(cfg["simulation"]["days"]),
            "seed": int(cfg["simulation"]["seed"]),
            "start_date": str(cfg["simulation"]["start_date"]),
            "checkpoint_date": str(cfg["bank"]["checkpoint_date"]),
            "checkpoint_day": int(chk),
            "runtime_s": _num(result.runtime_s),
            "gpu_used": False,
        },
        "known_inputs": {
            "cash_initial": cash0,
            "founder_external_income_daily": float(cfg["company"].get("founder_external_income_daily", 0.0)),
            "external_liquidity": float(cfg["company"].get("external_liquidity", 0.0)),
            "bank_funding_available_now": float(cfg["company"].get("bank_funding_available_now", 0.0)),
            "deal_target_capital_not_received": float(cfg["deal"]["target_capital"]),
        },
        "assumptions": {
            "tag": "ASSUMPTION / PLACEHOLDER — not empirical frequencies",
            "cash_minimum": float(cfg["company"]["cash_minimum"]),
            "burn_monthly": float(cfg["company"]["burn_monthly"]),
            "bridge_capital": bridge0,
            "discretionary_rd_share": float(cfg["company"]["discretionary_rd_share"]),
            "bank_approval_probability": float(cfg["bank"]["approval_probability"]),
            "deal_close_probability": float(cfg["deal"]["close_probability"]),
            "deal_failure_probability": float(cfg["deal"]["failure_probability"]),
            "contract_daily_inflow": float(cfg["contract"].get("daily_inflow", 0.0)),
            "black_swan_p_daily": float(cfg["hazards"]["BLACK_SWAN"]["p_daily"]),
        },
        "derived": {
            "daily_burn": _num(burn),
            "runway_days_if_no_inflows": _num((cash0 + bridge0) / burn) if burn > 1e-12 else None,
            "p_breach_before_checkpoint": _pct(p_breach_by_day(fb, max(chk - 1, 0))),
        },
        "metrics": {
            "survival_probability": _pct(m["P(SURVIVAL)"]),
            "cash_breach_probability": _pct(m["P(CASH BREACH)"]),
            "financing_before_breach": _pct(m.get("P(FINANCING BEFORE BREACH)")),
            "deal_close_probability_realized": _pct(m.get("P(DEAL CLOSE)")),
            "deal_fail_probability_realized": _pct(m.get("P(DEAL FAIL)")),
            "bank_approval_probability_realized": _pct(m.get("P(BANK APPROVAL)")),
            "bank_decline_probability_realized": _pct(m.get("P(BANK DECLINE)")),
            "bank_delay_probability_realized": _pct(m.get("P(BANK DELAY)")),
            "median_cash_d90": _num(m["c90_median"]),
            "p5_cash_d90": _num(m["c90_p5"]),
            "p95_cash_d90": _num(m["c90_p95"]),
            "expected_shortfall_5": _num(m["es5"]),
            "median_min_cash": _num(m.get("median_min_cash")),
            "p_fail_class": _pct(m.get("P(FAIL)")),
            "p_escape_class": _pct(m.get("P(ESCAPE)")),
            "p_trap_class": _pct(m.get("P(TRAP)")),
        },
        "policies": policies,
        "policy_comparison": {
            "best_policy_by_survival": best,
            "best_minus_HOLD_pp": _num(policy_lift_pp),
            "ranking": ranking,
        },
        "escape_solver": {
            "target_survival": _num(escape.get("target_survival")),
            "bridge_for_80pct_survival": _num(bridge.get("value")) if bridge.get("feasible") else None,
            "bridge_feasible": bridge.get("feasible"),
            "bridge_survival_at_value": _num(bridge.get("survival")),
            "max_burn_for_80pct_survival": _num(max_burn.get("value")) if max_burn.get("feasible") else None,
            "max_burn_feasible": max_burn.get("feasible"),
            "chosen_escape_set": (escape.get("minimal_escape_set") or {}).get("chosen"),
            "note": "Null fields mean solver did not run or target not reachable in search range.",
        },
        "sensitivity_top": sensitivity_top,
        "absurdity": {
            "median_max_ai": _num(result.ai_summary.get("median_max_ai")),
            "p95_max_ai": _num(result.ai_summary.get("p95_max_ai")),
            "max_ai": _num(result.ai_summary.get("p95_max_ai")),
            "label": result.ai_summary.get("median_max_band"),
        },
        "live_events_n": len(live),
    }
    return payload


def build_delta_payload(payload_a: Mapping, payload_b: Mapping) -> dict:
    """B minus A on shared metric keys. No invented drivers beyond sensitivity of B if present."""

    def g(p, *keys):
        cur: Any = p
        for k in keys:
            if not isinstance(cur, Mapping) or k not in cur:
                return None
            cur = cur[k]
        return _num(cur) if not isinstance(cur, Mapping) else cur

    surv_pp = None
    sa, sb = g(payload_a, "metrics", "survival_probability"), g(payload_b, "metrics", "survival_probability")
    if sa is not None and sb is not None:
        surv_pp = (sb - sa) * 100.0
    br_pp = None
    ba, bb = g(payload_a, "metrics", "cash_breach_probability"), g(payload_b, "metrics", "cash_breach_probability")
    if ba is not None and bb is not None:
        br_pp = (bb - ba) * 100.0

    top_b = list(payload_b.get("sensitivity_top") or [])
    main_driver = top_b[0][0] if top_b and top_b[0] else None

    return {
        "label_a": payload_a.get("scenario"),
        "label_b": payload_b.get("scenario"),
        "policy_a": payload_a.get("policy"),
        "policy_b": payload_b.get("policy"),
        "survival_delta_pp": _num(surv_pp),
        "cash_breach_delta_pp": _num(br_pp),
        "median_cash_delta": _num(
            None
            if g(payload_a, "metrics", "median_cash_d90") is None
            or g(payload_b, "metrics", "median_cash_d90") is None
            else g(payload_b, "metrics", "median_cash_d90") - g(payload_a, "metrics", "median_cash_d90")
        ),
        "es5_delta": _num(
            None
            if g(payload_a, "metrics", "expected_shortfall_5") is None
            or g(payload_b, "metrics", "expected_shortfall_5") is None
            else g(payload_b, "metrics", "expected_shortfall_5") - g(payload_a, "metrics", "expected_shortfall_5")
        ),
        "main_driver_from_b_sensitivity": main_driver,
        "main_driver_note": (
            "main_driver is the top |delta| sensitivity parameter of scenario B, "
            "not a causal claim about the universe."
            if main_driver
            else "данных недостаточно"
        ),
        "disclaimer": DISCLAIMER,
    }


def write_json(obj: Mapping, path: str | Path) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
