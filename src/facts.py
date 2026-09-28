"""Rule-based analyst. Emits FACTS only from payload. Invents nothing."""

from __future__ import annotations

from typing import Mapping


def _fmt_pct(x) -> str:
    if x is None:
        return "данных недостаточно"
    return f"{100.0 * float(x):.1f}%"


def _fmt_rub(x) -> str:
    if x is None:
        return "данных недостаточно"
    return f"{float(x):,.0f} RUB"


def _fmt_pp(x) -> str:
    if x is None:
        return "данных недостаточно"
    return f"{float(x):+.1f} п.п."


def extract_facts(payload: Mapping, delta: Mapping | None = None) -> list[str]:
    m = payload.get("metrics") or {}
    esc = payload.get("escape_solver") or {}
    pol = payload.get("policy_comparison") or {}
    ass = payload.get("assumptions") or {}
    known = payload.get("known_inputs") or {}
    der = payload.get("derived") or {}
    ai = payload.get("absurdity") or {}
    sim = payload.get("simulation") or {}
    top = payload.get("sensitivity_top") or []

    facts = [
        f"SCENARIO={payload.get('scenario')} POLICY={payload.get('policy')} "
        f"WORLDS={sim.get('worlds')} SEED={sim.get('seed')} DAYS={sim.get('days')}",
        f"KNOWN Cash0={_fmt_rub(known.get('cash_initial'))}; "
        f"founder income/day={_fmt_rub(known.get('founder_external_income_daily'))}; "
        f"bank funding now={_fmt_rub(known.get('bank_funding_available_now'))}; "
        f"deal target (NOT received)={_fmt_rub(known.get('deal_target_capital_not_received'))}",
        f"ASSUMPTION burn_monthly={_fmt_rub(ass.get('burn_monthly'))}; "
        f"bridge={_fmt_rub(ass.get('bridge_capital'))}; "
        f"P(bank approval prior)={_fmt_pct(ass.get('bank_approval_probability'))}; "
        f"P(deal close prior)={_fmt_pct(ass.get('deal_close_probability'))}",
        f"DERIVED daily_burn={_fmt_rub(der.get('daily_burn'))}; "
        f"runway_if_no_inflows={der.get('runway_days_if_no_inflows')}; "
        f"P(breach before checkpoint)={_fmt_pct(der.get('p_breach_before_checkpoint'))}",
        f"OUTPUT P(SURVIVAL)={_fmt_pct(m.get('survival_probability'))}",
        f"OUTPUT P(CASH BREACH)={_fmt_pct(m.get('cash_breach_probability'))}",
        f"OUTPUT P(FINANCING BEFORE BREACH)={_fmt_pct(m.get('financing_before_breach'))}",
        f"OUTPUT realized P(DEAL CLOSE)={_fmt_pct(m.get('deal_close_probability_realized'))}; "
        f"P(BANK APPROVAL)={_fmt_pct(m.get('bank_approval_probability_realized'))}",
        f"OUTPUT C90 median={_fmt_rub(m.get('median_cash_d90'))}; "
        f"P5={_fmt_rub(m.get('p5_cash_d90'))}; "
        f"ES5={_fmt_rub(m.get('expected_shortfall_5'))}",
        f"OUTPUT class P(FAIL)={_fmt_pct(m.get('p_fail_class'))}; "
        f"P(ESCAPE)={_fmt_pct(m.get('p_escape_class'))}; "
        f"P(TRAP)={_fmt_pct(m.get('p_trap_class'))}",
        f"POLICIES best_by_survival={pol.get('best_policy_by_survival')}; "
        f"best minus HOLD={_fmt_pp(pol.get('best_minus_HOLD_pp'))}",
        (
            f"ESCAPE bridge_for_80pct={_fmt_rub(esc.get('bridge_for_80pct_survival'))}"
            if esc.get("bridge_for_80pct_survival") is not None
            else "ESCAPE bridge_for_80pct=данных недостаточно"
        ),
        (
            f"ESCAPE max_burn_for_80pct={_fmt_rub(esc.get('max_burn_for_80pct_survival'))}"
            if esc.get("max_burn_for_80pct_survival") is not None
            else "ESCAPE max_burn_for_80pct=данных недостаточно"
        ),
        (
            "SENSITIVITY top="
            + ", ".join(f"{n} Δsurv={d:+.1%}" if d is not None else str(n) for n, d in top[:5])
            if top
            else "SENSITIVITY=данных недостаточно"
        ),
        f"ABSURDITY median-max AI={ai.get('median_max_ai')} label={ai.get('label')}",
        "CONSTRAINT LLM must not invent probabilities, must not treat assumptions as facts, "
        "must not forecast real geopolitical/military/economic events.",
    ]
    if delta:
        facts.append(
            f"DELTA vs {delta.get('label_a')}→{delta.get('label_b')}: "
            f"survival_pp={_fmt_pp(delta.get('survival_delta_pp'))}; "
            f"breach_pp={_fmt_pp(delta.get('cash_breach_delta_pp'))}; "
            f"median_cash_delta={_fmt_rub(delta.get('median_cash_delta'))}; "
            f"driver_from_B_sensitivity={delta.get('main_driver_from_b_sensitivity') or 'данных недостаточно'}"
        )
    return facts


def facts_block(payload: Mapping, delta: Mapping | None = None) -> str:
    return "FACTS\n" + "\n".join(f"- {line}" for line in extract_facts(payload, delta))


def present_without_llm(payload: Mapping, mode: str = "board", delta: Mapping | None = None) -> str:
    """Deterministic presenter. Used when Qwen is not loaded."""
    block = facts_block(payload, delta)
    disc = payload.get("disclaimer") or ""
    if mode == "maga":
        m = payload.get("metrics") or {}
        ai = payload.get("absurdity") or {}
        head = (
            f"MAGA MODE\n"
            f"Survival {_fmt_pct(m.get('survival_probability'))} | "
            f"Breach {_fmt_pct(m.get('cash_breach_probability'))} | "
            f"AI {ai.get('median_max_ai')} / {ai.get('label')}\n\n"
            "Ниже только факты симуляции. Это не прогноз мира.\n\n"
        )
        return head + block + "\n\n" + disc
    return (
        "BOARD MODE\n"
        "All figures below are Monte Carlo outputs or tagged assumptions.\n\n"
        + block
        + "\n\n"
        + disc
    )
