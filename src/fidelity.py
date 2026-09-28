"""Immutable AUTHORITATIVE FACTS + post-hoc numeric fidelity rail."""

from __future__ import annotations

import re
from typing import Any, Mapping


def _prob(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def _rub(x: Any) -> float | None:
    return _prob(x)


def authoritative_map(payload: Mapping, delta: Mapping | None = None) -> dict[str, float | None]:
    m = payload.get("metrics") or {}
    esc = payload.get("escape_solver") or {}
    der = payload.get("derived") or {}
    ass = payload.get("assumptions") or {}
    known = payload.get("known_inputs") or {}
    ai = payload.get("absurdity") or {}
    pol = payload.get("policy_comparison") or {}
    out = {
        "P(SURVIVAL)": _prob(m.get("survival_probability")),
        "P(CASH BREACH)": _prob(m.get("cash_breach_probability")),
        "P(FINANCING BEFORE BREACH)": _prob(m.get("financing_before_breach")),
        "P(DEAL CLOSE REALIZED)": _prob(m.get("deal_close_probability_realized")),
        "P(BANK APPROVAL REALIZED)": _prob(m.get("bank_approval_probability_realized")),
        "C90_MEDIAN": _rub(m.get("median_cash_d90")),
        "C90_P5": _rub(m.get("p5_cash_d90")),
        "ES5": _rub(m.get("expected_shortfall_5")),
        "BRIDGE@80%": _rub(esc.get("bridge_for_80pct_survival")),
        "MAX_BURN@80%": _rub(esc.get("max_burn_for_80pct_survival")),
        "CASH0": _rub(known.get("cash_initial")),
        "BURN_MONTHLY": _rub(ass.get("burn_monthly")),
        "P(BANK APPROVAL ASSUMPTION)": _prob(ass.get("bank_approval_probability")),
        "P(DEAL CLOSE ASSUMPTION)": _prob(ass.get("deal_close_probability")),
        "RUNWAY_DAYS": _prob(der.get("runway_days_if_no_inflows")),
        "P(BREACH BEFORE CHECKPOINT)": _prob(der.get("p_breach_before_checkpoint")),
        "BEST_MINUS_HOLD_PP": _prob(pol.get("best_minus_HOLD_pp")),
        "AI_MEDIAN_MAX": _prob(ai.get("median_max_ai")),
    }
    if delta:
        out["SURVIVAL_DELTA_PP"] = _prob(delta.get("survival_delta_pp"))
        out["BREACH_DELTA_PP"] = _prob(delta.get("cash_breach_delta_pp"))
    return out


def _fmt_auth_value(key: str, v: float | None) -> str:
    if v is None:
        return "данных недостаточно"
    if key.endswith("_PP") or key.endswith("PP"):
        return f"{v:.4f}"
    if key in {"RUNWAY_DAYS", "AI_MEDIAN_MAX"}:
        return f"{v:.6g}"
    if key.startswith("P(") or "ASSUMPTION" in key:
        return f"{v:.6f}"
    return f"{v:.0f}"


def authoritative_facts_block(payload: Mapping, delta: Mapping | None = None) -> str:
    rows = []
    amap = authoritative_map(payload, delta)
    for k, v in amap.items():
        rows.append(f"{k}: {_fmt_auth_value(k, v)}")
    ass_tag = (payload.get("assumptions") or {}).get("tag") or "ASSUMPTION / PLACEHOLDER"
    lines = [
        "AUTHORITATIVE FACTS",
        "These numbers are immutable. Copy them. Do not recompute. Do not round into a new probability.",
        f"ASSUMPTION_TAG: {ass_tag}",
        *rows,
        "END AUTHORITATIVE FACTS",
    ]
    return "\n".join(lines)


def _parse_quoted(raw: str) -> float | None:
    s = raw.strip().replace(" ", "").replace(",", "")
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    s = s.replace("\xa0", "")
    try:
        v = float(s)
    except ValueError:
        return None
    if pct:
        return v / 100.0
    return v


def _prob_close(got: float, expected: float, tol: float = 5e-4) -> bool:
    if abs(got - expected) <= tol:
        return True
    if abs(got / 100.0 - expected) <= tol:
        return True
    if abs(got - expected * 100.0) <= 0.05:
        return True
    return False


def _rub_close(got: float, expected: float, rel: float = 0.005, abs_tol: float = 1.0) -> bool:
    if abs(got - expected) <= abs_tol:
        return True
    scale = max(abs(expected), 1.0)
    return abs(got - expected) / scale <= rel


_LABEL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("P(SURVIVAL)", r"P\(\s*SURVIVAL\s*\)\s*[:=]\s*([+-]?\d[\d\s,.]*%?)"),
    ("P(CASH BREACH)", r"P\(\s*CASH\s*BREACH\s*\)\s*[:=]\s*([+-]?\d[\d\s,.]*%?)"),
    ("P(FINANCING BEFORE BREACH)", r"P\(\s*FINANCING BEFORE BREACH\s*\)\s*[:=]\s*([+-]?\d[\d\s,.]*%?)"),
    ("BRIDGE@80%", r"BRIDGE@80%\s*[:=]\s*([+-]?\d[\d\s,.]*)"),
    ("ES5", r"\bES5\b\s*[:=]\s*([+-]?\d[\d\s,.]*)"),
    ("C90_MEDIAN", r"C90(?:_MEDIAN|\s*median)\s*[:=]\s*([+-]?\d[\d\s,.]*)"),
)


def check_numeric_fidelity(text: str, payload: Mapping, delta: Mapping | None = None) -> dict:
    """If the briefing quotes a labeled metric, it must match Monte Carlo."""
    if payload.get("experiment") == "cash_sweep":
        return _check_sweep_fidelity(text, payload)
    expected = authoritative_map(payload, delta)
    issues: list[str] = []
    for label, pat in _LABEL_PATTERNS:
        exp = expected.get(label)
        if exp is None:
            continue
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            got = _parse_quoted(m.group(1))
            if got is None:
                continue
            is_prob = label.startswith("P(")
            ok = _prob_close(got, exp) if is_prob else _rub_close(got, exp)
            if not ok:
                issues.append(
                    f"{label}: quoted {m.group(1).strip()} != authoritative {exp}"
                )
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "checked_labels": [a[0] for a in _LABEL_PATTERNS],
    }


def _check_sweep_fidelity(text: str, payload: Mapping) -> dict:
    rows = ((payload.get("cash_sweep") or {}).get("rows")) or []
    issues: list[str] = []
    checked: list[str] = []
    for row in rows:
        name = row.get("name") or ""
        exp = row.get("P(SURVIVAL)")
        if exp is None:
            continue
        label = f"{name} P(SURVIVAL)"
        checked.append(label)
        pat = rf"{re.escape(name)}[\s\S]{{0,80}}P\(\s*SURVIVAL\s*\)\s*[:=]\s*([+-]?\d[\d\s,.]*%?)"
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            got = _parse_quoted(m.group(1))
            if got is None:
                continue
            if not _prob_close(got, float(exp)):
                issues.append(f"{label}: quoted {m.group(1).strip()} != authoritative {exp}")
        exp_b = row.get("P(CASH BREACH)")
        if exp_b is None:
            continue
        checked.append(f"{name} P(CASH BREACH)")
        pat_b = rf"{re.escape(name)}[\s\S]{{0,220}}P\(\s*CASH\s*BREACH\s*\)\s*[:=]\s*([+-]?\d[\d\s,.]*%?)"
        for m in re.finditer(pat_b, text, flags=re.IGNORECASE):
            got = _parse_quoted(m.group(1))
            if got is None:
                continue
            if not _prob_close(got, float(exp_b)):
                issues.append(
                    f"{name} P(CASH BREACH): quoted {m.group(1).strip()} != {exp_b}"
                )
    return {"ok": len(issues) == 0, "issues": issues, "checked_labels": checked}


def attach_fidelity_footer(text: str, rail: Mapping) -> str:
    if rail.get("ok"):
        note = "\n\n[FIDELITY RAIL] quoted labeled metrics match AUTHORITATIVE FACTS."
    else:
        issues = "; ".join(rail.get("issues") or [])
        note = (
            "\n\n[FIDELITY RAIL FAIL] Qwen quoted a number that is not in AUTHORITATIVE FACTS. "
            "Trust Monte Carlo / FACTS, not the mismatched quote. "
            f"Issues: {issues}"
        )
    return text.rstrip() + note
