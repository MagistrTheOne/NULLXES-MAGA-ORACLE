"""Cash0 geometry sweep on common random numbers.

Only company.cash_initial changes. Hazards, deal, and bank draws are shared.
Monte Carlo remains the only source of numbers. Qwen only presents the table.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from assumptions import (
    CLASS_NAMES,
    DISCLAIMER,
    checkpoint_day,
    daily_burn,
    set_path,
)
from briefing import p_breach_by_day
from escape_solver import policy_ranking
from payload import write_json
from simulation import OracleResult, make_streams, run_oracle, simulate_bundle

CASH_SCENARIOS: tuple[tuple[str, float], ...] = (
    ("CASH_500", 500.0),
    ("CASH_5K", 5_000.0),
    ("CASH_50K", 50_000.0),
    ("CASH_200K", 200_000.0),
    ("CASH_599K", 599_000.0),
    ("CASH_1M", 1_000_000.0),
    ("CASH_2M", 2_000_000.0),
    ("CASH_5M", 5_000_000.0),
    ("CASH_10M", 10_000_000.0),
    ("CASH_30M", 30_000_000.0),
)

_SURVIVAL_MATERIAL = 0.05
_BREACH_HALF = 0.50
_FINANCING_OPTIONAL = 0.50


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def _class_dist(m: Mapping) -> dict[str, float]:
    return {name: float(m.get(f"P({name})", 0.0)) for name in CLASS_NAMES.values()}


def _no_material_mask(result: OracleResult) -> np.ndarray:
    material = float(result.cfg["outcomes"]["material_financing_rub"])
    total = result.bundle.deal.inflows.sum(axis=1) + result.bundle.bank.inflows.sum(axis=1)
    return total < material


def scenario_row(name: str, cash0: float, result: OracleResult, policy: str = "HOLD") -> dict:
    cfg = result.cfg
    m = result.metrics[policy]
    fb = result.cash[policy].first_breach_day
    chk = checkpoint_day(cfg)
    burn = daily_burn(cfg)
    ttfb = m.get("ttfb") or {}
    ranking = policy_ranking(result)
    best = ranking[0]
    hold_s = float(result.metrics["HOLD"]["P(SURVIVAL)"])
    best_s = float(best["P(SURVIVAL)"])
    no_mat = _no_material_mask(result)
    survived = fb < 0
    if int(no_mat.sum()) == 0:
        p_surv_no_mat = None
    else:
        p_surv_no_mat = float(survived[no_mat].mean())
    med_breach = ttfb.get("median_day_if_breach")
    return {
        "name": name,
        "cash_initial": float(cash0),
        "arithmetic_runway_days": _f(cash0 / burn) if burn > 1e-12 else None,
        "P(SURVIVAL)": float(m["P(SURVIVAL)"]),
        "P(CASH BREACH)": float(m["P(CASH BREACH)"]),
        "P(FINANCING BEFORE BREACH)": float(m["P(FINANCING BEFORE BREACH)"]),
        "median_first_breach_day_if_breach": _f(med_breach),
        "p_never_breach": float((fb < 0).mean()),
        "p_breach_before_checkpoint": p_breach_by_day(fb, max(chk - 1, 0)),
        "C90_median": float(m["c90_median"]),
        "ES5": float(m["es5"]),
        "P(DEAL CLOSE)": float(m["P(DEAL CLOSE)"]),
        "P(BANK APPROVAL)": float(m["P(BANK APPROVAL)"]),
        "best_policy_by_survival": best["policy"],
        "best_minus_HOLD_pp": (best_s - hold_s) * 100.0,
        "absurdity_median_max": _f(result.ai_summary.get("median_max_ai")),
        "absurdity_label": result.ai_summary.get("median_max_band"),
        "p_survival_given_no_material_financing": p_surv_no_mat,
        "class_distribution": _class_dist(m),
        "policy_view": policy,
    }


def identify_cliffs(rows: list[dict], checkpoint: int, daily: float) -> dict:
    """Python, not the LLM, names the geometry breaks."""

    def first(pred) -> dict | None:
        for row in rows:
            if pred(row):
                return {"name": row["name"], "cash_initial": row["cash_initial"]}
        return None

    arith_chk = daily * float(checkpoint) if daily > 0 else None
    arith_90 = daily * 90.0 if daily > 0 else None
    return {
        "survival_material_ge_5pct": first(lambda r: r["P(SURVIVAL)"] >= _SURVIVAL_MATERIAL),
        "breach_probability_below_50pct": first(lambda r: r["P(CASH BREACH)"] < _BREACH_HALF),
        "p_breach_before_checkpoint_below_50pct": first(
            lambda r: r["p_breach_before_checkpoint"] < _BREACH_HALF
        ),
        "arithmetic_runway_covers_checkpoint": first(
            lambda r: (r.get("arithmetic_runway_days") or 0) + 1e-9 >= checkpoint
        ),
        "arithmetic_runway_covers_90d": first(
            lambda r: (r.get("arithmetic_runway_days") or 0) + 1e-9 >= 90.0
        ),
        "financing_optional_surv_no_material_ge_50pct": first(
            lambda r: (r.get("p_survival_given_no_material_financing") or 0.0)
            >= _FINANCING_OPTIONAL
        ),
        "arithmetic_cash_for_checkpoint_rub": _f(arith_chk),
        "arithmetic_cash_for_90d_burn_rub": _f(arith_90),
        "rule": (
            "Cliffs are the first Cash0 in this grid that meets the threshold. "
            "They are not interpolated. Missing cliff = threshold not reached on this grid."
        ),
    }


def run_cash_sweep(
    cfg: Mapping,
    n_worlds: int | None = None,
    seed: int | None = None,
    scenarios: tuple[tuple[str, float], ...] = CASH_SCENARIOS,
    policy_view: str = "HOLD",
    overlay=None,
    live_events: list | None = None,
) -> dict:
    n = int(n_worlds if n_worlds is not None else cfg["simulation"]["worlds"])
    seed_i = int(seed if seed is not None else cfg["simulation"]["seed"])
    t_days = int(cfg["simulation"]["days"])
    streams = make_streams(n, t_days, seed_i)
    bundle = simulate_bundle(cfg, streams, overlay=overlay)
    rows = []
    last: OracleResult | None = None
    for name, cash0 in scenarios:
        cfg_i = set_path(deepcopy(dict(cfg)), "company.cash_initial", float(cash0))
        result = run_oracle(
            cfg_i,
            n_worlds=n,
            seed=seed_i,
            streams=streams,
            bundle=bundle,
            overlay=overlay,
            live_events=live_events,
        )
        last = result
        rows.append(scenario_row(name, cash0, result, policy=policy_view))

    assert last is not None
    chk = checkpoint_day(cfg)
    burn = daily_burn(cfg)
    cliffs = identify_cliffs(rows, chk, burn)
    invariant_deal = {r["P(DEAL CLOSE)"] for r in rows}
    invariant_bank = {r["P(BANK APPROVAL)"] for r in rows}
    invariant_ai = {r["absurdity_median_max"] for r in rows}
    return {
        "disclaimer": DISCLAIMER,
        "experiment": "cash_sweep",
        "llm_must_not_invent_numbers": True,
        "only_changed": "company.cash_initial",
        "priors_frozen": True,
        "common_random_numbers": True,
        "seed": seed_i,
        "worlds": n,
        "days": t_days,
        "policy_view": policy_view,
        "checkpoint_day": chk,
        "burn_monthly_assumption": float(cfg["company"]["burn_monthly"]),
        "daily_burn_assumption": burn,
        "cash_minimum": float(cfg["company"]["cash_minimum"]),
        "crn_invariants": {
            "deal_close_identical_across_cash": len(invariant_deal) == 1,
            "bank_approval_identical_across_cash": len(invariant_bank) == 1,
            "absurdity_identical_across_cash": len(invariant_ai) == 1,
            "note": (
                "Deal/bank/AI live on the event layer. They must not move with Cash0. "
                "P(FINANCING BEFORE BREACH) MAY move because breach timing depends on cash."
            ),
        },
        "rows": rows,
        "cliffs": cliffs,
        "assumptions_tag": "ASSUMPTION / PLACEHOLDER — burn, deal/bank priors unchanged",
    }


def write_cash_sweep_charts(sweep: Mapping, chart_dir: str | Path) -> dict[str, Path]:
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)
    rows = list(sweep["rows"])
    x = np.array([r["cash_initial"] for r in rows], dtype=np.float64)
    surv = np.array([r["P(SURVIVAL)"] for r in rows], dtype=np.float64)
    br = np.array([r["P(CASH BREACH)"] for r in rows], dtype=np.float64)
    chk_cash = sweep["cliffs"].get("arithmetic_cash_for_checkpoint_rub")
    d90_cash = sweep["cliffs"].get("arithmetic_cash_for_90d_burn_rub")

    def _ax_cash(ax, y, ylabel, title, color):
        ax.plot(x, y, marker="o", color=color, lw=2.2)
        ax.set_xscale("log")
        ax.set_xlabel("Cash0, RUB (log)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, which="both", alpha=0.3)
        if chk_cash:
            ax.axvline(chk_cash, color="#8B1E1E", ls=":", lw=1.3, label="arithmetic Cash0 to checkpoint")
        if d90_cash:
            ax.axvline(d90_cash, color="#C4A35A", ls="--", lw=1.3, label="arithmetic 90-day burn")
        ax.legend(fontsize=8, loc="best")

    paths = {}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    _ax_cash(ax, surv, "P(SURVIVAL)", "Cash0 → P(SURVIVAL)  |  same worlds, only cash_initial", "#1F6B4A")
    p = chart_dir / "cash0_survival.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["cash0_survival"] = p

    fig, ax = plt.subplots(figsize=(10, 4.8))
    _ax_cash(ax, br, "P(CASH BREACH)", "Cash0 → P(CASH BREACH)  |  cash cliff", "#8B1E1E")
    p = chart_dir / "cash0_breach.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["cash0_breach"] = p
    return paths


def authoritative_sweep_block(sweep: Mapping) -> str:
    lines = [
        "AUTHORITATIVE FACTS",
        "Cash0 sweep. Same seed, same event/deal/bank worlds. Only cash_initial changes.",
        "Do not recompute. Do not interpolate between grid points.",
        f"SEED: {sweep['seed']} WORLDS: {sweep['worlds']} DAYS: {sweep['days']} "
        f"POLICY_VIEW: {sweep['policy_view']}",
        f"ASSUMPTION burn_monthly={sweep['burn_monthly_assumption']:.0f} "
        f"daily_burn={sweep['daily_burn_assumption']:.6g} "
        f"checkpoint_day={sweep['checkpoint_day']}",
        f"CRN deal_close_identical={sweep['crn_invariants']['deal_close_identical_across_cash']} "
        f"bank_approval_identical={sweep['crn_invariants']['bank_approval_identical_across_cash']} "
        f"absurdity_identical={sweep['crn_invariants']['absurdity_identical_across_cash']}",
    ]
    for r in sweep["rows"]:
        med = r["median_first_breach_day_if_breach"]
        med_s = f"{med:.4g}" if med is not None else "данных недостаточно"
        pnm = r["p_survival_given_no_material_financing"]
        pnm_s = f"{pnm:.6f}" if pnm is not None else "данных недостаточно"
        cd = r["class_distribution"]
        lines.append(
            f"{r['name']} Cash0: {r['cash_initial']:.0f} "
            f"runway_days: {r['arithmetic_runway_days']:.6g} "
            f"P(SURVIVAL): {r['P(SURVIVAL)']:.6f} "
            f"P(CASH BREACH): {r['P(CASH BREACH)']:.6f} "
            f"P(FINANCING BEFORE BREACH): {r['P(FINANCING BEFORE BREACH)']:.6f} "
            f"median_first_breach_if_breach: {med_s} "
            f"p_never_breach: {r['p_never_breach']:.6f} "
            f"p_breach_before_checkpoint: {r['p_breach_before_checkpoint']:.6f} "
            f"C90_MEDIAN: {r['C90_median']:.0f} "
            f"ES5: {r['ES5']:.0f} "
            f"P(DEAL CLOSE): {r['P(DEAL CLOSE)']:.6f} "
            f"P(BANK APPROVAL): {r['P(BANK APPROVAL)']:.6f} "
            f"best_policy: {r['best_policy_by_survival']} "
            f"best_minus_HOLD_pp: {r['best_minus_HOLD_pp']:.4f} "
            f"AI_MEDIAN_MAX: {r['absurdity_median_max']:.6g} "
            f"p_surv_no_material: {pnm_s} "
            f"class FAIL: {cd.get('FAIL', 0):.6f} ESCAPE: {cd.get('ESCAPE', 0):.6f} "
            f"TRAP: {cd.get('TRAP', 0):.6f} SURVIVE: {cd.get('SURVIVE', 0):.6f} "
            f"FREEZE: {cd.get('FREEZE', 0):.6f} RU_EXIT: {cd.get('RU_EXIT', 0):.6f}"
        )
    cliffs = sweep["cliffs"]
    for key in (
        "survival_material_ge_5pct",
        "breach_probability_below_50pct",
        "p_breach_before_checkpoint_below_50pct",
        "arithmetic_runway_covers_checkpoint",
        "arithmetic_runway_covers_90d",
        "financing_optional_surv_no_material_ge_50pct",
    ):
        hit = cliffs.get(key)
        if hit:
            lines.append(f"CLIFF {key}: {hit['name']} Cash0: {hit['cash_initial']:.0f}")
        else:
            lines.append(f"CLIFF {key}: данных недостаточно")
    lines.append(
        f"CLIFF arithmetic_cash_for_checkpoint_rub: {cliffs.get('arithmetic_cash_for_checkpoint_rub')}"
    )
    lines.append(
        f"CLIFF arithmetic_cash_for_90d_burn_rub: {cliffs.get('arithmetic_cash_for_90d_burn_rub')}"
    )
    lines.append("END AUTHORITATIVE FACTS")
    return "\n".join(lines)


def sweep_presenter(sweep: Mapping, mode: str = "maga") -> str:
    block = authoritative_sweep_block(sweep)
    cliffs = sweep["cliffs"]

    def _nm(key: str) -> str:
        hit = cliffs.get(key)
        return hit["name"] if hit else "данных недостаточно"

    task = (
        "Сравнительная задача (только по FACTS):\n"
        "1. Первый Cash0, где survival материально растёт (порог 5 п.п. на этой сетке).\n"
        "2. Первый Cash0, где P(CASH BREACH) < 50%.\n"
        "3. Первый Cash0, где обычно доходят до bank checkpoint без inflow "
        "(смотри p_breach_before_checkpoint и arithmetic runway).\n"
        "4. Первый Cash0, где внешнее финансирование перестаёт быть единственным "
        "условием выживания (p_surv_no_material ≥ 50%).\n"
        "5. Что чисто арифметика runway, а что interaction со shocks и timing deal/bank.\n"
        "P(DEAL CLOSE), P(BANK APPROVAL), AI не должны меняться по Cash0 — это CRN.\n"
        "Пропуски = «данных недостаточно». Не интерполируй между точками сетки.\n"
    )
    if mode == "maga":
        head = (
            "MAGA MODE — CASH GEOMETRY SWEEP\n"
            f"Cliff survival≥5%: {_nm('survival_material_ge_5pct')} | "
            f"breach<50%: {_nm('breach_probability_below_50pct')} | "
            f"financing optional: {_nm('financing_optional_surv_no_material_ge_50pct')}\n\n"
            "Ниже только факты симуляции. Это не прогноз мира.\n\n"
        )
    else:
        head = "BOARD MODE — CASH GEOMETRY SWEEP\n\n"
    return head + task + "\n" + block + "\n\n" + DISCLAIMER


def build_sweep_payload(sweep: Mapping) -> dict:
    """Compact JSON for Qwen. No world arrays."""
    rows = []
    for r in sweep["rows"]:
        d = {k: v for k, v in r.items() if k != "class_distribution"}
        d["class_distribution"] = dict(r["class_distribution"])
        rows.append(d)
    return {
        "disclaimer": DISCLAIMER,
        "experiment": "cash_sweep",
        "engine": "NULLXES-MAGA-ORACLE Monte Carlo",
        "llm_must_not_invent_numbers": True,
        "scenario": "cash_sweep",
        "policy": sweep["policy_view"],
        "simulation": {
            "worlds": sweep["worlds"],
            "days": sweep["days"],
            "seed": sweep["seed"],
            "gpu_used": False,
        },
        "known_inputs": {
            "cash_initial_grid": [r["cash_initial"] for r in rows],
            "only_changed": "company.cash_initial",
        },
        "assumptions": {
            "tag": sweep["assumptions_tag"],
            "burn_monthly": sweep["burn_monthly_assumption"],
            "cash_minimum": sweep["cash_minimum"],
        },
        "metrics": {
            "survival_probability": rows[0]["P(SURVIVAL)"] if rows else None,
            "cash_breach_probability": rows[0]["P(CASH BREACH)"] if rows else None,
        },
        "cash_sweep": {
            "rows": rows,
            "cliffs": sweep["cliffs"],
            "crn_invariants": sweep["crn_invariants"],
            "checkpoint_day": sweep["checkpoint_day"],
            "daily_burn_assumption": sweep["daily_burn_assumption"],
        },
        "absurdity": {
            "median_max_ai": rows[0]["absurdity_median_max"] if rows else None,
            "label": rows[0]["absurdity_label"] if rows else None,
        },
    }


def sweep_view_from_payload(payload: Mapping) -> dict:
    cs = payload.get("cash_sweep") or {}
    sim = payload.get("simulation") or {}
    ass = payload.get("assumptions") or {}
    return {
        "disclaimer": payload.get("disclaimer") or DISCLAIMER,
        "experiment": "cash_sweep",
        "seed": sim.get("seed"),
        "worlds": sim.get("worlds"),
        "days": sim.get("days"),
        "policy_view": payload.get("policy") or "HOLD",
        "checkpoint_day": cs.get("checkpoint_day"),
        "burn_monthly_assumption": ass.get("burn_monthly"),
        "daily_burn_assumption": cs.get("daily_burn_assumption"),
        "cash_minimum": ass.get("cash_minimum"),
        "crn_invariants": cs.get("crn_invariants") or {},
        "rows": cs.get("rows") or [],
        "cliffs": cs.get("cliffs") or {},
        "assumptions_tag": ass.get("tag") or "ASSUMPTION / PLACEHOLDER",
    }


def write_sweep_outputs(sweep: Mapping, root: Path, mode: str = "maga") -> dict:
    root = Path(root)
    sim = root / "outputs" / "simulations"
    charts = root / "outputs" / "charts"
    payload = build_sweep_payload(sweep)
    write_json(sweep, sim / "cash_sweep.json")
    write_json(payload, sim / "cash_sweep_payload.json")
    paths = write_cash_sweep_charts(sweep, charts)
    text = sweep_presenter(sweep, mode=mode)
    brief = root / "outputs" / "CASH_SWEEP_BRIEFING.md"
    brief.write_text(text + "\n", encoding="utf-8")
    return {"payload": payload, "briefing_text": text, "charts": paths, "brief_path": brief}
