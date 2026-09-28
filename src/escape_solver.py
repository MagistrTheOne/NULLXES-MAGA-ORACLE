"""Escape-route solver on common random numbers.

The solver does not recommend a 'true' strategy. It reports the values of
configured levers at which the *model* P(SURVIVAL) crosses the target
under the given ASSUMPTION / PLACEHOLDER priors.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from assumptions import set_path
from policies import enabled_policies
from sensitivity import evaluate_override, tornado_table
from simulation import OracleResult, run_oracle, survival_prob


def _nseed(baseline: OracleResult) -> tuple[int, int]:
    n = int(baseline.bundle.events.shape[0])
    seed = int(baseline.cfg["simulation"]["seed"])
    return n, seed


def _surv_cash(
    cfg: Mapping,
    baseline: OracleResult,
    policy: str,
) -> float:
    n, seed = _nseed(baseline)
    r = run_oracle(
        cfg,
        n_worlds=n,
        seed=seed,
        policies=[policy],
        streams=baseline.streams,
        bundle=baseline.bundle,
    )
    return survival_prob(r, policy)


def _surv_key(
    cfg: Mapping,
    baseline: OracleResult,
    key: str,
    value: Any,
    policy: str,
) -> float:
    return evaluate_override(cfg, baseline, key, value, policy)


def binary_search_min(
    fn,
    target: float,
    lo: float,
    hi: float,
    steps: int = 22,
) -> tuple[float, float, bool]:
    """Smallest x in [lo,hi] with fn(x) >= target. fn must be nondecreasing in x."""
    s_lo = fn(lo)
    s_hi = fn(hi)
    if s_hi < target:
        return hi, s_hi, False
    if s_lo >= target:
        return lo, s_lo, True
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        s = fn(mid)
        if s >= target:
            hi = mid
            s_hi = s
        else:
            lo = mid
            s_lo = s
    return hi, s_hi, True


def binary_search_max(
    fn,
    target: float,
    lo: float,
    hi: float,
    steps: int = 22,
) -> tuple[float, float, bool]:
    """Largest x in [lo,hi] with fn(x) >= target. fn must be nonincreasing in x."""
    s_lo = fn(lo)
    s_hi = fn(hi)
    if s_lo < target:
        return lo, s_lo, False
    if s_hi >= target:
        return hi, s_hi, True
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        s = fn(mid)
        if s >= target:
            lo = mid
            s_lo = s
        else:
            hi = mid
            s_hi = s
    return lo, s_lo, True


def min_bridge(
    cfg: Mapping, baseline: OracleResult, policy: str, target: float
) -> dict:
    hi = float(cfg["escape"]["bridge_search_hi"])

    def fn(x: float) -> float:
        cfg2 = set_path(cfg, "company.bridge_capital", float(x))
        return _surv_cash(cfg2, baseline, policy)

    x, s, ok = binary_search_min(fn, target, 0.0, hi)
    return {
        "lever": "bridge_capital",
        "policy": policy,
        "value": float(x),
        "survival": float(s),
        "feasible": ok,
        "note": "Minimum t=0 additive cash for target P(SURVIVAL) under this policy.",
    }


def max_burn(
    cfg: Mapping, baseline: OracleResult, policy: str, target: float
) -> dict:
    hi = float(cfg["escape"]["burn_search_hi"])

    def fn(x: float) -> float:
        cfg2 = set_path(cfg, "company.burn_monthly", float(max(x, 0.0)))
        return _surv_cash(cfg2, baseline, policy)

    x, s, ok = binary_search_max(fn, target, 0.0, hi)
    return {
        "lever": "burn_monthly",
        "policy": policy,
        "value": float(x),
        "survival": float(s),
        "feasible": ok,
        "note": "Maximum monthly burn for target P(SURVIVAL) under this policy.",
    }


def financing_timing_curve(
    cfg: Mapping, baseline: OracleResult, policy: str = "HOLD"
) -> list[dict]:
    """Shift bank max delay and deal max delay together (financing layer only)."""
    from sensitivity import _rebuild_financing

    n, seed = _nseed(baseline)
    rows = []
    bank_hi_grid = [3, 7, 14, 21, 30, 45, 60]
    deal_hi_grid = [7, 14, 21, 30, 45, 60, 80]
    for b, d in zip(bank_hi_grid, deal_hi_grid):
        cfg2 = set_path(cfg, "bank.decision_delay_distribution.max_days", int(b))
        cfg2 = set_path(cfg2, "bank.decision_delay_distribution.min_days", 0)
        cfg2 = set_path(cfg2, "deal.delay_distribution.max_days", int(d))
        cfg2 = set_path(cfg2, "deal.delay_distribution.min_days", 0)
        bundle = _rebuild_financing(cfg2, baseline.bundle, baseline.streams)
        r = run_oracle(
            cfg2,
            n_worlds=n,
            seed=seed,
            policies=[policy],
            streams=baseline.streams,
            bundle=bundle,
        )
        rows.append(
            {
                "bank_max_delay_days": int(b),
                "deal_max_delay_days": int(d),
                "survival": survival_prob(r, policy),
                "policy": policy,
            }
        )
    return rows


def bank_vs_freeze_curve(cfg: Mapping, baseline: OracleResult) -> dict:
    """For which P(bank approval) is waiting (HOLD) at least as good as immediate FREEZE_90?"""
    from sensitivity import _rebuild_financing

    n, seed = _nseed(baseline)
    cfg_fz = deepcopy(cfg)
    cfg_fz["policies"]["FREEZE_90"]["trigger"] = {
        "cash_below": 1e18,
        "min_day": 0,
    }
    grid = np.round(np.linspace(0.0, 1.0, 11), 4)
    rows = []
    crossover = None
    for p in grid:
        p = float(p)
        cfg_h = set_path(cfg, "bank.approval_probability", p)
        cfg_f = set_path(cfg_fz, "bank.approval_probability", p)
        bundle_h = _rebuild_financing(cfg_h, baseline.bundle, baseline.streams)
        bundle_f = _rebuild_financing(cfg_f, baseline.bundle, baseline.streams)
        r_h = run_oracle(
            cfg_h,
            n_worlds=n,
            seed=seed,
            policies=["HOLD"],
            streams=baseline.streams,
            bundle=bundle_h,
        )
        r_f = run_oracle(
            cfg_f,
            n_worlds=n,
            seed=seed,
            policies=["FREEZE_90"],
            streams=baseline.streams,
            bundle=bundle_f,
        )
        s_h = survival_prob(r_h, "HOLD")
        s_f = survival_prob(r_f, "FREEZE_90")
        rows.append(
            {
                "p_bank_approval": p,
                "survival_HOLD": s_h,
                "survival_immediate_FREEZE_90": s_f,
                "delta_HOLD_minus_FREEZE": s_h - s_f,
            }
        )
        if crossover is None and s_h >= s_f:
            crossover = p
    return {
        "curve": rows,
        "crossover_p_bank_approval": crossover,
        "note": (
            "Crossover is the smallest ASSUMPTION P(bank approval) at which HOLD "
            "survival >= immediate FREEZE_90 survival on the same worlds. "
            "This is not a recommendation."
        ),
    }


def min_contract_inflow(
    cfg: Mapping, baseline: OracleResult, policy: str
) -> dict:
    target_lift = float(cfg["escape"]["material_survival_lift"])
    base_s = float(baseline.metrics[policy]["P(SURVIVAL)"])
    goal = min(1.0, base_s + target_lift)
    hi = float(cfg["escape"]["contract_search_hi"])

    def fn(x: float) -> float:
        cfg2 = set_path(cfg, "contract.daily_inflow", float(x))
        return _surv_cash(cfg2, baseline, policy)

    x, s, ok = binary_search_min(fn, goal, 0.0, hi)
    return {
        "lever": "contract.daily_inflow",
        "policy": policy,
        "value": float(x),
        "survival": float(s),
        "baseline_survival": base_s,
        "goal_survival": goal,
        "feasible": ok,
        "note": (
            f"Minimum constant daily contract inflow that lifts P(SURVIVAL) by "
            f"at least {target_lift:.0%} (or hits 100%)."
        ),
    }


def policy_ranking(baseline: OracleResult) -> list[dict]:
    rows = []
    for name, m in baseline.metrics.items():
        rows.append(
            {
                "policy": name,
                "P(SURVIVAL)": m["P(SURVIVAL)"],
                "P(CASH BREACH)": m["P(CASH BREACH)"],
                "C90 median": m["c90_median"],
                "ES5": m["es5"],
                "P(class ESCAPE)": m.get("P(ESCAPE)", float("nan")),
                "P(class FAIL)": m.get("P(FAIL)", float("nan")),
            }
        )
    rows.sort(key=lambda r: r["P(SURVIVAL)"], reverse=True)
    return rows


def minimal_escape_set(
    cfg: Mapping,
    baseline: OracleResult,
    target: float,
) -> dict:
    """Small-cardinality controllable changes that reach target survival.

    Controllable levers: policy, bridge_capital, burn_monthly, contract.daily_inflow.
    Rank by (n_levers, bridge, contract, burn_cut). Not a unique optimum.
    """
    ranking = policy_ranking(baseline)
    best_pol = ranking[0]["policy"]
    candidates: list[dict] = []

    for pol in enabled_policies(cfg):
        s0 = float(baseline.metrics[pol]["P(SURVIVAL)"])
        if s0 >= target:
            candidates.append(
                {
                    "n_levers": 1 if pol != "HOLD" else 0,
                    "policy": pol,
                    "bridge_capital": 0.0,
                    "burn_monthly": float(cfg["company"]["burn_monthly"]),
                    "contract_daily_inflow": float(cfg["contract"]["daily_inflow"]),
                    "survival": s0,
                    "description": f"policy={pol} only",
                }
            )

    for pol in enabled_policies(cfg):
        br = min_bridge(cfg, baseline, pol, target)
        if br["feasible"]:
            n_lev = 1 + (0 if pol == "HOLD" else 1)
            candidates.append(
                {
                    "n_levers": n_lev,
                    "policy": pol,
                    "bridge_capital": br["value"],
                    "burn_monthly": float(cfg["company"]["burn_monthly"]),
                    "contract_daily_inflow": float(cfg["contract"]["daily_inflow"]),
                    "survival": br["survival"],
                    "description": f"policy={pol} + bridge {br['value']:.0f} RUB",
                }
            )
        bu = max_burn(cfg, baseline, pol, target)
        if bu["feasible"]:
            n_lev = 1 + (0 if pol == "HOLD" else 1)
            if abs(bu["value"] - float(cfg["company"]["burn_monthly"])) > 1.0:
                n_lev = 1 + (0 if pol == "HOLD" else 1)
            candidates.append(
                {
                    "n_levers": n_lev if bu["value"] != float(cfg["company"]["burn_monthly"]) else (0 if pol == "HOLD" else 1),
                    "policy": pol,
                    "bridge_capital": 0.0,
                    "burn_monthly": bu["value"],
                    "contract_daily_inflow": float(cfg["contract"]["daily_inflow"]),
                    "survival": bu["survival"],
                    "description": f"policy={pol} + burn_monthly {bu['value']:.0f} RUB",
                }
            )

    # policy + bridge + burn=0 (extreme but admissible)
    cfg_z = set_path(cfg, "company.burn_monthly", 0.0)
    br0 = min_bridge(cfg_z, baseline, best_pol, target)
    if br0["feasible"]:
        candidates.append(
            {
                "n_levers": 3,
                "policy": best_pol,
                "bridge_capital": br0["value"],
                "burn_monthly": 0.0,
                "contract_daily_inflow": 0.0,
                "survival": br0["survival"],
                "description": f"policy={best_pol} + burn=0 + bridge {br0['value']:.0f}",
            }
        )

    if not candidates:
        return {
            "found": False,
            "best_policy_by_survival": best_pol,
            "message": "No tested controllable combination reached the target under these assumptions.",
            "candidates": [],
        }

    def keyfn(c: dict) -> tuple:
        burn_cut = max(0.0, float(cfg["company"]["burn_monthly"]) - float(c["burn_monthly"]))
        return (
            int(c["n_levers"]),
            float(c["bridge_capital"]),
            float(c["contract_daily_inflow"]),
            burn_cut,
            -float(c["survival"]),
        )

    candidates.sort(key=keyfn)
    return {
        "found": True,
        "best_policy_by_survival": best_pol,
        "chosen": candidates[0],
        "candidates": candidates[:12],
        "note": (
            "Chosen set minimises (number of levers, bridge capital, contract inflow, "
            "burn cut) among combinations that hit the target on these worlds. "
            "It is not an objective recommendation."
        ),
    }


def solve_escape(
    cfg: Mapping,
    baseline: OracleResult,
    sensitivity_rows: list[dict] | None = None,
) -> dict:
    target = float(cfg["escape"]["target_survival"])
    ranking = policy_ranking(baseline)
    best_pol = ranking[0]["policy"]
    tornado = tornado_table(sensitivity_rows or [])
    strongest = tornado[0] if tornado else None

    bridge = min_bridge(cfg, baseline, best_pol, target)
    burn = max_burn(cfg, baseline, best_pol, target)
    timing = financing_timing_curve(cfg, baseline, policy="HOLD")
    bank_fz = bank_vs_freeze_curve(cfg, baseline)
    contract = min_contract_inflow(cfg, baseline, best_pol)
    combo = minimal_escape_set(cfg, baseline, target)

    return {
        "target_survival": target,
        "best_policy_by_survival": best_pol,
        "policy_ranking": ranking,
        "min_bridge_for_target": bridge,
        "max_burn_for_target": burn,
        "financing_timing_curve": timing,
        "strongest_sensitivity": strongest,
        "bank_vs_immediate_freeze": bank_fz,
        "min_contract_inflow_material": contract,
        "minimal_escape_set": combo,
        "disclaimer": (
            "Результаты являются условными исходами модели при заданных допущениях "
            "и не являются прогнозом реальных геополитических, военных или экономических событий."
        ),
    }
