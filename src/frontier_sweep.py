"""Cash0 × Burn survival frontier on common random numbers.

Only cash_initial and burn_monthly change. Same event/deal/bank worlds.
bridge_for_80 is the escape-solver min t=0 bridge on the best policy of that cell.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from assumptions import DISCLAIMER, daily_burn, set_path
from escape_solver import binary_search_min, policy_ranking
from payload import write_json
from simulation import make_streams, run_oracle, simulate_bundle

CASH_GRID: tuple[tuple[str, float], ...] = (
    ("200K", 200_000.0),
    ("599K", 599_000.0),
    ("1M", 1_000_000.0),
    ("2M", 2_000_000.0),
)
BURN_GRID: tuple[tuple[str, float], ...] = (
    ("200K", 200_000.0),
    ("500K", 500_000.0),
    ("1M", 1_000_000.0),
    ("1P5M", 1_500_000.0),
)

_DEAD = 0.05
_VIABLE = 0.50
_STABLE = 0.80


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def _cell_id(cash_lab: str, burn_lab: str) -> str:
    return f"C{cash_lab}_B{burn_lab}"


def identify_frontier(cells: list[dict], cash_labs: list[str], burn_labs: list[str]) -> dict:
    by = {(c["cash_label"], c["burn_label"]): c for c in cells}
    per_burn = []
    for bl in burn_labs:
        ordered = [by[(cl, bl)] for cl in cash_labs if (cl, bl) in by]

        def first(pred):
            for c in ordered:
                if pred(c):
                    return {"cell": c["id"], "cash0": c["cash_initial"], "burn": c["burn_monthly"]}
            return None

        bests = [c["best_P(SURVIVAL)"] for c in ordered]
        per_burn.append(
            {
                "burn_label": bl,
                "burn_monthly": ordered[0]["burn_monthly"] if ordered else None,
                "dead_zone": bool(ordered) and max(bests) < _DEAD,
                "min_cash_best_surv_ge_50": first(lambda c: c["best_P(SURVIVAL)"] >= _VIABLE),
                "min_cash_best_surv_ge_80": first(lambda c: c["best_P(SURVIVAL)"] >= _STABLE),
                "min_cash_hold_surv_ge_50": first(lambda c: c["HOLD_P(SURVIVAL)"] >= _VIABLE),
                "no_policy_saves_on_this_grid": bool(ordered) and max(bests) < _DEAD,
            }
        )
    return {
        "rule": (
            "Per burn: first Cash0 on this grid with best-policy P(SURVIVAL) "
            f">= {_VIABLE:.0%} / {_STABLE:.0%}. dead_zone = best < {_DEAD:.0%} at every Cash0. "
            "Not interpolated."
        ),
        "per_burn": per_burn,
    }


def run_frontier_sweep(
    cfg: Mapping,
    n_worlds: int | None = None,
    seed: int | None = None,
    cash_grid: tuple[tuple[str, float], ...] = CASH_GRID,
    burn_grid: tuple[tuple[str, float], ...] = BURN_GRID,
    solve_bridge: bool = True,
    overlay=None,
    live_events: list | None = None,
    bridge_steps: int = 16,
) -> dict:
    n = int(n_worlds if n_worlds is not None else cfg["simulation"]["worlds"])
    seed_i = int(seed if seed is not None else cfg["simulation"]["seed"])
    t_days = int(cfg["simulation"]["days"])
    streams = make_streams(n, t_days, seed_i)
    bundle = simulate_bundle(cfg, streams, overlay=overlay)
    target = float(cfg["escape"]["target_survival"])
    cells = []
    deals = set()
    banks = set()
    for burn_lab, burn in burn_grid:
        for cash_lab, cash0 in cash_grid:
            cfg_i = set_path(deepcopy(dict(cfg)), "company.cash_initial", float(cash0))
            cfg_i = set_path(cfg_i, "company.burn_monthly", float(burn))
            result = run_oracle(
                cfg_i,
                n_worlds=n,
                seed=seed_i,
                streams=streams,
                bundle=bundle,
                overlay=overlay,
                live_events=live_events,
            )
            ranking = policy_ranking(result)
            best = ranking[0]
            hold = result.metrics["HOLD"]
            daily = daily_burn(cfg_i)
            runway = (cash0 / daily) if daily > 1e-12 else None
            bridge = {
                "value": None,
                "feasible": None,
                "survival": None,
                "policy": best["policy"],
            }
            if solve_bridge:
                from escape_solver import _surv_cash

                hi = float(cfg_i["escape"]["bridge_search_hi"])

                def fn(x: float, _cfg=cfg_i, _res=result, _pol=best["policy"]) -> float:
                    cfg2 = set_path(_cfg, "company.bridge_capital", float(x))
                    return _surv_cash(cfg2, _res, _pol)

                x, s, ok = binary_search_min(fn, target, 0.0, hi, steps=int(bridge_steps))
                bridge = {
                    "value": float(x),
                    "feasible": bool(ok),
                    "survival": float(s),
                    "policy": best["policy"],
                }
            cid = _cell_id(cash_lab, burn_lab)
            deals.add(round(float(hold["P(DEAL CLOSE)"]), 12))
            banks.add(round(float(hold["P(BANK APPROVAL)"]), 12))
            cells.append(
                {
                    "id": cid,
                    "cash_label": cash_lab,
                    "burn_label": burn_lab,
                    "cash_initial": float(cash0),
                    "burn_monthly": float(burn),
                    "arithmetic_runway_days": _f(runway),
                    "HOLD_P(SURVIVAL)": float(hold["P(SURVIVAL)"]),
                    "HOLD_P(CASH BREACH)": float(hold["P(CASH BREACH)"]),
                    "best_policy": best["policy"],
                    "best_P(SURVIVAL)": float(best["P(SURVIVAL)"]),
                    "best_P(CASH BREACH)": float(best["P(CASH BREACH)"]),
                    "best_minus_HOLD_pp": (float(best["P(SURVIVAL)"]) - float(hold["P(SURVIVAL)"]))
                    * 100.0,
                    "P(DEAL CLOSE)": float(hold["P(DEAL CLOSE)"]),
                    "P(BANK APPROVAL)": float(hold["P(BANK APPROVAL)"]),
                    "bridge_for_80": bridge,
                    "band": (
                        "DEAD"
                        if float(best["P(SURVIVAL)"]) < _DEAD
                        else "FRAGILE"
                        if float(best["P(SURVIVAL)"]) < _VIABLE
                        else "VIABLE"
                        if float(best["P(SURVIVAL)"]) < _STABLE
                        else "STABLE"
                    ),
                }
            )
    cash_labs = [a[0] for a in cash_grid]
    burn_labs = [a[0] for a in burn_grid]
    frontier = identify_frontier(cells, cash_labs, burn_labs)
    return {
        "disclaimer": DISCLAIMER,
        "experiment": "frontier_sweep",
        "llm_must_not_invent_numbers": True,
        "only_changed": ["company.cash_initial", "company.burn_monthly"],
        "priors_frozen": True,
        "common_random_numbers": True,
        "seed": seed_i,
        "worlds": n,
        "days": t_days,
        "target_survival": target,
        "cash_grid": [{"label": a, "value": b} for a, b in cash_grid],
        "burn_grid": [{"label": a, "value": b} for a, b in burn_grid],
        "crn_invariants": {
            "deal_close_identical": len(deals) == 1,
            "bank_approval_identical": len(banks) == 1,
            "note": "Deal/bank sit on the event layer and must not move with Cash0 or burn.",
        },
        "cells": cells,
        "frontier": frontier,
        "assumptions_tag": "ASSUMPTION / PLACEHOLDER — other priors frozen",
    }


def write_frontier_charts(sweep: Mapping, chart_dir: str | Path) -> dict[str, Path]:
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)
    cash_labs = [x["label"] for x in sweep["cash_grid"]]
    burn_labs = [x["label"] for x in sweep["burn_grid"]]
    cells = {(c["cash_label"], c["burn_label"]): c for c in sweep["cells"]}
    n_b, n_c = len(burn_labs), len(cash_labs)
    best = np.full((n_b, n_c), np.nan)
    hold = np.full((n_b, n_c), np.nan)
    brdg = np.full((n_b, n_c), np.nan)
    for i, bl in enumerate(burn_labs):
        for j, cl in enumerate(cash_labs):
            c = cells[(cl, bl)]
            best[i, j] = c["best_P(SURVIVAL)"]
            hold[i, j] = c["HOLD_P(SURVIVAL)"]
            bv = (c.get("bridge_for_80") or {}).get("value")
            ok = (c.get("bridge_for_80") or {}).get("feasible")
            if ok and bv is not None:
                brdg[i, j] = bv

    def _heat(mat, title, path, cmap, vmin, vmax, fmt):
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        im = ax.imshow(mat, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_c), cash_labs)
        ax.set_yticks(range(n_b), burn_labs)
        ax.set_xlabel("Cash0")
        ax.set_ylabel("Burn monthly")
        ax.set_title(title)
        for i in range(n_b):
            for j in range(n_c):
                v = mat[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "n/a", ha="center", va="center", color="0.3", fontsize=8)
                else:
                    ax.text(j, i, fmt(v), ha="center", va="center", color="black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    paths = {}
    p = chart_dir / "frontier_best_survival.png"
    _heat(best, "Survival frontier  |  best policy P(SURVIVAL)  |  CRN", p, "RdYlGn", 0, 1, lambda v: f"{v:.0%}")
    paths["frontier_best_survival"] = p
    p = chart_dir / "frontier_hold_survival.png"
    _heat(hold, "HOLD P(SURVIVAL)  |  same worlds", p, "RdYlGn", 0, 1, lambda v: f"{v:.0%}")
    paths["frontier_hold_survival"] = p
    p = chart_dir / "frontier_bridge_80.png"
    _heat(
        brdg,
        "bridge_for_80% (best policy), RUB  |  n/a = infeasible on search range",
        p,
        "YlOrRd",
        0,
        np.nanmax(brdg) if np.isfinite(brdg).any() else 1,
        lambda v: f"{v/1e6:.2f}M" if v >= 1e6 else f"{v:,.0f}",
    )
    paths["frontier_bridge_80"] = p
    return paths


def authoritative_frontier_block(sweep: Mapping) -> str:
    lines = [
        "AUTHORITATIVE FACTS",
        "Cash0 × Burn frontier. Same seed, same event/deal/bank worlds.",
        "Only cash_initial and burn_monthly change. Do not interpolate.",
        f"SEED: {sweep['seed']} WORLDS: {sweep['worlds']} DAYS: {sweep['days']} "
        f"TARGET_SURVIVAL: {sweep['target_survival']}",
        f"CRN deal_close_identical={sweep['crn_invariants']['deal_close_identical']} "
        f"bank_approval_identical={sweep['crn_invariants']['bank_approval_identical']}",
    ]
    for c in sweep["cells"]:
        br = c["bridge_for_80"]
        if br.get("feasible") is True:
            btxt = f"{br['value']:.0f}"
        elif br.get("feasible") is False:
            btxt = "infeasible"
        else:
            btxt = "данных недостаточно"
        lines.append(
            f"{c['id']} Cash0: {c['cash_initial']:.0f} Burn: {c['burn_monthly']:.0f} "
            f"runway_days: {c['arithmetic_runway_days']:.6g} "
            f"HOLD P(SURVIVAL): {c['HOLD_P(SURVIVAL)']:.6f} "
            f"HOLD P(CASH BREACH): {c['HOLD_P(CASH BREACH)']:.6f} "
            f"best_policy: {c['best_policy']} "
            f"BEST P(SURVIVAL): {c['best_P(SURVIVAL)']:.6f} "
            f"BEST P(CASH BREACH): {c['best_P(CASH BREACH)']:.6f} "
            f"best_minus_HOLD_pp: {c['best_minus_HOLD_pp']:.4f} "
            f"BRIDGE@80%: {btxt} "
            f"band: {c['band']} "
            f"P(DEAL CLOSE): {c['P(DEAL CLOSE)']:.6f} "
            f"P(BANK APPROVAL): {c['P(BANK APPROVAL)']:.6f}"
        )
    lines.append("FRONTIER per burn (first Cash0 on grid):")
    for row in sweep["frontier"]["per_burn"]:
        def _hit(key):
            h = row.get(key)
            return f"{h['cell']} Cash0={h['cash0']:.0f}" if h else "данных недостаточно"

        lines.append(
            f"BURN {row['burn_label']} {row['burn_monthly']:.0f} "
            f"dead_zone={row['dead_zone']} "
            f"min_cash_best_ge_50: {_hit('min_cash_best_surv_ge_50')} "
            f"min_cash_best_ge_80: {_hit('min_cash_best_surv_ge_80')} "
            f"min_cash_HOLD_ge_50: {_hit('min_cash_hold_surv_ge_50')}"
        )
    lines.append("END AUTHORITATIVE FACTS")
    return "\n".join(lines)


def frontier_presenter(sweep: Mapping, mode: str = "maga") -> str:
    block = authoritative_frontier_block(sweep)
    task = (
        "Сравнительная задача (только по FACTS):\n"
        "1. При каком burn никакая политика на этой сетке не спасает (dead_zone).\n"
        "2. Для каждого burn — минимальный Cash0, где best P(SURVIVAL) ≥ 50% и ≥ 80%.\n"
        "3. Где HOLD мёртв, а политика ещё жива (best_minus_HOLD_pp).\n"
        "4. bridge_for_80: сколько моста нужно на best policy; infeasible = в диапазоне поиска не достали 80%.\n"
        "5. Что арифметика runway (Cash0/daily burn), а что interaction со shocks и timing.\n"
        "Не интерполируй. Пропуск = «данных недостаточно».\n"
    )
    dead = [r["burn_label"] for r in sweep["frontier"]["per_burn"] if r.get("dead_zone")]
    head = (
        "MAGA MODE — SURVIVAL FRONTIER (Cash0 × Burn)\n"
        f"Dead burns on this grid: {', '.join(dead) if dead else 'none'}\n\n"
        "Ниже только факты симуляции. Это не прогноз мира.\n\n"
        if mode == "maga"
        else "BOARD MODE — SURVIVAL FRONTIER\n\n"
    )
    return head + task + "\n" + block + "\n\n" + DISCLAIMER


def frontier_view_from_payload(payload: Mapping) -> dict:
    fs = payload.get("frontier_sweep") or payload
    sim = payload.get("simulation") or {}
    return {
        "disclaimer": payload.get("disclaimer") or DISCLAIMER,
        "experiment": "frontier_sweep",
        "seed": sim.get("seed") or fs.get("seed"),
        "worlds": sim.get("worlds") or fs.get("worlds"),
        "days": sim.get("days") or fs.get("days"),
        "target_survival": fs.get("target_survival"),
        "crn_invariants": fs.get("crn_invariants") or payload.get("crn_invariants") or {},
        "cells": fs.get("cells") or [],
        "frontier": fs.get("frontier") or {},
        "cash_grid": fs.get("cash_grid") or [],
        "burn_grid": fs.get("burn_grid") or [],
    }


def build_frontier_payload(sweep: Mapping) -> dict:
    return {
        "disclaimer": DISCLAIMER,
        "experiment": "frontier_sweep",
        "engine": "NULLXES-MAGA-ORACLE Monte Carlo",
        "llm_must_not_invent_numbers": True,
        "scenario": "frontier_sweep",
        "simulation": {
            "worlds": sweep["worlds"],
            "days": sweep["days"],
            "seed": sweep["seed"],
            "gpu_used": False,
        },
        "known_inputs": {"only_changed": sweep["only_changed"]},
        "assumptions": {"tag": sweep["assumptions_tag"]},
        "metrics": {},
        "frontier_sweep": {
            "cells": sweep["cells"],
            "frontier": sweep["frontier"],
            "crn_invariants": sweep["crn_invariants"],
            "target_survival": sweep["target_survival"],
            "cash_grid": sweep["cash_grid"],
            "burn_grid": sweep["burn_grid"],
        },
    }


def write_frontier_outputs(sweep: Mapping, root: Path, mode: str = "maga") -> dict:
    root = Path(root)
    sim = root / "outputs" / "simulations"
    charts = root / "outputs" / "charts"
    payload = build_frontier_payload(sweep)
    write_json(sweep, sim / "frontier_sweep.json")
    write_json(payload, sim / "frontier_sweep_payload.json")
    paths = write_frontier_charts(sweep, charts)
    text = frontier_presenter(sweep, mode=mode)
    brief = root / "outputs" / "FRONTIER_SWEEP_BRIEFING.md"
    brief.write_text(text + "\n", encoding="utf-8")
    return {"payload": payload, "briefing_text": text, "charts": paths, "brief_path": brief}
