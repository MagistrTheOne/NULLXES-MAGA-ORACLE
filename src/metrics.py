"""Outcome classification and Monte Carlo metrics.

Classes are mutually exclusive with this priority (first match wins):

FAIL     : exists t in [0, T): Cash_t < Cmin
RU_EXIT  : not FAIL, and EXIT_RU trigger fired
ESCAPE   : not FAIL, not RU_EXIT, material financing received,
           and Cash_T >= Cmin + escape_runway_days * daily_burn
FREEZE   : not FAIL, not RU_EXIT, not ESCAPE, and 90%-style freeze was active
           (FREEZE_90 policy, or EXIT_RU freeze, or freeze_fraction >= 0.90)
TRAP     : not FAIL, not RU_EXIT, not ESCAPE, not FREEZE,
           and (no material financing or Cash_T < Cmin + trap_runway_days * daily_burn)
SURVIVE  : not FAIL and none of the above

Metric P(SURVIVAL) := P(no cash breach) = 1 - P(FAIL).
This is NOT the same as P(class == SURVIVE).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from assumptions import (
    CLASS_ESCAPE,
    CLASS_FAIL,
    CLASS_FREEZE,
    CLASS_NAMES,
    CLASS_RU_EXIT,
    CLASS_SURVIVE,
    CLASS_TRAP,
    daily_burn as base_daily_burn,
)
from cashflow import CashResult
from financing import (
    BANK_APPROVE,
    BANK_DECLINE,
    BANK_DELAY,
    BANK_REDUCED_LIMIT,
    DEAL_CLOSED,
    DEAL_COUNTERPARTY,
    DEAL_DELAY,
    DEAL_FAIL,
    DEAL_LEGAL_BLOCK,
    DEAL_SANCTIONS_BLOCK,
    DealResult,
    BankResult,
)


def classify(
    cfg: Mapping,
    cash_res: CashResult,
    deal: DealResult,
    bank: BankResult,
    freeze_is_90: bool,
) -> np.ndarray:
    cmin = float(cfg["company"]["cash_minimum"])
    burn = max(base_daily_burn(cfg), 0.0)
    material = float(cfg["outcomes"]["material_financing_rub"])
    trap_run = float(cfg["outcomes"]["trap_runway_days"])
    esc_run = float(cfg["outcomes"]["escape_runway_days"])

    cash90 = cash_res.cash[:, -1]
    failed = cash_res.first_breach_day >= 0
    ru_exit = cash_res.exit_ever & ~failed
    total_in = deal.inflows.sum(axis=1) + bank.inflows.sum(axis=1)
    got_material = total_in >= material
    escape_floor = cmin + esc_run * burn
    trap_floor = cmin + trap_run * burn
    escaped = (~failed) & (~ru_exit) & got_material & (cash90 >= escape_floor)
    freeze90 = cash_res.freeze_ever if freeze_is_90 else np.zeros_like(failed)
    # CAPITAL_FIRST uses fraction 1.0; count as freeze-class only for FREEZE_90
    froze = (~failed) & (~ru_exit) & (~escaped) & freeze90
    trapped = (
        (~failed)
        & (~ru_exit)
        & (~escaped)
        & (~froze)
        & ((~got_material) | (cash90 < trap_floor))
    )
    survived = (~failed) & (~ru_exit) & (~escaped) & (~froze) & (~trapped)

    cls = np.empty(failed.shape[0], dtype=np.int64)
    cls[failed] = CLASS_FAIL
    cls[ru_exit] = CLASS_RU_EXIT
    cls[escaped] = CLASS_ESCAPE
    cls[froze] = CLASS_FREEZE
    cls[trapped] = CLASS_TRAP
    cls[survived] = CLASS_SURVIVE
    return cls


def _prob(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float64)))


def expected_shortfall(x: np.ndarray, q: float = 0.05) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    thresh = np.quantile(x, q)
    tail = x[x <= thresh]
    if tail.size == 0:
        return float(thresh)
    return float(tail.mean())


def time_to_first_breach_summary(first_breach: np.ndarray, days: int) -> dict:
    hit = first_breach[first_breach >= 0]
    hist = np.zeros(days, dtype=np.int64)
    if hit.size:
        binc = np.bincount(hit, minlength=days)
        hist[: min(days, binc.size)] = binc[:days]
    return {
        "n_breached": int((first_breach >= 0).sum()),
        "n_never": int((first_breach < 0).sum()),
        "median_day_if_breach": float(np.median(hit)) if hit.size else float("nan"),
        "mean_day_if_breach": float(hit.mean()) if hit.size else float("nan"),
        "histogram": hist,
    }


def percentile_bands(cash: np.ndarray, qs=(5, 25, 50, 75, 95)) -> dict[str, np.ndarray]:
    return {f"p{q}": np.quantile(cash, q / 100.0, axis=0) for q in qs}


def metrics_for_policy(
    cfg: Mapping,
    cash_res: CashResult,
    deal: DealResult,
    bank: BankResult,
    events: np.ndarray,
    shock_loss: np.ndarray,
    cls: np.ndarray,
    policy_name: str,
) -> dict:
    cash90 = cash_res.cash[:, -1]
    cmin = float(cfg["company"]["cash_minimum"])
    major = float(cfg["outcomes"]["major_shock_loss_rub"])
    breached = cash_res.first_breach_day >= 0
    survived = ~breached
    freeze90 = policy_name == "FREEZE_90"
    any_major = (shock_loss >= major).any(axis=1) | events[:, :, -1].any(axis=1)

    qs = {
        "c90_median": float(np.median(cash90)),
        "c90_p5": float(np.quantile(cash90, 0.05)),
        "c90_p25": float(np.quantile(cash90, 0.25)),
        "c90_p75": float(np.quantile(cash90, 0.75)),
        "c90_p95": float(np.quantile(cash90, 0.95)),
        "es5": expected_shortfall(cash90, 0.05),
    }

    class_share = {
        f"P({CLASS_NAMES[k]})": _prob(cls == k) for k in CLASS_NAMES
    }

    out = {
        "policy": policy_name,
        "n_worlds": int(cash90.size),
        "P(SURVIVAL)": _prob(survived),
        "P(CASH BREACH)": _prob(breached),
        "P(FINANCING BEFORE BREACH)": _prob(cash_res.financed_before_breach),
        "P(DEAL CLOSE)": _prob(deal.status == DEAL_CLOSED),
        "P(DEAL FAIL)": _prob(deal.status == DEAL_FAIL),
        "P(DEAL DELAY)": _prob(deal.status == DEAL_DELAY),
        "P(SANCTIONS_BLOCK)": _prob(deal.status == DEAL_SANCTIONS_BLOCK),
        "P(LEGAL_BLOCK)": _prob(deal.status == DEAL_LEGAL_BLOCK),
        "P(COUNTERPARTY_FAILURE)": _prob(deal.status == DEAL_COUNTERPARTY),
        "P(BANK APPROVAL)": _prob(
            (bank.status == BANK_APPROVE) | (bank.status == BANK_REDUCED_LIMIT)
        ),
        "P(BANK FULL APPROVE)": _prob(bank.status == BANK_APPROVE),
        "P(BANK REDUCED LIMIT)": _prob(bank.status == BANK_REDUCED_LIMIT),
        "P(BANK DECLINE)": _prob(bank.status == BANK_DECLINE),
        "P(BANK DELAY)": _prob(bank.status == BANK_DELAY),
        "P(90% R&D FREEZE)": _prob(cash_res.freeze_ever) if freeze90 else _prob(
            cash_res.freeze_ever & (policy_name in ("FREEZE_90", "EXIT_RU", "CAPITAL_FIRST"))
        ),
        "P(RU CONTOUR EXIT TRIGGER)": _prob(cash_res.exit_ever),
        "P(ANY MAJOR SHOCK <= 90 days)": _prob(any_major),
        "median_min_cash": float(np.median(cash_res.min_cash)),
        "median_max_drawdown": float(np.median(cash_res.max_drawdown)),
        "mean_crisis_days": float(cash_res.crisis_days.mean()),
        "mean_max_simultaneous_crises": float(cash_res.max_simultaneous.mean()),
        **qs,
        **class_share,
        "ttfb": time_to_first_breach_summary(
            cash_res.first_breach_day, int(cfg["simulation"]["days"])
        ),
        "bands": percentile_bands(cash_res.cash),
        "cash90": cash90,
        "class": cls,
        "first_breach_day": cash_res.first_breach_day,
    }
    return out


def finite_ok(x: np.ndarray) -> bool:
    return bool(np.isfinite(x).all())
