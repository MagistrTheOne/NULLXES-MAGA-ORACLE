"""Cash dynamics, vectorized over worlds, looped over days.

Cash_t =
    Cash_{t-1}
    + Revenue_t
    + Financing_t
    + DealInflows_t
    + ContractInflows_t
    - Burn_t
    - ShockLoss_t
    - FinancingCost_t
    - OneOffCosts_t
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from assumptions import CATEGORIES, daily_burn as base_daily_burn
from financing import BANK_DECLINE, DEAL_FAIL
from policies import burn_for_day, policy_spec, update_exit_flag, update_freeze_flag


@dataclass
class CashResult:
    cash: np.ndarray
    burn: np.ndarray
    freeze_on: np.ndarray
    exit_on: np.ndarray
    first_breach_day: np.ndarray
    min_cash: np.ndarray
    max_drawdown: np.ndarray
    crisis_days: np.ndarray
    max_simultaneous: np.ndarray
    freeze_ever: np.ndarray
    exit_ever: np.ndarray
    financed_before_breach: np.ndarray
    one_off: np.ndarray
    first_financing_day: np.ndarray


def _first_positive_day(x: np.ndarray) -> np.ndarray:
    hit = x > 0.0
    any_t = hit.any(axis=1)
    idx = np.argmax(hit, axis=1)
    return np.where(any_t, idx, -1)


def _macro_multiplier(cfg: Mapping, severity: np.ndarray) -> np.ndarray:
    midx = CATEGORIES.index("MACRO")
    kappa = float(cfg["macro_pass_through"]["burn_kappa"])
    clip = float(cfg["macro_pass_through"]["burn_cum_sev_clip"])
    cum = np.cumsum(severity[:, :, midx], axis=1)
    return np.exp(kappa * np.clip(cum, 0.0, clip))


def simulate_cash(
    cfg: Mapping,
    policy_name: str,
    shock_loss: np.ndarray,
    events: np.ndarray,
    severity: np.ndarray,
    deal_inflows: np.ndarray,
    bank_inflows: np.ndarray,
    revenue: np.ndarray,
    contract: np.ndarray,
    deal_status: np.ndarray,
    bank_status: np.ndarray,
    deal_fail_day: np.ndarray,
    bank_decision_day: np.ndarray,
) -> CashResult:
    n, t_days = shock_loss.shape
    cmin = float(cfg["company"]["cash_minimum"])
    cash0 = float(cfg["company"]["cash_initial"]) + float(
        cfg["company"].get("bridge_capital", 0.0)
    )
    annual = float(cfg["bank"]["financing_cost"]["annual_rate"])
    daily_rate = annual / 365.0
    base_b = max(base_daily_burn(cfg), 0.0)
    geo_idx = CATEGORIES.index("GEO")
    macro_mult = _macro_multiplier(cfg, severity)

    cash = np.empty((n, t_days), dtype=np.float64)
    burn_path = np.empty((n, t_days), dtype=np.float64)
    freeze_path = np.zeros((n, t_days), dtype=np.bool_)
    exit_path = np.zeros((n, t_days), dtype=np.bool_)
    one_off_path = np.zeros((n, t_days), dtype=np.float64)

    c = np.full(n, cash0, dtype=np.float64)
    freeze_on = np.zeros(n, dtype=np.bool_)
    exit_on = np.zeros(n, dtype=np.bool_)
    peak = c.copy()
    max_dd = np.zeros(n, dtype=np.float64)
    first_breach = np.full(n, -1, dtype=np.int64)
    drawn = np.zeros(n, dtype=np.float64)
    financed = np.zeros(n, dtype=np.bool_)
    geo_cum = np.zeros(n, dtype=np.int64)

    bank_declined_flags = bank_status == BANK_DECLINE
    deal_failed_flags = deal_status == DEAL_FAIL
    exit_cost = (
        float(policy_spec(cfg, "EXIT_RU").get("one_off_cost", 0.0))
        if policy_name == "EXIT_RU"
        else 0.0
    )

    for t in range(t_days):
        opening = c
        freeze_on = update_freeze_flag(
            cfg, policy_name, opening, t, freeze_on, financed, base_b
        )
        geo_cum = geo_cum + events[:, t, geo_idx].astype(np.int64)
        deal_failed_now = deal_failed_flags & (deal_fail_day == t)
        bank_declined_now = bank_declined_flags & (bank_decision_day == t)

        if policy_name == "EXIT_RU":
            prev_exit = exit_on.copy()
            exit_on = update_exit_flag(
                cfg,
                opening,
                t,
                exit_on,
                deal_status,
                bank_status,
                geo_cum,
                deal_failed_now,
                bank_declined_now,
            )
            just_exit = exit_on & ~prev_exit
            one_off = np.where(just_exit, exit_cost, 0.0)
            freeze_on = freeze_on | exit_on
        else:
            one_off = np.zeros(n, dtype=np.float64)

        burn = burn_for_day(cfg, policy_name, freeze_on, exit_on, macro_mult[:, t])
        drawn = drawn + bank_inflows[:, t]
        fin_cost = drawn * daily_rate
        inflow_fin = deal_inflows[:, t] + bank_inflows[:, t]
        financed = financed | (inflow_fin > 0.0)
        founder = float(cfg["company"].get("founder_external_income_daily", 0.0))

        c = (
            c
            + founder
            + revenue[:, t]
            + bank_inflows[:, t]
            + deal_inflows[:, t]
            + contract[:, t]
            - burn
            - shock_loss[:, t]
            - fin_cost
            - one_off
        )

        newly = (first_breach < 0) & (c < cmin)
        first_breach[newly] = t
        peak = np.maximum(peak, c)
        max_dd = np.maximum(max_dd, peak - c)

        cash[:, t] = c
        burn_path[:, t] = burn
        freeze_path[:, t] = freeze_on
        exit_path[:, t] = exit_on
        one_off_path[:, t] = one_off

    ext_in = deal_inflows + bank_inflows
    first_fin = _first_positive_day(ext_in)
    # Formal: financing arrives on day f, first breach on day b.
    # "Before breach" iff f >= 0 and (never breached or f <= b).
    never = first_breach < 0
    financed_before_breach = (first_fin >= 0) & (never | (first_fin <= first_breach))

    crisis_days = events.any(axis=2).sum(axis=1).astype(np.int64)
    max_sim = events.sum(axis=2).max(axis=1).astype(np.int64)
    return CashResult(
        cash=cash,
        burn=burn_path,
        freeze_on=freeze_path,
        exit_on=exit_path,
        first_breach_day=first_breach,
        min_cash=cash.min(axis=1),
        max_drawdown=max_dd,
        crisis_days=crisis_days,
        max_simultaneous=max_sim,
        freeze_ever=freeze_path.any(axis=1),
        exit_ever=exit_path.any(axis=1),
        financed_before_breach=financed_before_breach,
        one_off=one_off_path,
        first_financing_day=first_fin,
    )
