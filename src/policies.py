"""NULLXES policy engine.

Policies are cash/burn overlays on a shared stochastic world (common random numbers).

HOLD          — no discretionary cut.
FREEZE_50     — cut eligible R&D burn by 50% after trigger (sticky).
FREEZE_90     — cut eligible R&D burn by 90% after trigger (sticky).
CAPITAL_FIRST — cut 100% of discretionary burn until any material financing arrives.
EXIT_RU       — on trigger, pay one-off cost and switch to post-exit burn factor.

Burn identity (enforced, tested):
    base = burn_monthly / days_per_month
    core = base * (1 - discretionary_rd_share)
    disc = base * discretionary_rd_share
    burn = (core + disc * (1 - freeze_fraction * active)) * macro_mult
           * (post_exit_burn_factor if exited else 1)
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from assumptions import POLICY_NAMES, daily_burn as base_daily_burn


def policy_spec(cfg: Mapping, name: str) -> dict:
    if name not in cfg["policies"]:
        raise KeyError(name)
    return cfg["policies"][name]


def freeze_fraction(cfg: Mapping, name: str) -> float:
    return float(policy_spec(cfg, name).get("freeze_discretionary_fraction", 0.0))


def update_freeze_flag(
    cfg: Mapping,
    name: str,
    opening_cash: np.ndarray,
    t: int,
    already: np.ndarray,
    financed: np.ndarray,
    daily_burn_ref: float,
) -> np.ndarray:
    spec = policy_spec(cfg, name)
    if name == "HOLD":
        return already
    if name == "CAPITAL_FIRST" and bool(spec.get("until_financing", False)):
        min_day = int(spec.get("trigger", {}).get("min_day", 0))
        # Not sticky after financing: freeze only while still unfunded.
        return (~financed) & (t >= min_day)

    trig = spec.get("trigger") or {}
    min_day = int(trig.get("min_day", 0))
    if t < min_day:
        return already
    hit = np.zeros(opening_cash.shape, dtype=np.bool_)
    if "cash_below" in trig:
        hit |= opening_cash < float(trig["cash_below"])
    if "runway_days_below" in trig:
        hit |= (opening_cash / max(daily_burn_ref, 1e-12)) < float(trig["runway_days_below"])
    if "or_after_day" in trig and t >= int(trig["or_after_day"]):
        hit |= True
    return already | hit


def update_exit_flag(
    cfg: Mapping,
    opening_cash: np.ndarray,
    t: int,
    already: np.ndarray,
    deal_status: np.ndarray,
    bank_status: np.ndarray,
    geo_events_to_date: np.ndarray,
    deal_failed_now: np.ndarray,
    bank_declined_now: np.ndarray,
) -> np.ndarray:
    from financing import (
        BANK_DECLINE,
        DEAL_FAIL,
        DEAL_LEGAL_BLOCK,
        DEAL_SANCTIONS_BLOCK,
    )

    spec = policy_spec(cfg, "EXIT_RU")
    trig = spec.get("trigger") or {}
    min_day = int(trig.get("min_day", 0))
    if t < min_day:
        return already
    hit = np.zeros(opening_cash.shape, dtype=np.bool_)
    if bool(trig.get("on_sanctions_block", False)):
        hit |= deal_status == DEAL_SANCTIONS_BLOCK
    if bool(trig.get("on_legal_block", False)):
        hit |= deal_status == DEAL_LEGAL_BLOCK
    if bool(trig.get("on_deal_fail_and_bank_decline", False)):
        hit |= (deal_status == DEAL_FAIL) & (bank_status == BANK_DECLINE)
        hit |= deal_failed_now & bank_declined_now
    if "cash_below" in trig:
        hit |= opening_cash < float(trig["cash_below"])
    if "min_geo_events" in trig:
        hit |= geo_events_to_date >= int(trig["min_geo_events"])
    return already | hit


def burn_for_day(
    cfg: Mapping,
    name: str,
    freeze_on: np.ndarray,
    exit_on: np.ndarray,
    macro_mult: np.ndarray,
) -> np.ndarray:
    spec = policy_spec(cfg, name)
    base = base_daily_burn(cfg)
    disc_share = float(cfg["company"]["discretionary_rd_share"])
    disc_share = float(np.clip(disc_share, 0.0, 1.0))
    core = base * (1.0 - disc_share)
    disc = base * disc_share
    frac = float(spec.get("freeze_discretionary_fraction", 0.0))
    disc_eff = np.where(freeze_on, disc * (1.0 - frac), disc)
    burn = (core + disc_eff) * macro_mult
    if name == "EXIT_RU":
        post = float(spec.get("post_exit_burn_factor", 1.0))
        burn = np.where(exit_on, burn * post, burn)
    return burn


def freeze_reduces_exactly(cfg: Mapping, name: str) -> tuple[float, float]:
    """Return (base_daily_burn, frozen_daily_burn) with macro_mult=1, freeze on, no exit."""
    base = base_daily_burn(cfg)
    disc_share = float(np.clip(cfg["company"]["discretionary_rd_share"], 0.0, 1.0))
    frac = freeze_fraction(cfg, name)
    core = base * (1.0 - disc_share)
    disc = base * disc_share
    frozen = core + disc * (1.0 - frac)
    return float(base), float(frozen)


def enabled_policies(cfg: Mapping) -> list[str]:
    out = []
    for name in POLICY_NAMES:
        spec = cfg.get("policies", {}).get(name, {})
        if spec.get("enabled", True):
            out.append(name)
    return out
