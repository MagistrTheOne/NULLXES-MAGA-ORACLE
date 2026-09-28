"""Deal and bank financing state machines (vectorized over worlds).

Deal outcomes (precedence, permanent unless allow_recovery_after_failure):
    SANCTIONS_BLOCK > LEGAL_BLOCK > COUNTERPARTY_FAILURE > DEAL_FAIL
    > DEAL_CLOSE > DEAL_DELAY (still pending at horizon)

Bank outcomes (decision cannot occur before checkpoint unless enabled):
    BANK_APPROVE | BANK_REDUCED_LIMIT | BANK_DECLINE | BANK_DELAY

Invariants enforced in code:
- no deal cash after a permanent failure/block
- no second bank/deal principal inflow unless staged tranches / allow_repeat_draw
- bank decision day >= checkpoint_day
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from assumptions import CATEGORIES, checkpoint_day

DEAL_PENDING = 0
DEAL_CLOSED = 1
DEAL_FAIL = 2
DEAL_LEGAL_BLOCK = 3
DEAL_SANCTIONS_BLOCK = 4
DEAL_COUNTERPARTY = 5
DEAL_DELAY = 6

DEAL_NAME = {
    DEAL_PENDING: "DEAL_PENDING",
    DEAL_CLOSED: "DEAL_CLOSE",
    DEAL_FAIL: "DEAL_FAIL",
    DEAL_LEGAL_BLOCK: "LEGAL_BLOCK",
    DEAL_SANCTIONS_BLOCK: "SANCTIONS_BLOCK",
    DEAL_COUNTERPARTY: "COUNTERPARTY_FAILURE",
    DEAL_DELAY: "DEAL_DELAY",
}

BANK_NONE = 0
BANK_APPROVE = 1
BANK_REDUCED_LIMIT = 2
BANK_DECLINE = 3
BANK_DELAY = 4

BANK_NAME = {
    BANK_NONE: "BANK_NONE",
    BANK_APPROVE: "BANK_APPROVE",
    BANK_REDUCED_LIMIT: "BANK_REDUCED_LIMIT",
    BANK_DECLINE: "BANK_DECLINE",
    BANK_DELAY: "BANK_DELAY",
}


def _first_true(mask: np.ndarray, default: int = -1) -> np.ndarray:
    any_t = mask.any(axis=1)
    idx = np.argmax(mask, axis=1)
    return np.where(any_t, idx, default)


def draw_uniform_int(u: np.ndarray, lo: int, hi: int) -> np.ndarray:
    lo_i = int(lo)
    hi_i = int(hi)
    if hi_i < lo_i:
        lo_i, hi_i = hi_i, lo_i
    span = hi_i - lo_i + 1
    return lo_i + np.floor(np.clip(u, 0.0, 1.0 - 1e-12) * span).astype(np.int64)


def draw_lognormal(u_gauss: np.ndarray, median: float, sigma: float) -> np.ndarray:
    mu = np.log(max(float(median), 1e-18))
    return np.exp(mu + float(sigma) * u_gauss)


@dataclass
class DealResult:
    status: np.ndarray          # (N,) int
    close_day: np.ndarray       # (N,) int, -1 if never
    fail_day: np.ndarray        # (N,) int, -1 if never
    inflows: np.ndarray         # (N, T)
    n_inflow_days: np.ndarray   # (N,) count of days with deal cash


@dataclass
class BankResult:
    status: np.ndarray          # (N,) int
    decision_day: np.ndarray    # (N,) int, -1 if none
    limit: np.ndarray           # (N,)
    inflows: np.ndarray         # (N, T)
    n_inflow_days: np.ndarray


def simulate_deal(
    cfg: Mapping,
    events: np.ndarray,
    u_sanc: np.ndarray,
    u_legal: np.ndarray,
    u_cp: np.ndarray,
    u_fail: np.ndarray,
    u_close: np.ndarray,
    u_delay: np.ndarray,
    u_geo_delay: np.ndarray,
    u_geo_fail: np.ndarray,
) -> DealResult:
    n, t_days, _k = events.shape
    geo = events[:, :, CATEGORIES.index("GEO")]
    dcfg = cfg["deal"]
    allow_recovery = bool(dcfg.get("allow_recovery_after_failure", False))

    sanc = u_sanc < float(dcfg["sanctions_block_probability"])
    legal = (~sanc) & (u_legal < float(dcfg["legal_block_probability"]))
    cp = (~sanc) & (~legal) & (u_cp < float(dcfg["counterparty_failure_probability"]))
    blocked = sanc | legal | cp

    p_fail = float(dcfg["failure_probability"])
    p_close = float(dcfg["close_probability"])
    fail0 = (~blocked) & (u_fail < p_fail)
    close0 = (~blocked) & (~fail0) & (u_close < p_close)
    # residual unblocked worlds stay pending → DELAY at horizon if still open

    delay0 = draw_uniform_int(
        u_delay,
        int(dcfg["delay_distribution"]["min_days"]),
        int(dcfg["delay_distribution"]["max_days"]),
    )

    p_geo_delay = float(dcfg["geo_delay_probability"])
    extra_each = int(dcfg["geo_extra_delay_days"])
    p_geo_fail = float(dcfg["geo_fail_probability"])

    status = np.full(n, DEAL_PENDING, dtype=np.int64)
    status[sanc] = DEAL_SANCTIONS_BLOCK
    status[legal] = DEAL_LEGAL_BLOCK
    status[cp] = DEAL_COUNTERPARTY
    status[fail0] = DEAL_FAIL

    close_day = np.full(n, -1, dtype=np.int64)
    fail_day = np.full(n, -1, dtype=np.int64)
    fail_day[fail0] = 0

    pending_close = close0.copy()
    target = delay0.copy()

    extra_hits = (geo & (u_geo_delay < p_geo_delay)).astype(np.int64)
    extra_total = extra_hits.sum(axis=1) * extra_each
    target = target + extra_total

    geo_fail_hit = geo & (u_geo_fail < p_geo_fail)
    first_geo_fail = _first_true(geo_fail_hit, default=-1)
    # GEO fail can kill a still-pending close path on/before scheduled close.
    geo_kills = pending_close & (first_geo_fail >= 0) & (first_geo_fail <= np.clip(target, 0, t_days))
    if not allow_recovery:
        status[geo_kills] = DEAL_FAIL
        fail_day[geo_kills] = first_geo_fail[geo_kills]
        pending_close[geo_kills] = False

    still = pending_close & (status == DEAL_PENDING)
    cd = np.clip(target, 0, 10**9)
    in_horizon = still & (cd < t_days)
    close_day[in_horizon] = cd[in_horizon]
    status[in_horizon] = DEAL_CLOSED
    delayed = still & (cd >= t_days)
    status[delayed] = DEAL_DELAY

    # remaining PENDING at end of assignment: unblocked, not fail0, not close0
    leftover = status == DEAL_PENDING
    status[leftover] = DEAL_DELAY

    inflows = np.zeros((n, t_days), dtype=np.float64)
    capital = float(dcfg["target_capital"])
    closed = status == DEAL_CLOSED
    tranches = dcfg.get("tranches") or {}
    staged = bool(tranches.get("enabled", False)) and bool(tranches.get("schedule"))

    if staged:
        for tr in tranches["schedule"]:
            off = int(tr["offset_days"])
            frac = float(tr["fraction"])
            day = close_day + off
            ok = closed & (close_day >= 0) & (day >= 0) & (day < t_days)
            idx = np.flatnonzero(ok)
            if idx.size:
                inflows[idx, day[idx]] += capital * frac
    else:
        ok = closed & (close_day >= 0) & (close_day < t_days)
        idx = np.flatnonzero(ok)
        if idx.size:
            inflows[idx, close_day[idx]] += capital

    # Hard invariant: no cash if not CLOSED
    not_closed = ~closed
    inflows[not_closed, :] = 0.0

    n_inflow_days = (inflows > 0.0).sum(axis=1).astype(np.int64)
    return DealResult(
        status=status,
        close_day=close_day,
        fail_day=fail_day,
        inflows=inflows,
        n_inflow_days=n_inflow_days,
    )


def simulate_bank(
    cfg: Mapping,
    events: np.ndarray,
    u_approve: np.ndarray,
    u_reduced: np.ndarray,
    u_delay: np.ndarray,
    u_limit_g: np.ndarray,
    u_fin_delay: np.ndarray,
) -> BankResult:
    n, t_days, _k = events.shape
    fin = events[:, :, CATEGORIES.index("FIN")]
    bcfg = cfg["bank"]
    chk = int(checkpoint_day(cfg))
    allow_pre = bool(bcfg.get("allow_pre_checkpoint_decision", False))
    if not allow_pre:
        chk = max(chk, 0)

    delay = draw_uniform_int(
        u_delay,
        int(bcfg["decision_delay_distribution"]["min_days"]),
        int(bcfg["decision_delay_distribution"]["max_days"]),
    )
    decision = chk + delay

    p_extra = float(bcfg["fin_event_delay_probability"])
    extra_days = int(bcfg["fin_event_extra_delay_days"])
    # FIN events on days strictly before the tentative decision add delay.
    # Use a conservative bound: count FIN events before max possible day.
    for t in range(t_days):
        pre = (t < decision) & (t >= (0 if allow_pre else chk))
        hit = pre & fin[:, t] & (u_fin_delay[:, t] < p_extra)
        decision = decision + extra_days * hit.astype(np.int64)

    if not allow_pre:
        decision = np.maximum(decision, chk)

    p0 = float(bcfg["approval_probability"])
    decay = float(bcfg["fin_stress_approval_decay"])
    # count FIN events before decision
    t_idx = np.arange(t_days)[None, :]
    pre_mask = t_idx < decision[:, None]
    if not allow_pre:
        pre_mask &= t_idx >= chk
    n_fin = (fin & pre_mask).sum(axis=1).astype(np.float64)
    p_eff = p0 * np.exp(-decay * n_fin)
    p_eff = np.clip(p_eff, 0.0, 1.0)

    in_horizon = (decision >= 0) & (decision < t_days)
    approved = in_horizon & (u_approve < p_eff)
    declined = in_horizon & ~approved
    delayed = ~in_horizon

    limit = draw_lognormal(
        u_limit_g,
        float(bcfg["limit_distribution"]["median"]),
        float(bcfg["limit_distribution"]["sigma"]),
    )
    reduced = approved & (u_reduced < float(bcfg["reduced_limit_probability"]))
    # FIN stress can force a reduced limit on otherwise full approvals
    reduced = reduced | (approved & (n_fin >= 2.0) & (u_reduced < 0.5 + 0.5 * float(bcfg["reduced_limit_probability"])))
    limit_paid = np.where(reduced, limit * float(bcfg["reduced_limit_factor"]), limit)
    limit_paid = np.where(approved, limit_paid, 0.0)

    status = np.full(n, BANK_NONE, dtype=np.int64)
    status[delayed] = BANK_DELAY
    status[declined] = BANK_DECLINE
    status[approved & ~reduced] = BANK_APPROVE
    status[reduced] = BANK_REDUCED_LIMIT

    inflows = np.zeros((n, t_days), dtype=np.float64)
    ok = approved & in_horizon
    idx = np.flatnonzero(ok)
    if idx.size:
        inflows[idx, decision[idx]] += limit_paid[idx]

    if not bool(bcfg.get("allow_repeat_draw", False)):
        # already a single credit by construction
        pass

    # Hard invariant: no bank cash before checkpoint
    if not allow_pre and chk > 0:
        inflows[:, :chk] = 0.0

    decision_out = np.where(in_horizon, decision, -1)
    n_inflow_days = (inflows > 0.0).sum(axis=1).astype(np.int64)
    return BankResult(
        status=status,
        decision_day=decision_out,
        limit=limit_paid,
        inflows=inflows,
        n_inflow_days=n_inflow_days,
    )


def simulate_aux_inflows(
    cfg: Mapping,
    events: np.ndarray,
    u_rev_kind: np.ndarray,
    u_contract_kind: np.ndarray,
    g_rev: np.ndarray,
    g_contract: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unexpected BIZ revenue/contract + configured daily contract inflow.

    Returns revenue (N,T), contract (N,T).
    """
    n, t_days, _k = events.shape
    biz = events[:, :, CATEGORIES.index("BIZ")]
    sec = events[:, :, CATEGORIES.index("SEC")]
    geo = events[:, :, CATEGORIES.index("GEO")]

    rcfg = cfg["revenue"]
    ccfg = cfg["contract"]
    p_rev = float(rcfg["unexpected_revenue_probability_per_biz_event"])
    p_con = float(ccfg["unexpected_contract_probability_per_biz_event"])

    rev = np.zeros((n, t_days), dtype=np.float64)
    con = np.zeros((n, t_days), dtype=np.float64)

    rev_hit = biz & (u_rev_kind < p_rev)
    con_hit = biz & (u_contract_kind < p_con)
    rev[rev_hit] = draw_lognormal(
        g_rev[rev_hit],
        float(rcfg["unexpected_revenue_median"]),
        float(rcfg["unexpected_revenue_sigma"]),
    )
    con[con_hit] = draw_lognormal(
        g_contract[con_hit],
        float(ccfg["unexpected_contract_median"]),
        float(ccfg["unexpected_contract_sigma"]),
    )

    daily = float(ccfg.get("daily_inflow", 0.0))
    if daily != 0.0:
        sched = np.full((n, t_days), daily, dtype=np.float64)
        if bool(ccfg.get("interrupted_by_sec", True)):
            sched[sec] = 0.0
        if bool(ccfg.get("interrupted_by_geo", True)):
            sched[geo] = 0.0
        con = con + sched
    return rev, con
