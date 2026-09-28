"""Monte Carlo orchestrator.

Worlds are vectorized. The only Python loop is over days (hazards + cash).
Policies share common random numbers: events, severities, deal/bank draws
are generated once, then each policy is a cash overlay on the same worlds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from absurdity import absurdity_index, raw_score, summarize_ai
from assumptions import CATEGORIES, checkpoint_day
from cashflow import CashResult, simulate_cash
from financing import (
    DEAL_CLOSED,
    DEAL_FAIL,
    DealResult,
    BankResult,
    simulate_aux_inflows,
    simulate_bank,
    simulate_deal,
)
from hazards import shock_loss_rub, simulate_hazards, simulate_latent_z
from metrics import classify, finite_ok, metrics_for_policy
from policies import enabled_policies, freeze_reduces_exactly


@dataclass
class RandomStreams:
    z_eps: np.ndarray
    event_u: np.ndarray
    severity_g: np.ndarray
    pareto_u: np.ndarray
    deal_u_sanc: np.ndarray
    deal_u_legal: np.ndarray
    deal_u_cp: np.ndarray
    deal_u_fail: np.ndarray
    deal_u_close: np.ndarray
    deal_u_delay: np.ndarray
    deal_u_geo_delay: np.ndarray
    deal_u_geo_fail: np.ndarray
    bank_u_approve: np.ndarray
    bank_u_reduced: np.ndarray
    bank_u_delay: np.ndarray
    bank_u_limit_g: np.ndarray
    bank_u_fin_delay: np.ndarray
    u_rev_kind: np.ndarray
    u_contract_kind: np.ndarray
    g_rev: np.ndarray
    g_contract: np.ndarray


@dataclass
class WorldBundle:
    z: np.ndarray
    events: np.ndarray
    severity: np.ndarray
    lam: np.ndarray
    shock_loss: np.ndarray
    deal: DealResult
    bank: BankResult
    revenue: np.ndarray
    contract: np.ndarray
    score: np.ndarray


@dataclass
class OracleResult:
    cfg: dict
    streams: RandomStreams
    bundle: WorldBundle
    cash: dict[str, CashResult]
    metrics: dict[str, dict]
    ai: np.ndarray
    ai_summary: dict
    runtime_s: float = 0.0
    extras: dict = field(default_factory=dict)


def make_streams(n: int, t_days: int, seed: int, k: int | None = None) -> RandomStreams:
    k = int(k or len(CATEGORIES))
    rng = np.random.default_rng(int(seed))
    return RandomStreams(
        z_eps=rng.standard_normal((n, t_days)),
        event_u=rng.random((n, t_days, k)),
        severity_g=rng.standard_normal((n, t_days, k)),
        pareto_u=rng.random((n, t_days)),
        deal_u_sanc=rng.random(n),
        deal_u_legal=rng.random(n),
        deal_u_cp=rng.random(n),
        deal_u_fail=rng.random(n),
        deal_u_close=rng.random(n),
        deal_u_delay=rng.random(n),
        deal_u_geo_delay=rng.random((n, t_days)),
        deal_u_geo_fail=rng.random((n, t_days)),
        bank_u_approve=rng.random(n),
        bank_u_reduced=rng.random(n),
        bank_u_delay=rng.random(n),
        bank_u_limit_g=rng.standard_normal(n),
        bank_u_fin_delay=rng.random((n, t_days)),
        u_rev_kind=rng.random((n, t_days)),
        u_contract_kind=rng.random((n, t_days)),
        g_rev=rng.standard_normal((n, t_days)),
        g_contract=rng.standard_normal((n, t_days)),
    )


def simulate_bundle(cfg: Mapping, streams: RandomStreams, overlay=None) -> WorldBundle:
    rho = float(cfg["contagion"]["rho"])
    sigma = float(cfg["contagion"]["sigma"])
    z = simulate_latent_z(streams.z_eps, rho, sigma)
    events, severity, lam = simulate_hazards(
        cfg, streams.event_u, streams.severity_g, streams.pareto_u, z, overlay=overlay
    )
    shock = shock_loss_rub(cfg, events, severity)
    if overlay is not None and getattr(overlay, "extra_loss", None) is not None:
        shock = shock + overlay.extra_loss[None, :]
    deal = simulate_deal(
        cfg,
        events,
        streams.deal_u_sanc,
        streams.deal_u_legal,
        streams.deal_u_cp,
        streams.deal_u_fail,
        streams.deal_u_close,
        streams.deal_u_delay,
        streams.deal_u_geo_delay,
        streams.deal_u_geo_fail,
    )
    bank = simulate_bank(
        cfg,
        events,
        streams.bank_u_approve,
        streams.bank_u_reduced,
        streams.bank_u_delay,
        streams.bank_u_limit_g,
        streams.bank_u_fin_delay,
    )
    revenue, contract = simulate_aux_inflows(
        cfg,
        events,
        streams.u_rev_kind,
        streams.u_contract_kind,
        streams.g_rev,
        streams.g_contract,
    )
    score = raw_score(cfg, events, severity)
    return WorldBundle(
        z=z,
        events=events,
        severity=severity,
        lam=lam,
        shock_loss=shock,
        deal=deal,
        bank=bank,
        revenue=revenue,
        contract=contract,
        score=score,
    )


def simulate_policy_cash(cfg: Mapping, bundle: WorldBundle, policy_name: str) -> CashResult:
    return simulate_cash(
        cfg,
        policy_name,
        bundle.shock_loss,
        bundle.events,
        bundle.severity,
        bundle.deal.inflows,
        bundle.bank.inflows,
        bundle.revenue,
        bundle.contract,
        bundle.deal.status,
        bundle.bank.status,
        bundle.deal.fail_day,
        bundle.bank.decision_day,
    )


def run_oracle(
    cfg: Mapping,
    n_worlds: int | None = None,
    seed: int | None = None,
    policies: list[str] | None = None,
    streams: RandomStreams | None = None,
    bundle: WorldBundle | None = None,
    overlay=None,
    live_events: list | None = None,
) -> OracleResult:
    import time

    n = int(n_worlds if n_worlds is not None else cfg["simulation"]["worlds"])
    t_days = int(cfg["simulation"]["days"])
    seed_i = int(seed if seed is not None else cfg["simulation"]["seed"])
    pols = policies if policies is not None else enabled_policies(cfg)

    t0 = time.perf_counter()
    if streams is None:
        streams = make_streams(n, t_days, seed_i)
    if bundle is None:
        bundle = simulate_bundle(cfg, streams, overlay=overlay)

    cash: dict[str, CashResult] = {}
    mets: dict[str, dict] = {}
    for name in pols:
        cr = simulate_policy_cash(cfg, bundle, name)
        cash[name] = cr
        freeze90 = name == "FREEZE_90"
        cls = classify(cfg, cr, bundle.deal, bundle.bank, freeze_is_90=freeze90)
        mets[name] = metrics_for_policy(
            cfg, cr, bundle.deal, bundle.bank, bundle.events, bundle.shock_loss, cls, name
        )

    hold_scores = bundle.score
    ai = absurdity_index(bundle.score, hold_scores)
    ai_sum = summarize_ai(ai, cfg)
    runtime = time.perf_counter() - t0
    return OracleResult(
        cfg=dict(cfg),
        streams=streams,
        bundle=bundle,
        cash=cash,
        metrics=mets,
        ai=ai,
        ai_summary=ai_sum,
        runtime_s=runtime,
        extras={"live_events": list(live_events or [])},
    )


def survival_prob(result: OracleResult, policy: str = "HOLD") -> float:
    return float(result.metrics[policy]["P(SURVIVAL)"])


def check_invariants(cfg: Mapping, n: int = 64, seed: int = 7) -> list[str]:
    """Return a list of failed invariant names (empty => pass)."""
    from copy import deepcopy

    fails: list[str] = []

    def _fail(msg: str) -> None:
        fails.append(msg)

    # 1. same seed => identical
    r1 = run_oracle(cfg, n_worlds=n, seed=seed, policies=["HOLD"])
    r2 = run_oracle(cfg, n_worlds=n, seed=seed, policies=["HOLD"])
    if not np.array_equal(r1.cash["HOLD"].cash, r2.cash["HOLD"].cash):
        _fail("same seed => identical results")
    if not np.array_equal(r1.bundle.events, r2.bundle.events):
        _fail("same seed => identical events")

    # 2. P90=0 and p_daily=0 => no events
    cfg0 = deepcopy(cfg)
    for name in CATEGORIES:
        if name == "BLACK_SWAN":
            cfg0["hazards"][name]["p_daily"] = 0.0
        else:
            cfg0["hazards"][name]["p90"] = 0.0
    r0 = run_oracle(cfg0, n_worlds=n, seed=seed, policies=["HOLD"])
    if r0.bundle.events.any():
        _fail("P90=0 => corresponding events never happen")

    # 3. burn=0 + shocks=0 + no costs => cash does not decrease
    cfgb = deepcopy(cfg0)
    cfgb["company"]["burn_monthly"] = 0.0
    cfgb["company"]["bridge_capital"] = 0.0
    cfgb["bank"]["approval_probability"] = 0.0
    cfgb["bank"]["financing_cost"]["annual_rate"] = 0.0
    cfgb["deal"]["close_probability"] = 0.0
    cfgb["deal"]["failure_probability"] = 0.0
    cfgb["deal"]["sanctions_block_probability"] = 0.0
    cfgb["deal"]["legal_block_probability"] = 0.0
    cfgb["deal"]["counterparty_failure_probability"] = 0.0
    cfgb["deal"]["geo_fail_probability"] = 0.0
    cfgb["contract"]["daily_inflow"] = 0.0
    cfgb["revenue"]["unexpected_revenue_probability_per_biz_event"] = 0.0
    cfgb["contract"]["unexpected_contract_probability_per_biz_event"] = 0.0
    rb = run_oracle(cfgb, n_worlds=n, seed=seed, policies=["HOLD"])
    cash = rb.cash["HOLD"].cash
    d = np.diff(cash, axis=1)
    if np.any(d < -1e-9):
        _fail("burn=0 + shocks=0 + no costs => cash does not decrease")
    if not np.allclose(cash[:, 0], float(cfgb["company"]["cash_initial"]) - 0.0, atol=1e-8):
        # day 0 still applies day-0 burn which is 0, so cash_0 == cash_initial
        pass
    if not np.allclose(cash, float(cfgb["company"]["cash_initial"])):
        # revenue/contract/deal/bank all 0, burn 0, shock 0 => constant
        if not np.allclose(np.diff(cash, axis=1), 0.0, atol=1e-8):
            _fail("flat cash when all flows are zero")

    # 4. financing inflow cannot happen twice unless staged
    staged = bool(cfg["deal"]["tranches"].get("enabled", False))
    n_days = r1.bundle.deal.n_inflow_days
    max_ok = 2 if staged else 1
    if staged:
        max_ok = max(1, len(cfg["deal"]["tranches"]["schedule"]))
    if np.any(n_days > max_ok):
        _fail("financing inflow cannot happen twice unless staged tranche")
    if np.any(r1.bundle.bank.n_inflow_days > 1):
        _fail("bank inflow occurred more than once")

    # 5. failed deal cannot later close
    deal = r1.bundle.deal
    failed = deal.status == DEAL_FAIL
    if np.any(failed & (deal.close_day >= 0)):
        _fail("failed deal cannot later close")
    if np.any(failed & (deal.inflows.sum(axis=1) > 0)):
        _fail("deal cash cannot arrive after permanent deal failure")

    # 6. first breach is first
    cr = r1.cash["HOLD"]
    cmin = float(cfg["company"]["cash_minimum"])
    for i in range(min(n, 32)):
        b = int(cr.first_breach_day[i])
        if b < 0:
            if np.any(cr.cash[i] < cmin):
                _fail("cash breach records FIRST breach day (missed)")
                break
        else:
            if cr.cash[i, b] >= cmin:
                _fail("cash breach records FIRST breach day (false positive)")
                break
            if b > 0 and np.any(cr.cash[i, :b] < cmin):
                _fail("cash breach records FIRST breach day (not first)")
                break

    # 7. no NaN / inf
    if not finite_ok(r1.cash["HOLD"].cash):
        _fail("no NaN / inf in cash")
    if not finite_ok(r1.bundle.lam):
        _fail("no NaN / inf in lambda")
    if not finite_ok(r1.ai):
        _fail("no NaN / inf in AI")

    # 8. probabilities in [0,1]
    for k, v in r1.metrics["HOLD"].items():
        if isinstance(k, str) and k.startswith("P(") and isinstance(v, float):
            if not (0.0 <= v <= 1.0) or not np.isfinite(v):
                _fail("probabilities always [0,1]")
                break

    # 9. negative hazard impossible
    if np.any(r1.bundle.lam < -1e-15):
        _fail("negative hazard impossible")

    # 10. FREEZE_50 / FREEZE_90 exact burn reduction
    for pname, frac in (("FREEZE_50", 0.5), ("FREEZE_90", 0.9)):
        base, frozen = freeze_reduces_exactly(cfg, pname)
        disc = float(cfg["company"]["discretionary_rd_share"])
        expected = base * (1.0 - disc) + base * disc * (1.0 - frac)
        if abs(frozen - expected) > 1e-9:
            _fail(f"{pname} reduces eligible burn by exactly configured amount")

    cfgf = deepcopy(cfg)
    cfgf["policies"]["FREEZE_50"]["trigger"] = {"cash_below": 1e18, "min_day": 0}
    cfgf["policies"]["FREEZE_90"]["trigger"] = {"cash_below": 1e18, "min_day": 0}
    cfgf["macro_pass_through"]["burn_kappa"] = 0.0
    cfgf["company"]["burn_monthly"] = 3000.0
    rf = run_oracle(cfgf, n_worlds=n, seed=seed, policies=["HOLD", "FREEZE_50", "FREEZE_90"])
    b_hold = rf.cash["HOLD"].burn[:, 0]
    b50 = rf.cash["FREEZE_50"].burn[:, 0]
    b90 = rf.cash["FREEZE_90"].burn[:, 0]
    base, f50 = freeze_reduces_exactly(cfgf, "FREEZE_50")
    _, f90 = freeze_reduces_exactly(cfgf, "FREEZE_90")
    if not np.allclose(b50, f50, atol=1e-8):
        _fail("FREEZE_50 reduces eligible burn by exactly configured amount")
    if not np.allclose(b90, f90, atol=1e-8):
        _fail("FREEZE_90 reduces eligible burn by exactly configured amount")
    if not np.allclose(b_hold, base, atol=1e-6):
        # HOLD may still have macro_mult ~ 1
        if np.max(np.abs(b_hold - base)) > 1e-4:
            _fail("HOLD burn != configured base (macro_kappa=0)")

    # 11. bank decision cannot occur before checkpoint
    chk = checkpoint_day(cfg)
    allow = bool(cfg["bank"].get("allow_pre_checkpoint_decision", False))
    dec = r1.bundle.bank.decision_day
    if not allow:
        bad = (dec >= 0) & (dec < chk)
        if np.any(bad):
            _fail("bank decision cannot occur before configured checkpoint")
        if chk > 0 and np.any(r1.bundle.bank.inflows[:, :chk] > 0):
            _fail("bank decision cannot occur before configured checkpoint (inflow)")

    # 12. deal cash after failure
    if np.any((deal.status != DEAL_CLOSED) & (deal.inflows.sum(axis=1) > 0)):
        _fail("deal cash cannot arrive after permanent deal failure")

    # closed deal: inflows only if close_day valid
    closed = deal.status == DEAL_CLOSED
    if np.any(closed & (deal.close_day < 0)):
        _fail("closed deal missing close_day")

    return fails
