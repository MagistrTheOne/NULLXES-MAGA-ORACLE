"""Deal/bank invariants: checkpoint, no double inflow, fail cannot close."""

from __future__ import annotations

import numpy as np

from assumptions import checkpoint_day, load_config
from financing import DEAL_CLOSED, DEAL_FAIL
from simulation import run_oracle


def test_bank_not_before_checkpoint():
    cfg = load_config("baseline.yaml")
    assert cfg["bank"]["allow_pre_checkpoint_decision"] is False
    chk = checkpoint_day(cfg)
    assert chk == 28  # 2026-09-28 -> 2026-10-26
    r = run_oracle(cfg, n_worlds=128, seed=8, policies=["HOLD"])
    dec = r.bundle.bank.decision_day
    bad = (dec >= 0) & (dec < chk)
    assert not np.any(bad)
    if chk > 0:
        assert not np.any(r.bundle.bank.inflows[:, :chk] > 0)


def test_failed_deal_has_no_cash_and_no_close_day():
    cfg = load_config("baseline.yaml")
    cfg["deal"]["failure_probability"] = 1.0
    cfg["deal"]["close_probability"] = 1.0
    cfg["deal"]["sanctions_block_probability"] = 0.0
    cfg["deal"]["legal_block_probability"] = 0.0
    cfg["deal"]["counterparty_failure_probability"] = 0.0
    cfg["deal"]["geo_fail_probability"] = 0.0
    r = run_oracle(cfg, n_worlds=64, seed=12, policies=["HOLD"])
    deal = r.bundle.deal
    assert np.all(deal.status == DEAL_FAIL)
    assert np.all(deal.close_day < 0)
    assert np.all(deal.inflows == 0.0)


def test_closed_deal_cannot_fail_later_without_recovery():
    cfg = load_config("baseline.yaml")
    assert cfg["deal"]["allow_recovery_after_failure"] is False
    r = run_oracle(cfg, n_worlds=200, seed=13, policies=["HOLD"])
    deal = r.bundle.deal
    failed = deal.status == DEAL_FAIL
    closed = deal.status == DEAL_CLOSED
    assert not np.any(failed & closed)
    assert not np.any(failed & (deal.inflows.sum(axis=1) > 0))
    assert not np.any((~closed) & (deal.inflows.sum(axis=1) > 0))


def test_bank_inflow_at_most_once():
    cfg = load_config("baseline.yaml")
    r = run_oracle(cfg, n_worlds=128, seed=14, policies=["HOLD"])
    assert np.all(r.bundle.bank.n_inflow_days <= 1)


def test_staged_tranche_cap():
    cfg = load_config("baseline.yaml")
    n_tr = len(cfg["deal"]["tranches"]["schedule"])
    r = run_oracle(cfg, n_worlds=128, seed=15, policies=["HOLD"])
    assert np.all(r.bundle.deal.n_inflow_days <= n_tr)


def test_approval_not_automatic_after_checkpoint():
    cfg = load_config("baseline.yaml")
    r = run_oracle(cfg, n_worlds=256, seed=16, policies=["HOLD"])
    # With p=0.22 some worlds should decline or delay
    assert r.metrics["HOLD"]["P(BANK APPROVAL)"] < 0.999
    assert r.metrics["HOLD"]["P(BANK APPROVAL)"] >= 0.0
