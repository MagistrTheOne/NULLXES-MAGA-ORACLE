"""Reproducibility and the full invariant suite."""

from __future__ import annotations

import numpy as np

from assumptions import load_config
from simulation import check_invariants, run_oracle


def test_same_seed_identical():
    cfg = load_config("baseline.yaml")
    a = run_oracle(cfg, n_worlds=40, seed=20260928, policies=["HOLD", "FREEZE_90"])
    b = run_oracle(cfg, n_worlds=40, seed=20260928, policies=["HOLD", "FREEZE_90"])
    assert np.array_equal(a.cash["HOLD"].cash, b.cash["HOLD"].cash)
    assert np.array_equal(a.cash["FREEZE_90"].cash, b.cash["FREEZE_90"].cash)
    assert np.array_equal(a.bundle.events, b.bundle.events)
    assert np.allclose(a.ai, b.ai)


def test_different_seed_changes_draws():
    cfg = load_config("baseline.yaml")
    a = run_oracle(cfg, n_worlds=40, seed=1, policies=["HOLD"])
    b = run_oracle(cfg, n_worlds=40, seed=2, policies=["HOLD"])
    assert not np.array_equal(a.bundle.events, b.bundle.events) or not np.allclose(
        a.cash["HOLD"].cash, b.cash["HOLD"].cash
    )


def test_common_random_numbers_policy_delta_uses_same_events():
    cfg = load_config("baseline.yaml")
    r = run_oracle(cfg, n_worlds=40, seed=21, policies=["HOLD", "FREEZE_50", "FREEZE_90"])
    # Events are generated once and shared
    assert r.cash["HOLD"].cash.shape == r.cash["FREEZE_90"].cash.shape
    # With positive discretionary share, freeze cannot increase burn
    assert np.all(r.cash["FREEZE_90"].burn <= r.cash["HOLD"].burn + 1e-9)
    assert np.all(r.cash["FREEZE_50"].burn <= r.cash["HOLD"].burn + 1e-9)


def test_all_invariants_pass():
    cfg = load_config("baseline.yaml")
    fails = check_invariants(cfg, n=48, seed=7)
    assert fails == [], fails


def test_probabilities_in_unit_interval():
    cfg = load_config("baseline.yaml")
    r = run_oracle(cfg, n_worlds=80, seed=22)
    for name, m in r.metrics.items():
        for k, v in m.items():
            if isinstance(k, str) and k.startswith("P(") and isinstance(v, float):
                assert 0.0 <= v <= 1.0, (name, k, v)
                assert np.isfinite(v)
