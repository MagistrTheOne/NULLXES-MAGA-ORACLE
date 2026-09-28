"""Cash identity, freeze reduction, first-breach recording."""

from __future__ import annotations

import numpy as np

from assumptions import load_config
from policies import freeze_reduces_exactly
from simulation import run_oracle


def test_freeze_50_and_90_exact_reduction():
    cfg = load_config("baseline.yaml")
    base, f50 = freeze_reduces_exactly(cfg, "FREEZE_50")
    _, f90 = freeze_reduces_exactly(cfg, "FREEZE_90")
    disc = float(cfg["company"]["discretionary_rd_share"])
    assert abs(f50 - (base * (1 - disc) + base * disc * 0.5)) < 1e-12
    assert abs(f90 - (base * (1 - disc) + base * disc * 0.1)) < 1e-12


def test_forced_freeze_matches_identity_on_day0():
    cfg = load_config("baseline.yaml")
    cfg["policies"]["FREEZE_50"]["trigger"] = {"cash_below": 1e18, "min_day": 0}
    cfg["policies"]["FREEZE_90"]["trigger"] = {"cash_below": 1e18, "min_day": 0}
    cfg["macro_pass_through"]["burn_kappa"] = 0.0
    cfg["company"]["burn_monthly"] = 3000.0
    r = run_oracle(cfg, n_worlds=32, seed=11, policies=["HOLD", "FREEZE_50", "FREEZE_90"])
    base, f50 = freeze_reduces_exactly(cfg, "FREEZE_50")
    _, f90 = freeze_reduces_exactly(cfg, "FREEZE_90")
    assert np.allclose(r.cash["FREEZE_50"].burn[:, 0], f50)
    assert np.allclose(r.cash["FREEZE_90"].burn[:, 0], f90)
    assert np.allclose(r.cash["HOLD"].burn[:, 0], base, atol=1e-8)


def test_zero_flows_cash_constant():
    cfg = load_config("baseline.yaml")
    for name in cfg["hazards"]:
        if name == "BLACK_SWAN":
            cfg["hazards"][name]["p_daily"] = 0.0
        else:
            cfg["hazards"][name]["p90"] = 0.0
    cfg["company"]["burn_monthly"] = 0.0
    cfg["bank"]["approval_probability"] = 0.0
    cfg["bank"]["financing_cost"]["annual_rate"] = 0.0
    cfg["deal"]["close_probability"] = 0.0
    cfg["deal"]["failure_probability"] = 0.0
    cfg["deal"]["sanctions_block_probability"] = 0.0
    cfg["deal"]["legal_block_probability"] = 0.0
    cfg["deal"]["counterparty_failure_probability"] = 0.0
    cfg["deal"]["geo_fail_probability"] = 0.0
    cfg["contract"]["daily_inflow"] = 0.0
    cfg["revenue"]["unexpected_revenue_probability_per_biz_event"] = 0.0
    cfg["contract"]["unexpected_contract_probability_per_biz_event"] = 0.0
    r = run_oracle(cfg, n_worlds=16, seed=3, policies=["HOLD"])
    cash = r.cash["HOLD"].cash
    assert np.allclose(cash, float(cfg["company"]["cash_initial"]))
    assert not np.any(np.diff(cash, axis=1) < -1e-12)


def test_first_breach_is_first():
    cfg = load_config("baseline.yaml")
    r = run_oracle(cfg, n_worlds=48, seed=5, policies=["HOLD"])
    cash = r.cash["HOLD"].cash
    fb = r.cash["HOLD"].first_breach_day
    cmin = float(cfg["company"]["cash_minimum"])
    for i in range(cash.shape[0]):
        b = int(fb[i])
        if b < 0:
            assert np.all(cash[i] >= cmin)
        else:
            assert cash[i, b] < cmin
            if b > 0:
                assert np.all(cash[i, :b] >= cmin)


def test_no_nan_inf_cash():
    cfg = load_config("baseline.yaml")
    r = run_oracle(cfg, n_worlds=24, seed=9, policies=["HOLD"])
    assert np.isfinite(r.cash["HOLD"].cash).all()
    assert np.isfinite(r.bundle.shock_loss).all()
