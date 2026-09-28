"""Cash0 sweep: same CRN worlds, only cash_initial moves."""

from __future__ import annotations

from assumptions import load_config
from cash_sweep import (
    identify_cliffs,
    run_cash_sweep,
    sweep_presenter,
    write_sweep_outputs,
)
from fidelity import check_numeric_fidelity
from qwen_analyst import generate_briefing
from simulation import run_oracle


def test_hold_survival_monotone_in_cash0():
    cfg = load_config("baseline.yaml")
    sweep = run_cash_sweep(
        cfg,
        n_worlds=32,
        seed=11,
        scenarios=(
            ("CASH_500", 500.0),
            ("CASH_200K", 200_000.0),
            ("CASH_5M", 5_000_000.0),
        ),
    )
    s = [r["P(SURVIVAL)"] for r in sweep["rows"]]
    assert s[0] <= s[1] + 1e-12
    assert s[1] <= s[2] + 1e-12
    assert sweep["crn_invariants"]["deal_close_identical_across_cash"] is True
    assert sweep["crn_invariants"]["bank_approval_identical_across_cash"] is True
    assert sweep["crn_invariants"]["absurdity_identical_across_cash"] is True
    closes = {r["P(DEAL CLOSE)"] for r in sweep["rows"]}
    assert len(closes) == 1


def test_same_seed_same_deal_as_singleton():
    cfg = load_config("baseline.yaml")
    r = run_oracle(cfg, n_worlds=24, seed=9, policies=["HOLD"])
    sweep = run_cash_sweep(
        cfg,
        n_worlds=24,
        seed=9,
        scenarios=(("CASH_500", 500.0),),
    )
    assert abs(sweep["rows"][0]["P(DEAL CLOSE)"] - r.metrics["HOLD"]["P(DEAL CLOSE)"]) < 1e-12
    assert abs(sweep["rows"][0]["P(SURVIVAL)"] - r.metrics["HOLD"]["P(SURVIVAL)"]) < 1e-12


def test_cliffs_first_grid_point():
    rows = [
        {"name": "A", "cash_initial": 500, "P(SURVIVAL)": 0.0, "P(CASH BREACH)": 1.0,
         "p_breach_before_checkpoint": 1.0, "arithmetic_runway_days": 0.07,
         "p_survival_given_no_material_financing": 0.0},
        {"name": "B", "cash_initial": 200000, "P(SURVIVAL)": 0.08, "P(CASH BREACH)": 0.92,
         "p_breach_before_checkpoint": 0.4, "arithmetic_runway_days": 30.0,
         "p_survival_given_no_material_financing": 0.02},
        {"name": "C", "cash_initial": 1000000, "P(SURVIVAL)": 0.7, "P(CASH BREACH)": 0.3,
         "p_breach_before_checkpoint": 0.1, "arithmetic_runway_days": 150.0,
         "p_survival_given_no_material_financing": 0.55},
    ]
    cliffs = identify_cliffs(rows, checkpoint=28, daily=200000 / 30)
    assert cliffs["survival_material_ge_5pct"]["name"] == "B"
    assert cliffs["breach_probability_below_50pct"]["name"] == "C"
    assert cliffs["financing_optional_surv_no_material_ge_50pct"]["name"] == "C"
    assert cliffs["arithmetic_runway_covers_checkpoint"]["name"] == "B"


def test_facts_backend_sweep_fidelity(tmp_path):
    cfg = load_config("baseline.yaml")
    sweep = run_cash_sweep(
        cfg,
        n_worlds=16,
        seed=4,
        scenarios=(("CASH_500", 500.0), ("CASH_1M", 1_000_000.0)),
    )
    payload = write_sweep_outputs(sweep, tmp_path, mode="maga")["payload"]
    out = generate_briefing(payload, backend="facts", mode="maga")
    assert out["used_llm"] is False
    assert out["fidelity"]["ok"] is True
    assert "CASH_500" in out["text"]
    rail = check_numeric_fidelity("CASH_500 P(SURVIVAL): 0.99", payload)
    assert rail["ok"] is False
