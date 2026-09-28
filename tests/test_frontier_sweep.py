"""Cash0 × Burn frontier: CRN, monotone cash/burn, fidelity."""

from __future__ import annotations

from assumptions import load_config
from fidelity import check_numeric_fidelity
from frontier_sweep import identify_frontier, run_frontier_sweep, write_frontier_outputs
from qwen_analyst import generate_briefing


def test_hold_monotone_cash_up_burn_down():
    cfg = load_config("baseline.yaml")
    sweep = run_frontier_sweep(
        cfg,
        n_worlds=24,
        seed=8,
        cash_grid=(("200K", 200_000.0), ("2M", 2_000_000.0)),
        burn_grid=(("200K", 200_000.0), ("1M", 1_000_000.0)),
        solve_bridge=False,
    )
    by = {(c["cash_label"], c["burn_label"]): c for c in sweep["cells"]}
    lo = by[("200K", "200K")]["HOLD_P(SURVIVAL)"]
    hi = by[("2M", "200K")]["HOLD_P(SURVIVAL)"]
    assert lo <= hi + 1e-12
    hot = by[("2M", "1M")]["HOLD_P(SURVIVAL)"]
    cool = by[("2M", "200K")]["HOLD_P(SURVIVAL)"]
    assert hot <= cool + 1e-12
    assert sweep["crn_invariants"]["deal_close_identical"] is True
    assert sweep["crn_invariants"]["bank_approval_identical"] is True


def test_frontier_first_grid_point():
    cells = [
        {"id": "A", "cash_label": "L", "burn_label": "B", "cash_initial": 1, "burn_monthly": 9,
         "best_P(SURVIVAL)": 0.01, "HOLD_P(SURVIVAL)": 0.0},
        {"id": "C", "cash_label": "H", "burn_label": "B", "cash_initial": 9, "burn_monthly": 9,
         "best_P(SURVIVAL)": 0.6, "HOLD_P(SURVIVAL)": 0.55},
    ]
    fr = identify_frontier(cells, ["L", "H"], ["B"])
    row = fr["per_burn"][0]
    assert row["dead_zone"] is False
    assert row["min_cash_best_surv_ge_50"]["cell"] == "C"
    assert row["min_cash_hold_surv_ge_50"]["cell"] == "C"


def test_frontier_facts_fidelity(tmp_path):
    cfg = load_config("baseline.yaml")
    sweep = run_frontier_sweep(
        cfg,
        n_worlds=16,
        seed=5,
        cash_grid=(("200K", 200_000.0),),
        burn_grid=(("200K", 200_000.0),),
        solve_bridge=True,
        bridge_steps=8,
    )
    payload = write_frontier_outputs(sweep, tmp_path)["payload"]
    out = generate_briefing(payload, backend="facts", mode="maga")
    assert out["used_llm"] is False
    assert out["fidelity"]["ok"] is True
    assert "C200K_B200K" in out["text"]
    bad = check_numeric_fidelity("C200K_B200K HOLD P(SURVIVAL): 0.99 BEST P(SURVIVAL): 0.99", payload)
    assert bad["ok"] is False
