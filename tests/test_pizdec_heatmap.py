"""RF window pizdec heatmap: horizon, weeks, FACTS rail."""

from __future__ import annotations

import datetime as dt

from assumptions import load_config
from fidelity import check_numeric_fidelity
from pizdec_heatmap import (
    WINDOW_END,
    WINDOW_START,
    horizon_days,
    run_pizdec_heatmap,
    write_pizdec_outputs,
)
from qwen_analyst import generate_briefing


def test_horizon_includes_nov28():
    assert horizon_days() == 62
    assert WINDOW_START.isoformat() == "2026-09-28"
    assert WINDOW_END.isoformat() == "2026-11-28"


def test_rf_config_days():
    cfg = load_config("rf_sep28_nov28.yaml")
    assert int(cfg["simulation"]["days"]) == 62
    assert cfg["simulation"]["start_date"] == "2026-09-28"
    start = dt.date.fromisoformat(cfg["simulation"]["start_date"])
    last = start + dt.timedelta(days=int(cfg["simulation"]["days"]) - 1)
    assert last == WINDOW_END


def test_pizdec_weeks_and_facts(tmp_path):
    cfg = load_config("rf_sep28_nov28.yaml")
    cal = run_pizdec_heatmap(cfg, n_worlds=24, seed=11)
    assert cal["window"]["days"] == 62
    assert cal["window"]["end"] == "2026-11-28"
    assert cal["weeks"]
    assert cal["weeks"][0]["id"] == "W0"
    assert cal["hottest_week"]["id"] in {w["id"] for w in cal["weeks"]}
    assert "FIN" in cal["weeks"][0]["P(EVENT)"]
    payload = write_pizdec_outputs(cal, tmp_path)["payload"]
    out = generate_briefing(payload, backend="facts", mode="maga")
    assert out["used_llm"] is False
    assert out["fidelity"]["ok"] is True
    assert "HOTTEST_WEEK" in out["text"]
    assert "W0" in out["text"]
    bad = check_numeric_fidelity("HOLD P(SURVIVAL): 0.99", payload)
    assert bad["ok"] is False
