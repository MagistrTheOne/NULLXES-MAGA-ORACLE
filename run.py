"""NULLXES 90D Monte Carlo runner.

Designed for Google Colab CPU. GPU is not used.

Examples (Colab):
    !python run.py --ladder
    !python run.py --worlds 1000 --skip-solver
    !python run.py --config configs/stress.yaml --worlds 10000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assumptions import DISCLAIMER, deep_merge, load_config, load_yaml, set_path  # noqa: E402
from briefing import build_briefing, generate_situation_charts, write_briefing_markdown  # noqa: E402
from escape_solver import solve_escape  # noqa: E402
from export_excel import write_workbook  # noqa: E402
from live_events import (  # noqa: E402
    apply_event_config_overrides,
    collect_events,
    overlay_from_events,
    payload_hash,
)
from payload import build_analysis_payload, write_json  # noqa: E402
from qwen_analyst import generate_briefing  # noqa: E402
from sensitivity import run_sensitivity, tornado_table  # noqa: E402
from simulation import check_invariants, run_oracle  # noqa: E402


def _print_banner() -> None:
    print("=" * 72)
    print("NULLXES 90D ORACLE")
    print(DISCLAIMER)
    print("Backend: NumPy CPU. GPU is not used and is not required.")
    print("=" * 72)


def _run_pytest() -> int:
    try:
        import pytest
    except ImportError:
        print("pytest not installed — running in-process invariants only.")
        return 0
    return pytest.main(["-q", str(ROOT / "tests")])


def _coerce(raw: str):
    s = str(raw).strip()
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _apply_sets(cfg: dict, items: list[str]) -> dict:
    out = cfg
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, val = item.split("=", 1)
        out = set_path(out, key.strip(), _coerce(val))
    return out


def _print_metrics(tag: str, result) -> None:
    m = result.metrics["HOLD"]
    print(f"\n[{tag}] worlds={m['n_worlds']} runtime={result.runtime_s:.3f}s")
    keys = [
        "P(SURVIVAL)",
        "P(CASH BREACH)",
        "P(FINANCING BEFORE BREACH)",
        "P(DEAL CLOSE)",
        "P(DEAL FAIL)",
        "P(BANK APPROVAL)",
        "P(BANK DECLINE)",
        "P(BANK DELAY)",
        "P(ANY MAJOR SHOCK <= 90 days)",
        "c90_median",
        "c90_p5",
        "c90_p95",
        "es5",
        "median_min_cash",
        "median_max_drawdown",
        "mean_crisis_days",
    ]
    for k in keys:
        v = m.get(k)
        if isinstance(v, float):
            if k.startswith("P("):
                print(f"  {k:40s} {v:8.4%}")
            else:
                print(f"  {k:40s} {v:,.2f}")
        else:
            print(f"  {k:40s} {v}")
    print("  class shares:")
    for k, v in m.items():
        if k.startswith("P(") and k.endswith(")") and k.split("(")[1].rstrip(")") in {
            "FAIL",
            "RU_EXIT",
            "ESCAPE",
            "FREEZE",
            "TRAP",
            "SURVIVE",
        }:
            print(f"    {k:38s} {v:8.4%}")
    print(
        f"  AI median-max={result.ai_summary['median_max_ai']:.1f} "
        f"({result.ai_summary['median_max_band']})"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="NULLXES 90-day Monte Carlo oracle")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--live-input", default=None, help="YAML overlay with editable run inputs")
    p.add_argument("--set", dest="sets", action="append", default=[], help="dotted.key=value (repeatable)")
    p.add_argument("--worlds", type=int, default=None, help="Override simulation.worlds")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--burn", type=float, default=None, help="Override company.burn_monthly")
    p.add_argument("--bridge", type=float, default=None, help="Override company.bridge_capital")
    p.add_argument("--p-bank", type=float, default=None, help="Override bank.approval_probability")
    p.add_argument("--p-deal", type=float, default=None, help="Override deal.close_probability")
    p.add_argument("--cmin", type=float, default=None, help="Override company.cash_minimum")
    p.add_argument("--events", default=None, help="Path to JSONL/YAML/JSON/CSV live events")
    p.add_argument("--events-url", default=None, help="HTTP JSON/JSONL event feed")
    p.add_argument("--events-text", default=None, help="Pasted headlines, one per line")
    p.add_argument("--ladder", action="store_true", help="1k then 10k then 100k + solver + xlsx")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--skip-solver", action="store_true")
    p.add_argument("--skip-excel", action="store_true")
    p.add_argument("--policy-view", default="HOLD", help="Primary policy in analysis_payload.json")
    p.add_argument("--analyst-backend", default="facts", help="facts | transformers | openai")
    p.add_argument("--analyst-mode", default="maga", help="board | maga")
    p.add_argument("--analyst-model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--delta-json", default=None, help="Optional other analysis_payload.json for delta")
    p.add_argument("--out", default="outputs/NULLXES_90D_ORACLE.xlsx")
    args = p.parse_args(argv)

    _print_banner()
    cfg = load_config(args.config)
    print(f"Config: {cfg.get('_config_path')}")
    print(f"Checkpoint: {cfg['bank']['checkpoint_date']}")

    inv = check_invariants(cfg, n=64, seed=int(cfg["simulation"]["seed"]))
    if inv:
        print("INVARIANT FAILURES:")
        for msg in inv:
            print("  -", msg)
        return 2
    print("Invariants: PASS (n=64)")

    if args.live_input:
        overlay_cfg = load_yaml(ROOT / args.live_input if not Path(args.live_input).is_absolute() else Path(args.live_input))
        # live_input may `extends` or be a pure overlay
        if overlay_cfg.get("extends"):
            cfg = load_config(args.live_input)
        else:
            cfg = deep_merge(cfg, overlay_cfg)
        print(f"Live input merged: {args.live_input}")
    cfg = _apply_sets(cfg, args.sets)
    if args.burn is not None:
        cfg = set_path(cfg, "company.burn_monthly", float(args.burn))
    if args.bridge is not None:
        cfg = set_path(cfg, "company.bridge_capital", float(args.bridge))
    if args.p_bank is not None:
        cfg = set_path(cfg, "bank.approval_probability", float(args.p_bank))
    if args.p_deal is not None:
        cfg = set_path(cfg, "deal.close_probability", float(args.p_deal))
    if args.cmin is not None:
        cfg = set_path(cfg, "company.cash_minimum", float(args.cmin))
    if args.worlds is not None:
        cfg = set_path(cfg, "simulation.worlds", int(args.worlds))
    if args.seed is not None:
        cfg = set_path(cfg, "simulation.seed", int(args.seed))

    live_events = collect_events(
        cfg,
        file=args.events,
        url=args.events_url,
        text=args.events_text,
    )
    if live_events:
        cfg = apply_event_config_overrides(cfg, live_events)
        print(f"Live events: {len(live_events)}  hash={payload_hash(live_events)}")
        for ev in live_events:
            print(f"  day={ev.day:3d}  {ev.category:11s}  {ev.mode:9s}  {ev.text[:80]}")
    overlay = overlay_from_events(live_events, int(cfg["simulation"]["days"])) if live_events else None


    if not args.skip_tests:
        rc = _run_pytest()
        if rc not in (0, 5):
            print(f"pytest failed with code {rc}")
            return rc
        print("pytest: PASS")

    ladder_rows: list[dict] = []
    production = None

    if args.ladder:
        for n in (1000, 10000, 100000):
            t0 = time.perf_counter()
            res = run_oracle(
                cfg,
                n_worlds=n,
                seed=args.seed,
                overlay=overlay,
                live_events=live_events,
            )
            dt = time.perf_counter() - t0
            _print_metrics(f"{n:,} worlds", res)
            ladder_rows.append(
                {
                    "worlds": n,
                    "runtime_s": round(dt, 3),
                    "P(SURVIVAL)_HOLD": res.metrics["HOLD"]["P(SURVIVAL)"],
                    "P(CASH BREACH)_HOLD": res.metrics["HOLD"]["P(CASH BREACH)"],
                    "c90_median": res.metrics["HOLD"]["c90_median"],
                    "es5": res.metrics["HOLD"]["es5"],
                    "backend": "numpy_cpu",
                    "gpu_used": False,
                }
            )
            production = res
            # Drop huge arrays from previous scale before next
            if n != 100000:
                del res
    else:
        n = int(args.worlds if args.worlds is not None else cfg["simulation"]["worlds"])
        t0 = time.perf_counter()
        production = run_oracle(
            cfg,
            n_worlds=n,
            seed=args.seed,
            overlay=overlay,
            live_events=live_events,
        )
        dt = time.perf_counter() - t0
        _print_metrics(f"{n:,} worlds", production)
        ladder_rows.append(
            {
                "worlds": n,
                "runtime_s": round(dt, 3),
                "P(SURVIVAL)_HOLD": production.metrics["HOLD"]["P(SURVIVAL)"],
                "P(CASH BREACH)_HOLD": production.metrics["HOLD"]["P(CASH BREACH)"],
                "c90_median": production.metrics["HOLD"]["c90_median"],
                "es5": production.metrics["HOLD"]["es5"],
                "backend": "numpy_cpu",
                "gpu_used": False,
            }
        )

    assert production is not None
    brief = build_briefing(production, None)
    print("\n--- SITUATION BRIEFING ---")
    print(brief["text"])
    print("--------------------------")
    generate_situation_charts(
        production, None, ROOT / "outputs" / "charts", brief
    )
    write_briefing_markdown(brief, ROOT / "outputs" / "PIZDEC_BRIEFING.md")

    print("\nPolicy comparison (same worlds):")
    for name, m in production.metrics.items():
        print(
            f"  {name:16s}  P(SURVIVAL)={m['P(SURVIVAL)']:.4%}  "
            f"P(BREACH)={m['P(CASH BREACH)']:.4%}  C90med={m['c90_median']:,.0f}"
        )

    sens_rows: list[dict] = []
    escape: dict = {}
    if not args.skip_solver:
        print("\nSensitivity (CRN)...")
        t0 = time.perf_counter()
        sens_rows = run_sensitivity(cfg, production)
        print(f"  sensitivity runtime {time.perf_counter() - t0:.1f}s")
        torn = tornado_table(sens_rows)
        print("  top |delta| factors:")
        for r in torn[:8]:
            print(
                f"    {r['parameter']:28s}  side={r['side']:4s}  "
                f"delta={r['delta_survival']:+.4%}  ({r['tag']})"
            )
        print("\nEscape solver (CRN)...")
        t0 = time.perf_counter()
        escape = solve_escape(cfg, production, sens_rows)
        print(f"  escape runtime {time.perf_counter() - t0:.1f}s")
        print(f"  best policy by P(SURVIVAL): {escape['best_policy_by_survival']}")
        print(f"  min bridge: {escape['min_bridge_for_target']}")
        print(f"  max burn: {escape['max_burn_for_target']}")
        print(f"  escape set: {escape['minimal_escape_set']}")
    else:
        from escape_solver import policy_ranking

        escape = {
            "target_survival": float(cfg["escape"]["target_survival"]),
            "best_policy_by_survival": policy_ranking(production)[0]["policy"],
            "policy_ranking": policy_ranking(production),
            "min_bridge_for_target": {},
            "max_burn_for_target": {},
            "financing_timing_curve": [],
            "strongest_sensitivity": None,
            "bank_vs_immediate_freeze": {},
            "min_contract_inflow_material": {},
            "minimal_escape_set": {},
        }

    out_path = ROOT / args.out
    if not args.skip_excel:
        print(f"\nWriting {out_path} ...")
        write_workbook(
            out_path,
            production,
            sens_rows,
            escape,
            ladder=ladder_rows,
            invariant_failures=inv,
            chart_dir=ROOT / "outputs" / "charts",
        )
        print(f"Workbook: {out_path}")

    summary_path = ROOT / "outputs" / "simulations" / "last_run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "disclaimer": DISCLAIMER,
        "ladder": ladder_rows,
        "hold_metrics": {
            k: v
            for k, v in production.metrics["HOLD"].items()
            if isinstance(v, (int, float, str, bool))
        },
        "policies": {
            name: {
                k: v
                for k, v in m.items()
                if isinstance(v, (int, float, str, bool))
            }
            for name, m in production.metrics.items()
        },
        "sensitivity_top": tornado_table(sens_rows)[:10] if sens_rows else [],
        "escape_best_policy": escape.get("best_policy_by_survival"),
        "escape_chosen": (escape.get("minimal_escape_set") or {}).get("chosen"),
        "gpu_used": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Summary JSON: {summary_path}")

    payload = build_analysis_payload(
        cfg,
        production,
        escape,
        sens_rows,
        policy=args.policy_view,
        scenario=(cfg.get("meta") or {}).get("name"),
    )
    payload_path = ROOT / "outputs" / "simulations" / "analysis_payload.json"
    write_json(payload, payload_path)
    print(f"Analyst payload: {payload_path}")

    delta = None
    if args.delta_json:
        from payload import build_delta_payload

        other = json.loads(Path(args.delta_json).read_text(encoding="utf-8"))
        delta = build_delta_payload(other, payload)
        write_json(delta, ROOT / "outputs" / "simulations" / "delta_payload.json")
        print("Delta payload written (other → current)")

    analyst = generate_briefing(
        payload,
        mode=args.analyst_mode,
        delta=delta,
        backend=args.analyst_backend,
        model_id=args.analyst_model,
    )
    brief_path = ROOT / "outputs" / "QWEN_BRIEFING.md"
    brief_path.write_text(analyst["text"] + "\n", encoding="utf-8")
    print(f"\n--- ANALYST ({analyst['backend']} / {analyst['mode']}) ---")
    print(analyst["text"])
    print(f"Wrote {brief_path}")
    print("\n" + DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
