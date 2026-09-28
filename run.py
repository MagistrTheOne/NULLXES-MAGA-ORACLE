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

from assumptions import DISCLAIMER, load_config  # noqa: E402
from escape_solver import solve_escape  # noqa: E402
from export_excel import write_workbook  # noqa: E402
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
    p.add_argument("--worlds", type=int, default=None, help="Override simulation.worlds")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--ladder", action="store_true", help="1k then 10k then 100k + solver + xlsx")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--skip-solver", action="store_true")
    p.add_argument("--skip-excel", action="store_true")
    p.add_argument("--out", default="outputs/NULLXES_90D_ORACLE.xlsx")
    args = p.parse_args(argv)

    _print_banner()
    cfg = load_config(args.config)
    print(f"Config: {cfg.get('_config_path')}")
    print(f"Checkpoint day index: see bank.checkpoint_date={cfg['bank']['checkpoint_date']}")

    inv = check_invariants(cfg, n=64, seed=int(cfg["simulation"]["seed"]))
    if inv:
        print("INVARIANT FAILURES:")
        for msg in inv:
            print("  -", msg)
        return 2
    print("Invariants: PASS (n=64)")

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
            res = run_oracle(cfg, n_worlds=n, seed=args.seed)
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
        production = run_oracle(cfg, n_worlds=n, seed=args.seed)
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
    print("\n" + DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
