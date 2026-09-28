"""RF 28.09–28.11 pizdec heatmap.

    python scripts/run_pizdec_heatmap.py --worlds 1000 --analyst-backend facts
    python scripts/run_pizdec_heatmap.py --from-payload outputs/simulations/pizdec_heatmap_payload.json --analyst-backend qwen
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _qwen_only(payload_path: Path, backend: str, mode: str, model: str) -> int:
    from qwen_analyst import generate_briefing

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    out = generate_briefing(
        payload,
        mode=mode,
        backend=backend,
        model_id=model,
        max_new_tokens=1400,
    )
    qpath = ROOT / "outputs" / "PIZDEC_HEATMAP_QWEN.md"
    qpath.write_text(out["text"] + "\n", encoding="utf-8")
    (ROOT / "outputs" / "QWEN_BRIEFING.md").write_text(out["text"] + "\n", encoding="utf-8")
    print(f"wrote {qpath}")
    print(f"fidelity={'OK' if out['fidelity']['ok'] else 'FAIL'}  hf_token={out['hf_token']}")
    return 0 if out["fidelity"]["ok"] else 3


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", default="configs/rf_sep28_nov28.yaml")
    p.add_argument("--analyst-backend", default="facts")
    p.add_argument("--analyst-mode", default="maga")
    p.add_argument("--analyst-model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--from-payload", default=None)
    args = p.parse_args()
    if args.from_payload:
        return _qwen_only(
            Path(args.from_payload),
            args.analyst_backend,
            args.analyst_mode,
            args.analyst_model,
        )
    from run import main as run_main

    argv = [
        "--pizdec-heatmap",
        "--skip-tests",
        "--config",
        args.config,
        "--worlds",
        str(args.worlds),
        "--analyst-backend",
        args.analyst_backend,
        "--analyst-mode",
        args.analyst_mode,
        "--analyst-model",
        args.analyst_model,
    ]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    return run_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
