"""Opt-in Qwen briefing from existing analysis_payload.json.

Does not re-run Monte Carlo. Downloads weights. Intended for Colab.

    python scripts/run_qwen_briefing.py --payload outputs/simulations/analysis_payload.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from payload import write_json  # noqa: E402
from qwen_analyst import generate_briefing  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--payload", default="outputs/simulations/analysis_payload.json")
    p.add_argument("--delta", default=None)
    p.add_argument("--mode", default="maga")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--backend", default="qwen")
    args = p.parse_args()
    path = Path(args.payload)
    if not path.is_file():
        print("missing payload; run Monte Carlo first (facts backend)")
        return 2
    payload = json.loads(path.read_text(encoding="utf-8"))
    delta = None
    if args.delta:
        delta = json.loads(Path(args.delta).read_text(encoding="utf-8"))
    out = generate_briefing(
        payload,
        mode=args.mode,
        delta=delta,
        backend=args.backend,
        model_id=args.model,
    )
    brief_path = ROOT / "outputs" / "QWEN_BRIEFING.md"
    brief_path.write_text(out["text"] + "\n", encoding="utf-8")
    write_json(
        {
            "backend": out["backend"],
            "mode": out["mode"],
            "model_id": out["model_id"],
            "used_llm": out["used_llm"],
            "hf_token": out["hf_token"],
            "fidelity": out["fidelity"],
        },
        ROOT / "outputs" / "simulations" / "analyst_meta.json",
    )
    print(f"wrote {brief_path}")
    print(f"fidelity={'OK' if out['fidelity']['ok'] else 'FAIL'}  hf_token={out['hf_token']}")
    if not out["fidelity"]["ok"]:
        print("trust AUTHORITATIVE FACTS / Monte Carlo, not mismatched quotes")
        for issue in out["fidelity"].get("issues") or []:
            print(" ", issue)
    return 0 if out["fidelity"]["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
