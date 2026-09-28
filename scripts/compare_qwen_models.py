"""Compare Qwen sizes on the SAME analysis_payload.json.

Opt-in. Downloads weights. Run in Colab after Secrets HF_TOKEN (optional for public models).

    python scripts/compare_qwen_models.py --payload outputs/simulations/analysis_payload.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fidelity import check_numeric_fidelity
from hf_session import token_status
from qwen_analyst import generate_briefing

MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B-Instruct-2507",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--payload", default="outputs/simulations/analysis_payload.json")
    p.add_argument("--mode", default="maga")
    p.add_argument("--models", nargs="*", default=MODELS)
    args = p.parse_args()
    path = Path(args.payload)
    if not path.is_file():
        print("missing payload; run Monte Carlo first")
        return 2
    payload = json.loads(path.read_text(encoding="utf-8"))
    print("hf token:", token_status())
    print("payload:", path)
    rows = []
    for model_id in args.models:
        t0 = time.perf_counter()
        out = generate_briefing(
            payload,
            mode=args.mode,
            backend="qwen",
            model_id=model_id,
        )
        dt = time.perf_counter() - t0
        rail = out.get("fidelity") or check_numeric_fidelity(out["text"], payload)
        rec = {
            "model_id": model_id,
            "seconds": round(dt, 2),
            "fidelity_ok": rail.get("ok"),
            "issues": rail.get("issues"),
            "chars": len(out["text"]),
        }
        rows.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
        out_path = ROOT / "outputs" / "simulations" / f"briefing_{model_id.replace('/', '_')}.md"
        out_path.write_text(out["text"], encoding="utf-8")
        print("wrote", out_path)
    summary = ROOT / "outputs" / "simulations" / "qwen_model_compare.json"
    summary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("summary", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
