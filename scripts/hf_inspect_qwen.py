"""Hub metadata only. Does not download weights."""
from huggingface_hub import HfApi

api = HfApi()
ids = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-1.7B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]
print("=== model_info ===")
for i in ids:
    try:
        m = api.model_info(i)
        print(
            i,
            "downloads=", getattr(m, "downloads", None),
            "pipeline=", m.pipeline_tag,
            "gated=", getattr(m, "gated", False),
            "license=", (m.cardData or {}).get("license") if isinstance(m.cardData, dict) else None,
        )
    except Exception as e:
        print(i, "ERR", type(e).__name__, str(e)[:120])
