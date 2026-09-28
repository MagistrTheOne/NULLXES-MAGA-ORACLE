# Live events

The model does **not** scrape the internet by default.

Ways to inject realtime-ish news:

1. Edit `events/live.jsonl` in this repo, push, `git pull` in Colab, re-run **same seed**.
2. Paste headlines into the Colab form (`LIVE_TEXT`).
3. Serve your own JSON/JSONL and pass `--events-url https://...`
4. Google Sheet → Apps Script → JSON endpoint → `--events-url`.

Schema (JSONL):

```json
{"ts":"2026-09-29","category":"GEO","mode":"observed","text":"...","p_boost":0.1,"lambda_add":0.2,"loss_rub":0,"config":{"deal.geo_fail_probability":0.15}}
```

`category` optional if `text` matches the lexicon (санкции→GEO, банк→FIN, blackout→SEC, …).

Re-runs use common random numbers: delta is the overlay, not new dice.
