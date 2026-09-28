"""Live / realtime event ingestion.

This is NOT a news classifier with known accuracy.
Mapping headline → category is an ASSUMPTION / PLACEHOLDER lexicon.

Supported sources:
  - JSONL / JSON / YAML / CSV files
  - pasted free-text (one event per line)
  - HTTP(S) JSON or JSONL URL
  - Colab form dict

Event record (any extra keys ignored):
  ts / date / day     calendar date or integer simulation day
  category            SEC|FIN|MACRO|GEO|BIZ|BLACK_SWAN  (optional if text parses)
  text / headline     free text
  mode                observed | risk_up | config
      observed  — this already happened: force the category event that day
      risk_up   — raise remaining-day intensity (lambda add / p boost)
      config    — only apply config overrides
  p_boost             added to daily event probability that day, clip [0,1]
  lambda_add          added to hazard that day
  loss_rub            extra deterministic cash loss that day (all worlds)
  config              {dotted.key: value} overrides merged into cfg

Realtime idea that is actually implemented:
  1. You (or a bot) append lines to events/live.jsonl and re-run, same seed.
  2. Colab cell pastes today's headlines; parser maps them; overlay is applied
     on the SAME random streams (CRN) so delta is the news, not new dice.
  3. Optional URL poll: fetch JSON, if hash changed, rebuild overlay and rerun.
News APIs are not scraped by default. Point --events-url at your own JSON.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.request import Request, urlopen

import numpy as np
import yaml

from assumptions import CATEGORIES, ROOT, parse_date, set_path

LEXICON: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "GEO",
        (
            "санкц", "sanction", "ofac", "swift", "комплаенс", "compliance",
            "тамож", "export control", "legal block", "юрид", "embargo",
            "платеж не проход", "blocked payment",
        ),
    ),
    (
        "FIN",
        (
            "банк", "bank ", "кредит", "лимит", "цб ", "ставк", "ликвидн",
            "отказ в кред", "funding", "interest rate", "cbdc",
        ),
    ),
    (
        "SEC",
        (
            "blackout", "энерг", "электри", "интернет", "даунтайм", "downtime",
            "датацентр", "связь", "outage", "power", "dc failure",
        ),
    ),
    (
        "MACRO",
        (
            "инфляц", "топлив", "нефть", "логист", "курс", "fx ", "oil",
            "fuel", "cpi", "логистическ",
        ),
    ),
    (
        "BIZ",
        (
            "сделк", "контракт", "gpu", "kaira", "compute", "импорт",
            "заказчик", "revenue", "клиент",
        ),
    ),
    (
        "BLACK_SWAN",
        (
            "black swan", "катастроф", "обвал", "коллапс", "tail event",
        ),
    ),
)


@dataclass
class LiveEvent:
    day: int
    category: str
    mode: str
    text: str
    p_boost: float = 0.0
    lambda_add: float = 0.0
    loss_rub: float = 0.0
    config_overrides: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class HazardOverlay:
    lambda_add: np.ndarray
    p_boost: np.ndarray
    force_event: np.ndarray
    z_add: np.ndarray
    extra_loss: np.ndarray
    events: list[LiveEvent] = field(default_factory=list)

    @classmethod
    def zeros(cls, t_days: int, k: int | None = None) -> "HazardOverlay":
        k = int(k or len(CATEGORIES))
        return cls(
            lambda_add=np.zeros((t_days, k), dtype=np.float64),
            p_boost=np.zeros((t_days, k), dtype=np.float64),
            force_event=np.zeros((t_days, k), dtype=np.bool_),
            z_add=np.zeros(t_days, dtype=np.float64),
            extra_loss=np.zeros(t_days, dtype=np.float64),
        )


def classify_text(text: str) -> str | None:
    blob = (text or "").lower()
    if not blob.strip():
        return None
    scores: dict[str, int] = {}
    for cat, keys in LEXICON:
        scores[cat] = sum(1 for k in keys if k in blob)
    best = max(scores, key=scores.get)
    if scores[best] <= 0:
        return None
    return best


def _as_date(value: Any) -> dt.date | None:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return dt.datetime.strptime(s[:10], fmt).date()
        except ValueError:
            continue
    try:
        return parse_date(s[:10])
    except ValueError:
        return None


def _day_index(rec: Mapping[str, Any], start: dt.date, t_days: int) -> int | None:
    if rec.get("day") is not None and rec.get("day") != "":
        d = int(rec["day"])
        return d if 0 <= d < t_days else None
    date = _as_date(rec.get("ts") or rec.get("date") or rec.get("timestamp"))
    if date is None:
        return 0
    idx = (date - start).days
    if idx < 0:
        return 0
    if idx >= t_days:
        return None
    return idx


def _normalize_record(rec: Mapping[str, Any], start: dt.date, t_days: int) -> LiveEvent | None:
    text = str(rec.get("text") or rec.get("headline") or rec.get("title") or "")
    cat = str(rec.get("category") or rec.get("cat") or "").upper().replace(" ", "_")
    if cat not in CATEGORIES:
        guessed = classify_text(text)
        if guessed is None:
            return None
        cat = guessed
    day = _day_index(rec, start, t_days)
    if day is None:
        return None
    mode = str(rec.get("mode") or "observed").lower().strip()
    if mode not in {"observed", "risk_up", "config"}:
        mode = "observed"
    cfg_over = rec.get("config") or rec.get("overrides") or {}
    if not isinstance(cfg_over, dict):
        cfg_over = {}
    return LiveEvent(
        day=int(day),
        category=cat,
        mode=mode,
        text=text,
        p_boost=float(rec.get("p_boost") or 0.0),
        lambda_add=float(rec.get("lambda_add") or rec.get("lambda") or 0.0),
        loss_rub=float(rec.get("loss_rub") or rec.get("loss") or 0.0),
        config_overrides={str(k): v for k, v in cfg_over.items()},
        raw=dict(rec),
    )


def parse_free_text_line(line: str, start: dt.date, t_days: int) -> LiveEvent | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # "2026-09-29 GEO text..." or "GEO text" or "text"
    rec: dict[str, Any] = {"text": line}
    m = re.match(
        r"^(?P<date>\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4})?\s*"
        r"(?P<cat>SEC|FIN|MACRO|GEO|BIZ|BLACK_SWAN)?\s*"
        r"(?P<text>.*)$",
        line,
        flags=re.I,
    )
    if m:
        if m.group("date"):
            rec["ts"] = m.group("date")
        if m.group("cat"):
            rec["category"] = m.group("cat").upper()
        rec["text"] = m.group("text") or line
    return _normalize_record(rec, start, t_days)


def parse_payload(payload: Any, start: dt.date, t_days: int) -> list[LiveEvent]:
    rows: list[Mapping[str, Any]] = []
    if payload is None:
        return []
    if isinstance(payload, list):
        rows = [r for r in payload if isinstance(r, Mapping)]
    elif isinstance(payload, Mapping):
        if "events" in payload and isinstance(payload["events"], list):
            rows = [r for r in payload["events"] if isinstance(r, Mapping)]
        else:
            rows = [payload]
    out: list[LiveEvent] = []
    for rec in rows:
        ev = _normalize_record(rec, start, t_days)
        if ev is not None:
            out.append(ev)
    return out


def parse_jsonl_text(text: str, start: dt.date, t_days: int) -> list[LiveEvent]:
    events: list[LiveEvent] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                ev = parse_free_text_line(line, start, t_days)
                if ev:
                    events.append(ev)
                continue
            ev = _normalize_record(rec, start, t_days)
            if ev:
                events.append(ev)
        else:
            ev = parse_free_text_line(line, start, t_days)
            if ev:
                events.append(ev)
    return events


def load_events_file(path: str | Path, start: dt.date, t_days: int) -> list[LiveEvent]:
    path = Path(path)
    if not path.is_absolute():
        cand = ROOT / path
        path = cand if cand.exists() else path
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return parse_payload(yaml.safe_load(raw), start, t_days)
    if suffix == ".json":
        return parse_payload(json.loads(raw), start, t_days)
    if suffix == ".csv":
        rows = list(csv.DictReader(io.StringIO(raw)))
        return parse_payload(rows, start, t_days)
    return parse_jsonl_text(raw, start, t_days)


def fetch_events_url(url: str, start: dt.date, t_days: int, timeout: int = 20) -> list[LiveEvent]:
    req = Request(url, headers={"User-Agent": "NULLXES-MAGA-ORACLE/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        blob = resp.read().decode("utf-8", errors="replace")
    s = blob.lstrip()
    if s.startswith("[") or s.startswith("{"):
        try:
            return parse_payload(json.loads(blob), start, t_days)
        except json.JSONDecodeError:
            pass
    return parse_jsonl_text(blob, start, t_days)


def overlay_from_events(events: Iterable[LiveEvent], t_days: int) -> HazardOverlay:
    ov = HazardOverlay.zeros(t_days)
    ov.events = list(events)
    for ev in ov.events:
        k = CATEGORIES.index(ev.category)
        t = ev.day
        ov.extra_loss[t] += ev.loss_rub
        if ev.mode == "config":
            continue
        if ev.mode == "observed":
            ov.force_event[t, k] = True
            ov.p_boost[t, k] += max(ev.p_boost, 0.0)
            ov.lambda_add[t, k] += max(ev.lambda_add, 0.35)
            ov.z_add[t] += 0.25
        elif ev.mode == "risk_up":
            # remaining days including t
            ov.lambda_add[t:, k] += max(ev.lambda_add, 0.05)
            ov.p_boost[t:, k] += max(ev.p_boost, 0.02)
            ov.z_add[t:] += 0.05
    return ov


def apply_event_config_overrides(cfg: dict, events: Iterable[LiveEvent]) -> dict:
    out = cfg
    for ev in events:
        for key, val in ev.config_overrides.items():
            out = set_path(out, key, val)
    return out


def collect_events(
    cfg: Mapping[str, Any],
    file: str | Path | None = None,
    url: str | None = None,
    text: str | None = None,
    records: list[Mapping[str, Any]] | None = None,
) -> list[LiveEvent]:
    start = parse_date(cfg["simulation"]["start_date"])
    t_days = int(cfg["simulation"]["days"])
    events: list[LiveEvent] = []
    live_cfg = cfg.get("live") or {}
    file = file or live_cfg.get("events_file")
    url = url or live_cfg.get("events_url")
    text = text if text is not None else live_cfg.get("events_text")
    if file:
        events.extend(load_events_file(file, start, t_days))
    if url:
        events.extend(fetch_events_url(str(url), start, t_days))
    if text:
        events.extend(parse_jsonl_text(str(text), start, t_days))
    if records:
        events.extend(parse_payload(list(records), start, t_days))
    return events


def payload_hash(events: Iterable[LiveEvent]) -> str:
    blob = json.dumps(
        [
            {
                "day": e.day,
                "category": e.category,
                "mode": e.mode,
                "text": e.text,
                "p_boost": e.p_boost,
                "lambda_add": e.lambda_add,
                "loss_rub": e.loss_rub,
                "config": e.config_overrides,
            }
            for e in events
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
