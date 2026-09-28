"""Live event parser and overlay."""

from __future__ import annotations

from assumptions import load_config, parse_date
from live_events import (
    classify_text,
    collect_events,
    overlay_from_events,
    parse_free_text_line,
    parse_jsonl_text,
)


def test_lexicon_geo():
    assert classify_text("санкционный комплаенс остановил SWIFT платёж") == "GEO"
    assert classify_text("датацентр downtime без электричества") == "SEC"


def test_free_text_date_and_category():
    cfg = load_config("baseline.yaml")
    start = parse_date(cfg["simulation"]["start_date"])
    ev = parse_free_text_line(
        "2026-09-29 GEO санкции блок платежа", start, 90
    )
    assert ev is not None
    assert ev.category == "GEO"
    assert ev.day == 1  # start is 2026-09-28


def test_jsonl_observed_forces_event():
    cfg = load_config("baseline.yaml")
    start = parse_date(cfg["simulation"]["start_date"])
    text = '{"ts":"2026-09-28","category":"FIN","mode":"observed","text":"bank stress","lambda_add":0.4}\n'
    events = parse_jsonl_text(text, start, 90)
    ov = overlay_from_events(events, 90)
    assert ov.force_event[0, 1]  # FIN index 1
    assert ov.lambda_add[0, 1] > 0


def test_collect_from_repo_file_empty_comments():
    cfg = load_config("baseline.yaml")
    cfg = {**cfg, "live": {"events_file": "events/live.jsonl"}}
    events = collect_events(cfg)
    assert events == []
