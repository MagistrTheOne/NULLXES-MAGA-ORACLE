"""RF-updated 2026-09-28 → 2026-11-28 pizdec heatmap.

Monte Carlo is the only quantitative source. Public CBR/Minfin prints are
inputs to ASSUMPTION knobs + live overlay, not a forecast of the country.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from assumptions import CATEGORIES, DISCLAIMER, checkpoint_day, parse_date
from escape_solver import policy_ranking
from live_events import apply_event_config_overrides, collect_events, overlay_from_events
from payload import write_json
from simulation import run_oracle

WINDOW_START = dt.date(2026, 9, 28)
WINDOW_END = dt.date(2026, 11, 28)
_AI_HEAT_THRESHOLDS = (40.0, 60.0, 80.0, 95.0)


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def horizon_days(start: dt.date = WINDOW_START, end: dt.date = WINDOW_END) -> int:
    return (end - start).days + 1


def dates_for(cfg: Mapping) -> list[dt.date]:
    start = parse_date(cfg["simulation"]["start_date"])
    n = int(cfg["simulation"]["days"])
    return [start + dt.timedelta(days=i) for i in range(n)]


def week_slices(dates: list[dt.date]) -> list[dict]:
    rows = []
    i = 0
    w = 0
    while i < len(dates):
        chunk = dates[i : i + 7]
        rows.append(
            {
                "id": f"W{w}",
                "lo": chunk[0].isoformat(),
                "hi": chunk[-1].isoformat(),
                "start_day": i,
                "end_day": i + len(chunk),
                "n_days": len(chunk),
            }
        )
        i += 7
        w += 1
    return rows


def _band_fractions(ai_week: np.ndarray, cfg: Mapping) -> dict[str, float]:
    n = float(ai_week.size) if ai_week.size else 1.0
    out = {}
    for b in cfg["absurdity"]["bands"]:
        lo, hi = float(b["lo"]), float(b["hi"])
        out[str(b["label"])] = float(np.mean((ai_week >= lo) & (ai_week < hi)))
        _ = n
    return out


def build_calendar(result, live_events: list | None = None) -> dict:
    cfg = result.cfg
    dates = dates_for(cfg)
    events = result.bundle.events
    ai = result.ai
    p_event = events.mean(axis=0)  # (T, K)
    daily = {
        "p50": [float(x) for x in result.ai_summary["daily_p50"]],
        "p95": [float(x) for x in result.ai_summary["daily_p95"]],
        "mean": [float(x) for x in result.ai_summary["daily_mean"]],
    }
    daily_p_ai = {
        f"ge_{int(th)}": [float(x) for x in (ai >= th).mean(axis=0)]
        for th in _AI_HEAT_THRESHOLDS
    }
    weeks_meta = week_slices(dates)
    weeks = []
    for w in weeks_meta:
        sl = slice(w["start_day"], w["end_day"])
        p_cat = {
            cat: float(p_event[sl, k].mean())
            for k, cat in enumerate(CATEGORIES)
        }
        ai_w = ai[:, sl]
        row = {
            **w,
            "P(EVENT)": p_cat,
            "AI_P50": float(np.median(result.ai_summary["daily_p50"][sl])),
            "AI_P95": float(np.median(result.ai_summary["daily_p95"][sl])),
            "P(AI>=40)": float((ai_w >= 40.0).mean()),
            "P(AI>=60)": float((ai_w >= 60.0).mean()),
            "P(AI>=80)": float((ai_w >= 80.0).mean()),
            "P(AI>=95)": float((ai_w >= 95.0).mean()),
            "bands": _band_fractions(ai_w, cfg),
            "pizdec_index": float(
                0.35 * float(np.mean(result.ai_summary["daily_p50"][sl]))
                + 0.65 * float(np.mean(result.ai_summary["daily_p95"][sl]))
            ),
        }
        weeks.append(row)
    hottest = max(weeks, key=lambda r: r["pizdec_index"]) if weeks else None
    hold = result.metrics["HOLD"]
    ranking = policy_ranking(result)
    live = list(live_events or result.extras.get("live_events") or [])
    return {
        "disclaimer": DISCLAIMER,
        "experiment": "pizdec_heatmap",
        "llm_must_not_invent_numbers": True,
        "window": {
            "start": dates[0].isoformat() if dates else None,
            "end": dates[-1].isoformat() if dates else None,
            "days": len(dates),
        },
        "seed": int(cfg["simulation"]["seed"]),
        "worlds": int(hold["n_worlds"]),
        "checkpoint_day": int(checkpoint_day(cfg)),
        "checkpoint_date": str(cfg["bank"]["checkpoint_date"]),
        "cbr_meeting_date": "2026-10-23",
        "shock_severity_multiplier": _f(cfg["simulation"].get("shock_severity_multiplier")),
        "dates": [d.isoformat() for d in dates],
        "categories": list(CATEGORIES),
        "daily": {
            "P(EVENT)": {
                cat: [float(x) for x in p_event[:, k]]
                for k, cat in enumerate(CATEGORIES)
            },
            "AI": daily,
            "P(AI_ge)": daily_p_ai,
        },
        "weeks": weeks,
        "hottest_week": {
            "id": hottest["id"],
            "lo": hottest["lo"],
            "hi": hottest["hi"],
            "pizdec_index": hottest["pizdec_index"],
            "P(AI>=80)": hottest["P(AI>=80)"],
        }
        if hottest
        else None,
        "HOLD_P(SURVIVAL)": float(hold["P(SURVIVAL)"]),
        "HOLD_P(CASH BREACH)": float(hold["P(CASH BREACH)"]),
        "best_policy": ranking[0]["policy"],
        "best_P(SURVIVAL)": float(ranking[0]["P(SURVIVAL)"]),
        "P(DEAL CLOSE)": float(hold["P(DEAL CLOSE)"]),
        "P(BANK APPROVAL)": float(hold["P(BANK APPROVAL)"]),
        "AI_MEDIAN_MAX": float(result.ai_summary["median_max_ai"]),
        "AI_BAND": result.ai_summary["median_max_band"],
        "live_events_n": len(live),
        "live_events": [
            {
                "day": getattr(e, "day", None),
                "category": getattr(e, "category", None),
                "mode": getattr(e, "mode", None),
                "text": str(getattr(e, "text", ""))[:180],
            }
            for e in live
        ],
        "public_inputs": {
            "tag": "PUBLIC PRINT — not Monte Carlo",
            "nwf_total_rub_1sep": 13_194_879_600_000.0,
            "nwf_total_usd_1sep": 154_144_500_000.0,
            "nwf_liquid_rub_1sep": 3_997_967_100_000.0,
            "nwf_liquid_usd_1sep": 46_704_800_000.0,
            "nwf_liquid_gdp_pct": 0.017,
            "cbr_reserves_usd_18sep": 748_200_000_000.0,
            "cbr_key_rate": 0.14,
            "cpi_yoy_aug": 0.063,
            "usd_rub_26sep": 84.3414,
            "cny_rub_26sep": 12.5355,
            "budget_urals_cutoff_2026_usd": 59.0,
            "planned_urals_cutoff_2027_usd": 50.0,
        },
        "assumptions_tag": "ASSUMPTION / PLACEHOLDER — RF prints mapped to knobs, not estimated frequencies",
    }


def run_pizdec_heatmap(
    cfg: Mapping,
    n_worlds: int | None = None,
    seed: int | None = None,
    events_file: str | Path | None = None,
) -> dict:
    cfg = dict(cfg)
    live = collect_events(cfg, file=events_file)
    if live:
        cfg = apply_event_config_overrides(cfg, live)
    overlay = overlay_from_events(live, int(cfg["simulation"]["days"])) if live else None
    n = int(n_worlds if n_worlds is not None else cfg["simulation"]["worlds"])
    seed_i = int(seed if seed is not None else cfg["simulation"]["seed"])
    result = run_oracle(
        cfg,
        n_worlds=n,
        seed=seed_i,
        overlay=overlay,
        live_events=live,
    )
    cal = build_calendar(result, live_events=live)
    cal["runtime_s"] = float(result.runtime_s)
    cal["_result"] = result
    return cal


def write_pizdec_charts(cal: Mapping, chart_dir: str | Path) -> dict[str, Path]:
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)
    cats = list(cal["categories"])
    dates = [dt.date.fromisoformat(x) for x in cal["dates"]]
    weeks = cal["weeks"]
    p_daily = np.array([cal["daily"]["P(EVENT)"][c] for c in cats], dtype=np.float64)
    p_week = np.array(
        [[w["P(EVENT)"][c] for c in cats] for w in weeks], dtype=np.float64
    ).T
    week_labs = [f"{w['id']}\n{w['lo'][5:]}" for w in weeks]
    idx = np.array([w["pizdec_index"] for w in weeks], dtype=np.float64)
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    im = ax.imshow(p_week, origin="upper", cmap="YlOrRd", vmin=0, vmax=max(0.25, float(p_week.max())), aspect="auto")
    ax.set_yticks(range(len(cats)), cats)
    ax.set_xticks(range(len(weeks)), week_labs, fontsize=8)
    ax.set_title(
        "Pizdec heatmap  |  P(category event that week)  |  "
        f"{cal['window']['start']} → {cal['window']['end']}"
    )
    ax.set_xlabel("Week (not interpolated)")
    for i in range(len(cats)):
        for j in range(len(weeks)):
            ax.text(j, i, f"{p_week[i, j]:.0%}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(event | week)")
    p = chart_dir / "pizdec_weekly.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["pizdec_weekly"] = p

    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    im = ax.imshow(
        p_daily,
        origin="upper",
        cmap="YlOrRd",
        vmin=0,
        vmax=max(0.20, float(p_daily.max())),
        aspect="auto",
    )
    ax.set_yticks(range(len(cats)), cats)
    tick_i = list(range(0, len(dates), 7)) + ([len(dates) - 1] if (len(dates) - 1) % 7 else [])
    ax.set_xticks(tick_i, [dates[i].strftime("%d.%m") for i in tick_i], fontsize=8)
    ax.axvline(int(cal["checkpoint_day"]), color="black", ls=":", lw=1.0)
    ax.set_title("Daily P(event)  |  dotted = bank checkpoint")
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(event | day)")
    p = chart_dir / "pizdec_daily.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["pizdec_daily"] = p

    _BAND_PLOT = {
        "NORMAL": "NORMAL",
        "🗿": "MOAI",
        "БЛЯДЬ": "BLYAD",
        "КАКОГО ХУЯ": "KAKOGO",
        "MGS": "MGS",
        "ПОНЕДЕЛЬНИК": "PONED",
    }
    band_names = list(weeks[0]["bands"].keys()) if weeks else []
    band_labs = [_BAND_PLOT.get(b, b) for b in band_names]
    band_mat = np.array([[w["bands"][b] for b in band_names] for w in weeks], dtype=np.float64).T
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    im = ax.imshow(band_mat, origin="upper", cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(band_labs)), band_labs)
    ax.set_xticks(range(len(weeks)), week_labs, fontsize=8)
    ax.set_title("Share of world-days in Absurdity band  |  same CRN")
    for i in range(len(band_names)):
        for j in range(len(weeks)):
            ax.text(j, i, f"{band_mat[i, j]:.0%}", ha="center", va="center", color="black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    p = chart_dir / "pizdec_ai_bands.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["pizdec_ai_bands"] = p

    fig, ax = plt.subplots(figsize=(12.2, 4.8))
    x = mdates.date2num(dates)
    ax.plot(dates, cal["daily"]["AI"]["p50"], color="#1B2A4A", lw=2.0, label="AI median")
    ax.plot(dates, cal["daily"]["AI"]["p95"], color="#8B1E1E", lw=1.6, label="AI P95")
    ax.axhline(40, color="0.5", ls="--", lw=0.8, label="БЛЯДЬ ≥40")
    ax.axhline(80, color="#8B1E1E", ls=":", lw=0.8, label="MGS ≥80")
    chk = dates[int(cal["checkpoint_day"])] if dates else None
    if chk is not None:
        ax.axvline(chk, color="#8B1E1E", ls=":", lw=1.2, label=f"bank {cal['checkpoint_date']}")
    cbr = dt.date(2026, 10, 23)
    if dates[0] <= cbr <= dates[-1]:
        ax.axvline(cbr, color="#C4A35A", ls="--", lw=1.2, label="CBR 23.10")
    ax.set_ylim(0, 100)
    ax.set_xlim(x[0], x[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.set_ylabel("Absurdity Index")
    ax.set_title(
        f"AI calendar  |  median-max={cal['AI_MEDIAN_MAX']:.0f} {cal['AI_BAND']}  |  "
        "not a country forecast"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    p = chart_dir / "pizdec_ai_calendar.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["pizdec_ai_calendar"] = p

    fig, ax = plt.subplots(figsize=(12.2, 3.8))
    colors = ["#8B1E1E" if i == int(str(cal["hottest_week"]["id"]).lstrip("W")) else "#1B2A4A" for i, _ in enumerate(weeks)]
    ax.bar(range(len(weeks)), idx, color=colors)
    ax.set_xticks(range(len(weeks)), week_labs, fontsize=8)
    ax.set_ylabel("pizdec_index = 0.35·AI_p50 + 0.65·AI_p95")
    ax.set_title(
        f"Hottest week on this grid: {cal['hottest_week']['id']}  "
        f"{cal['hottest_week']['lo']}–{cal['hottest_week']['hi']}"
    )
    ax.grid(True, axis="y", alpha=0.3)
    p = chart_dir / "pizdec_index_weeks.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["pizdec_index_weeks"] = p
    return paths


def authoritative_pizdec_block(cal: Mapping) -> str:
    lines = [
        "AUTHORITATIVE FACTS",
        "Pizdec heatmap 2026-09-28 → 2026-11-28. Same seed. Public prints ≠ Monte Carlo.",
        "Do not interpolate weeks. Do not treat this as a forecast of RF.",
        f"SEED: {cal['seed']} WORLDS: {cal['worlds']} DAYS: {cal['window']['days']} "
        f"WINDOW: {cal['window']['start']} .. {cal['window']['end']}",
        f"CHECKPOINT: day {cal['checkpoint_day']} {cal['checkpoint_date']} "
        f"CBR_MEETING: {cal['cbr_meeting_date']}",
        f"SEVERITY_MULT: {cal['shock_severity_multiplier']}",
        f"HOLD P(SURVIVAL): {cal['HOLD_P(SURVIVAL)']:.6f} "
        f"HOLD P(CASH BREACH): {cal['HOLD_P(CASH BREACH)']:.6f} "
        f"best_policy: {cal['best_policy']} "
        f"BEST P(SURVIVAL): {cal['best_P(SURVIVAL)']:.6f} "
        f"P(DEAL CLOSE): {cal['P(DEAL CLOSE)']:.6f} "
        f"P(BANK APPROVAL): {cal['P(BANK APPROVAL)']:.6f} "
        f"AI_MEDIAN_MAX: {cal['AI_MEDIAN_MAX']:.6g} "
        f"AI_BAND: {cal['AI_BAND']}",
        f"HOTTEST_WEEK: {cal['hottest_week']['id']} {cal['hottest_week']['lo']}..{cal['hottest_week']['hi']} "
        f"pizdec_index: {cal['hottest_week']['pizdec_index']:.6g} "
        f"P(AI>=80): {cal['hottest_week']['P(AI>=80)']:.6f}",
        "PUBLIC_INPUTS (not MC): "
        f"NWF_LIQUID_USD: {cal['public_inputs']['nwf_liquid_usd_1sep']:.0f} "
        f"NWF_LIQUID_GDP: {cal['public_inputs']['nwf_liquid_gdp_pct']:.6f} "
        f"CBR_RESERVES_USD: {cal['public_inputs']['cbr_reserves_usd_18sep']:.0f} "
        f"KEY_RATE: {cal['public_inputs']['cbr_key_rate']:.6f} "
        f"CPI_AUG: {cal['public_inputs']['cpi_yoy_aug']:.6f} "
        f"USDRUB: {cal['public_inputs']['usd_rub_26sep']:.4f}",
    ]
    for w in cal["weeks"]:
        pe = " ".join(
            f"{w['id']}_{k}: {w['P(EVENT)'][k]:.6f}" for k in cal["categories"]
        )
        lines.append(
            f"WEEK {w['id']} {w['lo']}..{w['hi']} "
            f"{pe} "
            f"{w['id']}_AI_P50: {w['AI_P50']:.6g} {w['id']}_AI_P95: {w['AI_P95']:.6g} "
            f"{w['id']}_P(AI>=40): {w['P(AI>=40)']:.6f} "
            f"{w['id']}_P(AI>=60): {w['P(AI>=60)']:.6f} "
            f"{w['id']}_P(AI>=80): {w['P(AI>=80)']:.6f} "
            f"{w['id']}_P(AI>=95): {w['P(AI>=95)']:.6f} "
            f"{w['id']}_pizdec_index: {w['pizdec_index']:.6g}"
        )
    lines.append("END AUTHORITATIVE FACTS")
    return "\n".join(lines)


def pizdec_presenter(cal: Mapping, mode: str = "maga") -> str:
    block = authoritative_pizdec_block(cal)
    task = (
        "Сравнительная задача (только по FACTS):\n"
        "1. Какая неделя hottest по pizdec_index.\n"
        "2. Где на календаре вспыхивают FIN / MACRO / GEO (P(EVENT) по неделям).\n"
        "3. Что делает checkpoint 26.10 и заседание ЦБ 23.10 — только если это видно в FACTS.\n"
        "4. HOLD vs best policy. Cash0=500 это арифметика кассы, не «рынок РФ».\n"
        "5. PUBLIC_INPUTS отдельно от симуляции. Не выдавай Minfin/CBR за P(SURVIVAL).\n"
        "Не интерполируй. Пропуск = «данных недостаточно».\n"
    )
    head = (
        "MAGA MODE — PIZDEC HEATMAP 28.09–28.11\n"
        f"Hottest week: {cal['hottest_week']['id']} {cal['hottest_week']['lo']}–{cal['hottest_week']['hi']}\n\n"
        "Ниже только факты симуляции + отдельно помеченные public prints. Это не прогноз страны.\n\n"
        if mode == "maga"
        else "BOARD MODE — PIZDEC HEATMAP\n\n"
    )
    return head + task + "\n" + block + "\n\n" + DISCLAIMER


def pizdec_view_from_payload(payload: Mapping) -> dict:
    body = payload.get("pizdec_heatmap") or payload
    sim = payload.get("simulation") or {}
    return {
        **body,
        "disclaimer": payload.get("disclaimer") or DISCLAIMER,
        "experiment": "pizdec_heatmap",
        "seed": sim.get("seed") or body.get("seed"),
        "worlds": sim.get("worlds") or body.get("worlds"),
    }


def build_pizdec_payload(cal: Mapping) -> dict:
    slim = {k: v for k, v in cal.items() if k != "_result"}
    return {
        "disclaimer": DISCLAIMER,
        "experiment": "pizdec_heatmap",
        "engine": "NULLXES-MAGA-ORACLE Monte Carlo",
        "llm_must_not_invent_numbers": True,
        "scenario": "rf_sep28_nov28",
        "simulation": {
            "worlds": cal["worlds"],
            "days": cal["window"]["days"],
            "seed": cal["seed"],
            "gpu_used": False,
        },
        "known_inputs": {"window": cal["window"]},
        "assumptions": {"tag": cal["assumptions_tag"]},
        "metrics": {},
        "pizdec_heatmap": slim,
    }


def write_pizdec_outputs(cal: Mapping, root: Path, mode: str = "maga") -> dict:
    root = Path(root)
    sim = root / "outputs" / "simulations"
    charts = root / "outputs" / "charts"
    payload = build_pizdec_payload(cal)
    write_json({k: v for k, v in cal.items() if k != "_result"}, sim / "pizdec_heatmap.json")
    write_json(payload, sim / "pizdec_heatmap_payload.json")
    paths = write_pizdec_charts(cal, charts)
    text = pizdec_presenter(cal, mode=mode)
    brief = root / "outputs" / "PIZDEC_HEATMAP_BRIEFING.md"
    brief.write_text(text + "\n", encoding="utf-8")
    snap = root / "тест 28-09-2026" / "charts"
    snap.mkdir(parents=True, exist_ok=True)
    for src in paths.values():
        dest = snap / src.name
        dest.write_bytes(Path(src).read_bytes())
    return {"payload": payload, "briefing_text": text, "charts": paths, "brief_path": brief, "snapshot": snap}
