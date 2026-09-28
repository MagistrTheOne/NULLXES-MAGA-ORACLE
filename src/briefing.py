"""Auto charts + plain-language situation briefing (the пиздец layer).

Numbers stay numbers. Captions say what they mean under the given assumptions.
This is not a forecast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from assumptions import DISCLAIMER, checkpoint_day, daily_burn
from simulation import OracleResult


def _runway_days(cfg: Mapping) -> float:
    burn = daily_burn(cfg)
    cash = float(cfg["company"]["cash_initial"]) + float(cfg["company"].get("bridge_capital", 0.0))
    if burn <= 1e-12:
        return float("inf")
    return cash / burn


def p_breach_by_day(first_breach: np.ndarray, day: int) -> float:
    hit = first_breach >= 0
    return float(np.mean(hit & (first_breach <= day)))


def build_briefing(result: OracleResult, escape: dict | None = None) -> dict:
    cfg = result.cfg
    hold = result.metrics["HOLD"]
    chk = checkpoint_day(cfg)
    fb = result.cash["HOLD"].first_breach_day
    burn = daily_burn(cfg)
    cash0 = float(cfg["company"]["cash_initial"]) + float(cfg["company"].get("bridge_capital", 0.0))
    runway = _runway_days(cfg)
    p_s = hold["P(SURVIVAL)"]
    p_br = hold["P(CASH BREACH)"]
    p_chk = p_breach_by_day(fb, max(chk - 1, 0))
    ai = float(result.ai_summary["median_max_ai"])
    band = result.ai_summary["median_max_band"]

    arith = (
        f"Арифметика, не прогноз: Cash0={cash0:,.0f} RUB, "
        f"daily burn≈{burn:,.0f} RUB, runway без inflows ≈ {runway:.2f} дня. "
        f"Горизонт 90 дней. Банковский checkpoint day {chk} ({cfg['bank']['checkpoint_date']})."
    )

    if runway < 1:
        headline = "КАССА УМИРАЕТ В ДЕНЬ 0"
        why = (
            "500 RUB при ненулевом burn не покрывает даже один операционный день. "
            "Почти все миры идут в FAIL, пока не прилетит deal/bank/bridge/контракт. "
            "Это не «рынок плохой» — это баланс."
        )
    elif runway < chk:
        headline = "ДЫРА ДО БАНКА"
        why = (
            f"Runway ({runway:.1f} дн.) короче, чем путь до банковского checkpoint (day {chk}). "
            f"P(breach до checkpoint)={p_chk:.1%} на этих worlds. "
            "Ждать банк при текущем burn — это ставка, что сделка/шок/мост закроют дыру раньше."
        )
    elif p_br >= 0.8:
        headline = "ХВОСТ СОЖРАЛ БАЗУ"
        why = (
            "Даже если дни формально есть, shocks + deal fail + bank delay "
            "кладут большинство траекторий ниже Cmin."
        )
    else:
        headline = "ЕСТЬ МАССА МИРОВ, ГДЕ ВЫЖИВАЕМ"
        why = (
            "При текущих ASSUMPTION доля survival ненулевая. "
            "Смотри политики и escape solver — не нарратив."
        )

    verdict = (
        f"{headline}. P(SURVIVAL)={p_s:.1%}  P(CASH BREACH)={p_br:.1%}  "
        f"C90 median={hold['c90_median']:,.0f}  ES5={hold['es5']:,.0f}  "
        f"Absurdity median-max={ai:.0f} ({band})."
    )

    class_lines = []
    for key in ("P(FAIL)", "P(RU_EXIT)", "P(ESCAPE)", "P(FREEZE)", "P(TRAP)", "P(SURVIVE)"):
        if key in hold:
            class_lines.append(f"{key}={hold[key]:.1%}")

    policy_lines = []
    best = None
    best_s = -1.0
    for name, m in result.metrics.items():
        policy_lines.append(f"{name}: survival={m['P(SURVIVAL)']:.1%} breach={m['P(CASH BREACH)']:.1%}")
        if m["P(SURVIVAL)"] > best_s:
            best_s = m["P(SURVIVAL)"]
            best = name

    escape = escape or {}
    chosen = (escape.get("minimal_escape_set") or {}).get("chosen") or {}
    bridge = escape.get("min_bridge_for_target") or {}

    live_n = len(getattr(result, "extras", {}).get("live_events", []) or [])
    live_note = (
        f"Live overlay: {live_n} распарсенных событий наложено на те же random streams."
        if live_n
        else "Live overlay: нет (чистый config)."
    )

    paragraphs = [
        DISCLAIMER,
        arith,
        verdict,
        why,
        "Классы (HOLD): " + "  ".join(class_lines),
        "Политики на ОДИНАКОВЫХ мирах: " + " | ".join(policy_lines),
        f"Лучшая политика по P(SURVIVAL): {best} ({best_s:.1%}) — это целевая функция модели, не приказ.",
        live_note,
        (
            "Deal P(close)="
            + f"{hold['P(DEAL CLOSE)']:.1%}, bank P(approval)={hold['P(BANK APPROVAL)']:.1%}. "
            "Approval после 26.10 не автоматический."
        ),
        (
            "Как читать графики: cash bands — где живёт распределение кассы; "
            "если медиана сразу под Cmin, «типичный мир» уже мёртв. "
            "TTFB — в какой день впервые пробиваем Cmin. "
            "C90 — что остаётся на дне 90, включая огромный минус. "
            "AI — насколько сегодняшний score экстремальнее спокойных дней ЭТОЙ ЖЕ симуляции."
        ),
    ]
    if chosen:
        paragraphs.append("Escape set (математика, не совет): " + str(chosen.get("description", chosen)))
    if bridge:
        paragraphs.append("Min bridge @80% survival: " + str(bridge))

    return {
        "headline": headline,
        "verdict": verdict,
        "why": why,
        "arith": arith,
        "text": "\n\n".join(paragraphs),
        "paragraphs": paragraphs,
        "p_survival": p_s,
        "p_breach": p_br,
        "p_breach_before_checkpoint": p_chk,
        "runway_days": runway,
        "checkpoint_day": chk,
        "best_policy": best,
        "ai_band": band,
        "ai": ai,
    }


def write_briefing_markdown(brief: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# NULLXES situation briefing\n\n" + brief["text"] + "\n", encoding="utf-8")
    return path


def generate_situation_charts(
    result: OracleResult,
    escape: dict | None,
    chart_dir: str | Path,
    brief: dict | None = None,
) -> dict[str, Path]:
    """Annotated charts with captions that explain the mess."""
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)
    brief = brief or build_briefing(result, escape)
    paths: dict[str, Path] = {}
    hold = result.metrics["HOLD"]
    bands = hold["bands"]
    days = np.arange(bands["p50"].size)
    chk = brief["checkpoint_day"]
    cmin = float(result.cfg["company"]["cash_minimum"])

    def _caption(ax, text: str) -> None:
        ax.set_xlabel(ax.get_xlabel() + "\n" + text, fontsize=8)
        plt.tight_layout()

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.fill_between(days, bands["p5"], bands["p95"], color="#C4A35A", alpha=0.28, label="P5–P95: почти все миры")
    ax.fill_between(days, bands["p25"], bands["p75"], color="#1B2A4A", alpha=0.22, label="P25–P75: типичная масса")
    ax.plot(days, bands["p50"], color="#1B2A4A", lw=2.2, label="Медиана")
    ax.axhline(cmin, color="#8B1E1E", ls="--", lw=1.2, label="Cmin")
    ax.axvline(chk, color="#8B1E1E", ls=":", lw=1.4, label=f"Bank checkpoint day {chk}")
    ax.set_title(f"Касса 90 дней (HOLD)\n{brief['headline']}")
    ax.set_ylabel("RUB")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    _caption(
        ax,
        "Если медиана сразу под Cmin — «обычный» мир уже в FAIL. Шкала в минусах = дыра, не прогноз ВВП.",
    )
    p = chart_dir / "cash_bands.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["cash_bands"] = p

    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = list(result.metrics.keys())
    surv = [result.metrics[n]["P(SURVIVAL)"] for n in names]
    colors = ["#1F6B4A" if n == brief.get("best_policy") else "#1B2A4A" for n in names]
    ax.bar(names, surv, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(SURVIVAL) = 1 − P(FAIL)")
    ax.set_title("Политики на ОДИНАКОВЫХ случайных мирах (CRN)")
    ax.grid(True, axis="y", alpha=0.3)
    _caption(ax, "Зелёный = max survival. Это не «лучшая стратегия в жизни», это max целевой функции модели.")
    p = chart_dir / "survival_policy.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["survival_policy"] = p

    ttfb = hold["ttfb"]["histogram"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(np.arange(len(ttfb)), ttfb, color="#8B1E1E")
    ax.axvline(chk, color="#C4A35A", ls="--", label=f"checkpoint {chk}")
    ax.set_title(f"Time-to-first-breach  |  P(breach до банка)={brief['p_breach_before_checkpoint']:.1%}")
    ax.set_xlabel("День первого Cash < Cmin")
    ax.set_ylabel("Число миров")
    ax.legend(fontsize=8)
    _caption(ax, "Пик на дне 0–2 = касса не переживает burn. Пик после checkpoint = банк не успел / отказал.")
    p = chart_dir / "ttfb.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["ttfb"] = p

    c90 = hold["cash90"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hist(c90, bins=40, color="#1B2A4A", alpha=0.9)
    ax.axvline(np.median(c90), color="#C4A35A", lw=2, label="median")
    ax.axvline(np.quantile(c90, 0.05), color="#8B1E1E", lw=2, ls="--", label="P5 / хвост")
    ax.set_title(f"C90  median={hold['c90_median']:,.0f}  ES5={hold['es5']:,.0f}")
    ax.set_xlabel("Cash на дне 90 (RUB)")
    ax.legend(fontsize=8)
    _caption(ax, "ES5 = среднее по худшим 5% миров. Не смотри на среднее всего распределения — его сожрёт хвост.")
    p = chart_dir / "c90.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["c90"] = p

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(result.ai_summary["daily_p50"], color="#1B2A4A", label="AI median")
    ax.plot(result.ai_summary["daily_p95"], color="#8B1E1E", label="AI P95")
    ax.set_ylim(0, 100)
    ax.set_title(f"Absurdity Index  median-max={brief['ai']:.0f}  {brief['ai_band']}")
    ax.set_ylabel("AI")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _caption(
        ax,
        "0–20 NORMAL · 20–40 🗿 · 40–60 БЛЯДЬ · 60–80 КАКОГО ХУЯ · 80–95 MGS · 95–100 ПОНЕДЕЛЬНИК. "
        "AI = 100×P(score_спокойнее < сегодня) внутри ЭТОЙ симуляции.",
    )
    p = chart_dir / "ai_time.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["ai_time"] = p

    # One-pager
    fig = plt.figure(figsize=(12, 8.5))
    fig.suptitle("NULLXES 90D  —  situation one-pager", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.02, DISCLAIMER, fontsize=7, wrap=True)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28, left=0.07, right=0.98, top=0.88, bottom=0.12)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(days, bands["p5"], bands["p95"], color="#C4A35A", alpha=0.3)
    ax1.plot(days, bands["p50"], color="#1B2A4A")
    ax1.axhline(cmin, color="#8B1E1E", ls="--")
    ax1.axvline(chk, color="#8B1E1E", ls=":")
    ax1.set_title("Cash median + P5–P95")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(names, surv, color=colors)
    ax2.set_ylim(0, 1)
    ax2.set_title("Survival by policy")
    ax2.tick_params(axis="x", rotation=20)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(np.arange(len(ttfb)), ttfb, color="#8B1E1E")
    ax3.axvline(chk, color="#C4A35A", ls="--")
    ax3.set_title("First breach day")
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    block = (
        brief["headline"]
        + "\n\n"
        + brief["arith"]
        + "\n\n"
        + brief["verdict"]
        + "\n\n"
        + brief["why"]
    )
    ax4.text(0.0, 1.0, block, va="top", ha="left", fontsize=8, wrap=True, family="DejaVu Sans")
    p = chart_dir / "situation_onepager.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths["situation_onepager"] = p
    return paths
