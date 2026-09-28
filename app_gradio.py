"""Gradio control panel. Monte Carlo always runs. Qwen is optional.

Launch (Colab or local, no LLM download by default):
    python app_gradio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from assumptions import DISCLAIMER, load_config, set_path
from briefing import build_briefing, generate_situation_charts, write_briefing_markdown
from escape_solver import policy_ranking, solve_escape
from facts import present_without_llm
from live_events import (
    apply_event_config_overrides,
    collect_events,
    overlay_from_events,
)
from payload import build_analysis_payload, write_json
from qwen_analyst import generate_briefing
from sensitivity import run_sensitivity
from simulation import run_oracle

try:
    import gradio as gr
except ImportError as exc:
    raise SystemExit("pip install gradio") from exc


def run_panel(
    cash0,
    burn_monthly,
    p_bank,
    p_deal,
    deal_capital,
    bridge,
    scenario,
    policy,
    worlds,
    live_text,
    analyst_mode,
    analyst_backend,
    model_id,
    skip_solver,
):
    cfg_name = {
        "baseline": "configs/baseline.yaml",
        "stress": "configs/stress.yaml",
        "monday": "configs/monday.yaml",
    }[scenario]
    cfg = load_config(cfg_name)
    cfg = set_path(cfg, "company.cash_initial", float(cash0))
    cfg = set_path(cfg, "company.burn_monthly", float(burn_monthly))
    cfg = set_path(cfg, "bank.approval_probability", float(p_bank))
    cfg = set_path(cfg, "deal.close_probability", float(p_deal))
    cfg = set_path(cfg, "deal.target_capital", float(deal_capital))
    cfg = set_path(cfg, "company.bridge_capital", float(bridge))
    cfg = set_path(cfg, "simulation.worlds", int(worlds))

    events = collect_events(cfg, text=live_text or "")
    if events:
        cfg = apply_event_config_overrides(cfg, events)
    overlay = overlay_from_events(events, int(cfg["simulation"]["days"])) if events else None

    result = run_oracle(
        cfg,
        n_worlds=int(worlds),
        overlay=overlay,
        live_events=events,
        policies=None,
    )
    escape = {"policy_ranking": policy_ranking(result), "best_policy_by_survival": policy_ranking(result)[0]["policy"]}
    sens = []
    if not skip_solver:
        sens = run_sensitivity(cfg, result)
        escape = solve_escape(cfg, result, sens)

    pol = policy if policy in result.metrics else "HOLD"
    payload = build_analysis_payload(cfg, result, escape, sens, policy=pol, scenario=scenario)
    write_json(payload, ROOT / "outputs" / "simulations" / "analysis_payload.json")
    brief = build_briefing(result, escape)
    write_briefing_markdown(brief, ROOT / "outputs" / "PIZDEC_BRIEFING.md")
    charts = generate_situation_charts(result, escape, ROOT / "outputs" / "charts", brief)

    mm = result.metrics[pol]
    kpis = (
        f"SURVIVAL     {mm['P(SURVIVAL)']:.1%}\n"
        f"CASH BREACH  {mm['P(CASH BREACH)']:.1%}\n"
        f"ESCAPE class {mm.get('P(ESCAPE)', 0):.1%}\n"
        f"FAIL class   {mm.get('P(FAIL)', 0):.1%}\n"
        f"C90 median   {mm['c90_median']:,.0f} RUB\n"
        f"ES5          {mm['es5']:,.0f} RUB\n"
        f"BRIDGE@80%   {payload['escape_solver']['bridge_for_80pct_survival']}\n"
        f"AI           {result.ai_summary['median_max_ai']:.1f} — {result.ai_summary['median_max_band']}\n"
        f"best policy  {payload['policy_comparison']['best_policy_by_survival']}\n"
        f"live events  {payload['live_events_n']}\n"
        f"\n{DISCLAIMER}"
    )

    try:
        qwen = generate_briefing(
            payload,
            mode=analyst_mode,
            backend=analyst_backend,
            model_id=model_id,
        )
        qwen_text = qwen["text"]
        if not qwen["used_llm"]:
            qwen_text = (
                "[LLM выключен: backend=facts. Подключи HF/API в Colab, чтобы грузить Qwen.]\n\n"
                + qwen_text
            )
        (ROOT / "outputs" / "QWEN_BRIEFING.md").write_text(qwen_text + "\n", encoding="utf-8")
    except Exception as e:
        qwen_text = present_without_llm(payload, mode=analyst_mode) + f"\n\n[LLM error: {e}]"

    onepager = str(charts.get("situation_onepager", ""))
    return kpis, qwen_text, onepager, str(ROOT / "outputs" / "simulations" / "analysis_payload.json")


def build_ui():
    with gr.Blocks(title="NULLXES MAGA ORACLE") as demo:
        gr.Markdown(
            "# NULLXES MAGA ORACLE\n"
            "Слой 1 = Monte Carlo. Слой 2 = Qwen только как аналитик JSON. "
            "По умолчанию веса **не качаются** (backend=facts)."
        )
        with gr.Row():
            with gr.Column():
                cash0 = gr.Number(value=500, label="Cash0")
                burn = gr.Number(value=200000, label="Burn monthly (ASSUMPTION)")
                p_bank = gr.Slider(0, 1, value=0.22, step=0.01, label="P(bank approval) ASSUMPTION")
                p_deal = gr.Slider(0, 1, value=0.12, step=0.01, label="P(deal close) ASSUMPTION")
                deal_cap = gr.Number(value=30000000, label="Deal target capital (NOT received)")
                bridge = gr.Number(value=0, label="Bridge capital")
                scenario = gr.Dropdown(["baseline", "stress", "monday"], value="baseline", label="Scenario")
                policy = gr.Dropdown(
                    ["HOLD", "FREEZE_50", "FREEZE_90", "CAPITAL_FIRST", "EXIT_RU"],
                    value="HOLD",
                    label="Policy view",
                )
                worlds = gr.Dropdown([1000, 10000, 100000], value=1000, label="Worlds")
                live = gr.Textbox(
                    label="Live events (one per line)",
                    lines=4,
                    placeholder="2026-09-29 GEO санкционный комплаенс остановил платёж",
                )
                mode = gr.Radio(["board", "maga"], value="maga", label="Presenter")
                backend = gr.Dropdown(
                    ["facts", "qwen", "openai"],
                    value="facts",
                    label="Analyst backend (facts = no download)",
                )
                model_id = gr.Textbox(value="Qwen/Qwen3-1.7B", label="HF model id (Colab only)")
                skip_solver = gr.Checkbox(value=True, label="Skip sensitivity/escape (faster)")
                run_btn = gr.Button("RUN MGS", variant="primary")
            with gr.Column():
                kpis = gr.Textbox(label="KPIs", lines=14)
                qwen = gr.Textbox(label="Qwen / facts briefing", lines=18)
                img = gr.Image(label="Situation one-pager")
                payload_path = gr.Textbox(label="analysis_payload.json")
        run_btn.click(
            run_panel,
            inputs=[
                cash0, burn, p_bank, p_deal, deal_cap, bridge, scenario, policy,
                worlds, live, mode, backend, model_id, skip_solver,
            ],
            outputs=[kpis, qwen, img, payload_path],
        )
        gr.Markdown(DISCLAIMER)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0")
