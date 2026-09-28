"""Excel workbook: outputs/NULLXES_90D_ORACLE.xlsx

Sheets: DASHBOARD, ASSUMPTIONS, EVENTS, 90_DAY_PATH, MONTE_CARLO,
POLICIES, SENSITIVITY, ESCAPE_SOLVER, NULLXES
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd

from assumptions import (
    CATEGORIES,
    DISCLAIMER,
    CLASS_NAMES,
    assumption_table,
    flatten_leaves,
    parse_date,
)
from briefing import build_briefing, generate_situation_charts, write_briefing_markdown
from simulation import OracleResult

NAVY = "1B2A4A"
GOLD = "C4A35A"
RED = "8B1E1E"
GREEN = "1F6B4A"
GRAY = "F4F1EA"
WHITE = "FFFFFF"


def _fill(hex_color: str) -> PatternFill:
    return PatternFill("solid", fgColor=hex_color)


def _header_font() -> Font:
    return Font(name="Calibri", bold=True, color=WHITE, size=11)


def _title_font() -> Font:
    return Font(name="Calibri", bold=True, color=NAVY, size=16)


def _thin() -> Border:
    s = Side(style="thin", color="C5C1B7")
    return Border(left=s, right=s, top=s, bottom=s)


def _style_header_row(ws, row: int, n_cols: int, color: str = NAVY) -> None:
    for c in range(1, n_cols + 1):
        cell = ws.cell(row, c)
        cell.fill = _fill(color)
        cell.font = _header_font()
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = _thin()


def _autosize(ws, min_w=12, max_w=42) -> None:
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        width = min_w
        for cell in col[:40]:
            if cell.value is None:
                continue
            width = max(width, min(max_w, len(str(cell.value)) + 2))
        ws.column_dimensions[letter].width = width


def _write_df(ws, df: pd.DataFrame, start_row: int = 1, header_color: str = NAVY) -> int:
    rows = list(dataframe_to_rows(df, index=False, header=True))
    for i, row in enumerate(rows):
        for j, val in enumerate(row, start=1):
            cell = ws.cell(start_row + i, j, val)
            cell.border = _thin()
            cell.alignment = Alignment(vertical="center", wrap_text=True)
    _style_header_row(ws, start_row, df.shape[1], header_color)
    return start_row + len(rows)


def _kpi(ws, row: int, col: int, label: str, value: str, fill: str) -> None:
    lab = ws.cell(row, col, label)
    lab.fill = _fill(fill)
    lab.font = Font(name="Calibri", bold=True, color=WHITE, size=10)
    lab.alignment = Alignment(horizontal="center")
    val = ws.cell(row + 1, col, value)
    val.font = Font(name="Calibri", bold=True, size=14, color=NAVY)
    val.alignment = Alignment(horizontal="center")
    val.fill = _fill(GRAY)


def _pct(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{100.0 * float(x):.2f}%"


def _num(x: float, nd=0) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if nd == 0:
        return f"{float(x):,.0f}"
    return f"{float(x):,.{nd}f}"


def _dates(cfg: Mapping) -> list[dt.date]:
    start = parse_date(cfg["simulation"]["start_date"])
    days = int(cfg["simulation"]["days"])
    return [start + dt.timedelta(days=i) for i in range(days)]


def _save_charts(result: OracleResult, escape: dict, sens_rows: list[dict], chart_dir: Path) -> dict[str, Path]:
    chart_dir.mkdir(parents=True, exist_ok=True)
    brief = build_briefing(result, escape)
    write_briefing_markdown(brief, chart_dir.parent / "PIZDEC_BRIEFING.md")
    paths = generate_situation_charts(result, escape, chart_dir, brief)
    # keep tornado from sensitivity if present
    if sens_rows:
        from sensitivity import tornado_table

        torn = tornado_table(sens_rows)[:12]
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = [r["parameter"] for r in torn][::-1]
        deltas = [r["delta_survival"] for r in torn][::-1]
        cols = ["#1F6B4A" if d >= 0 else "#8B1E1E" for d in deltas]
        ax.barh(labels, deltas, color=cols)
        ax.axvline(0, color="#333", lw=1)
        ax.set_xlabel("Delta P(SURVIVAL) vs baseline")
        ax.set_title("Sensitivity tornado — какой рычаг двигает survival")
        fig.tight_layout()
        p = chart_dir / "sensitivity_tornado.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths["sensitivity_tornado"] = p
    result.extras["briefing"] = brief
    return paths


def _scalar_metrics(m: dict) -> dict:
    skip = {"bands", "cash90", "class", "first_breach_day", "ttfb"}
    out = {}
    for k, v in m.items():
        if k in skip:
            continue
        if isinstance(v, (int, float, str, bool, np.floating, np.integer)):
            out[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
    ttfb = m.get("ttfb") or {}
    out["ttfb_n_breached"] = ttfb.get("n_breached")
    out["ttfb_n_never"] = ttfb.get("n_never")
    out["ttfb_median_day_if_breach"] = ttfb.get("median_day_if_breach")
    return out


def write_workbook(
    path: str | Path,
    result: OracleResult,
    sensitivity_rows: list[dict],
    escape: dict,
    ladder: list[dict] | None = None,
    invariant_failures: list[str] | None = None,
    chart_dir: str | Path | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chart_dir = Path(chart_dir or path.parent / "charts")
    charts = _save_charts(result, escape, sensitivity_rows, chart_dir)

    cfg = result.cfg
    hold = result.metrics["HOLD"]
    wb = Workbook()

    # ---- DASHBOARD ----
    ws = wb.active
    ws.title = "DASHBOARD"
    ws["A1"] = "NULLXES 90D ORACLE — DASHBOARD"
    ws["A1"].font = _title_font()
    ws.merge_cells("A1:H1")
    ws["A2"] = DISCLAIMER
    ws["A2"].alignment = Alignment(wrap_text=True)
    ws.merge_cells("A2:H2")
    ws.row_dimensions[2].height = 36
    ws["A3"] = (
        f"Worlds={hold['n_worlds']}  Days={cfg['simulation']['days']}  "
        f"Seed={cfg['simulation']['seed']}  Config={cfg.get('meta', {}).get('name', 'baseline')}  "
        f"Runtime={result.runtime_s:.2f}s  GPU=NOT USED (NumPy CPU)"
    )
    brief = (result.extras or {}).get("briefing") or {}
    ws["A4"] = brief.get("verdict", "")
    ws["A4"].alignment = Alignment(wrap_text=True, vertical="center")
    ws.merge_cells("A4:L4")
    ws.row_dimensions[4].height = 48

    kpis = [
        ("P(SURVIVAL)", _pct(hold["P(SURVIVAL)"]), GREEN),
        ("P(CASH BREACH)", _pct(hold["P(CASH BREACH)"]), RED),
        ("P(FIN BEFORE BREACH)", _pct(hold["P(FINANCING BEFORE BREACH)"]), NAVY),
        ("P(DEAL CLOSE)", _pct(hold["P(DEAL CLOSE)"]), NAVY),
        ("P(BANK APPROVAL)", _pct(hold["P(BANK APPROVAL)"]), NAVY),
        ("P(FREEZE_90)", _pct(result.metrics.get("FREEZE_90", {}).get("P(90% R&D FREEZE)", 0)), GOLD),
        ("P(RU EXIT)", _pct(result.metrics.get("EXIT_RU", {}).get("P(RU CONTOUR EXIT TRIGGER)", 0)), RED),
        ("C90 median", _num(hold["c90_median"]), NAVY),
        ("C90 P5", _num(hold["c90_p5"]), NAVY),
        ("C90 P95", _num(hold["c90_p95"]), NAVY),
        ("ES5", _num(hold["es5"]), RED),
        ("AI median-max", f"{result.ai_summary['median_max_ai']:.1f} {result.ai_summary['median_max_band']}", GOLD),
    ]
    for i, (lab, val, col) in enumerate(kpis):
        _kpi(ws, 5, 1 + i, lab, val, col)

    best = escape.get("best_policy_by_survival", "HOLD")
    bridge = escape.get("min_bridge_for_target") or {}
    burn = escape.get("max_burn_for_target") or {}
    chosen = (escape.get("minimal_escape_set") or {}).get("chosen") or {}
    ws["A8"] = "Best survival policy (CRN objective P(SURVIVAL), not a recommendation)"
    ws["A9"] = best
    ws["C8"] = "Required bridge for 80% survival"
    ws["C9"] = (
        _num(bridge.get("value")) + " RUB"
        if bridge.get("feasible")
        else f"not feasible in search range (surv={_pct(bridge.get('survival'))})"
    )
    ws["E8"] = "Maximum monthly burn for 80% survival"
    ws["E9"] = (
        _num(burn.get("value")) + " RUB/mo"
        if burn.get("feasible")
        else f"not feasible (surv@0={_pct(burn.get('survival'))})"
    )
    ws["A10"] = "Escape set (minimal tested levers hitting target under assumptions)"
    ws["A11"] = str(chosen.get("description", (escape.get("minimal_escape_set") or {}).get("message", "")))
    ws.merge_cells("A11:H11")

    r = 13
    for key, pth in charts.items():
        img = XLImage(str(pth))
        img.width = 620
        img.height = 280
        ws.add_image(img, f"A{r}")
        r += 16

    if ladder:
        ws.cell(r, 1, "Scale ladder (Colab CPU)")
        r += 1
        df_l = pd.DataFrame(ladder)
        _write_df(ws, df_l, r)
        r += len(df_l) + 3
    ws.cell(r, 1, "Invariant failures (empty = pass)")
    ws.cell(r + 1, 1, "PASS" if not invariant_failures else "; ".join(invariant_failures))

    # ---- ASSUMPTIONS ----
    ws_a = wb.create_sheet("ASSUMPTIONS")
    ws_a["A1"] = DISCLAIMER
    ws_a.merge_cells("A1:D1")
    rows_a = assumption_table(cfg)
    df_a = pd.DataFrame(rows_a)
    df_a["value"] = df_a["value"].map(lambda x: str(x)[:500])
    _write_df(ws_a, df_a, 3)
    leaves = flatten_leaves({k: v for k, v in cfg.items() if not str(k).startswith("_")})
    df_lvs = pd.DataFrame(leaves, columns=["key", "value"])
    df_lvs["value"] = df_lvs["value"].map(lambda x: str(x)[:500])
    ws_a.cell(5 + len(df_a), 1, "Full config leaves")
    _write_df(ws_a, df_lvs, 6 + len(df_a), header_color=GOLD)
    _autosize(ws_a)

    # ---- EVENTS ----
    ws_e = wb.create_sheet("EVENTS")
    ws_e["A1"] = DISCLAIMER
    ev = result.bundle.events
    n, t_days, k = ev.shape
    realized = []
    for i, name in enumerate(CATEGORIES):
        any_w = ev[:, :, i].any(axis=1)
        h = cfg["hazards"][name]
        p90_ass = h.get("p90", h.get("p_daily"))
        realized.append(
            {
                "category": name,
                "ASSUMPTION_P90_or_p_daily": p90_ass,
                "realized_P(at_least_one)": float(any_w.mean()),
                "mean_event_days": float(ev[:, :, i].sum(axis=1).mean()),
                "mean_severity_if_event": float(
                    result.bundle.severity[:, :, i][ev[:, :, i]].mean()
                    if ev[:, :, i].any()
                    else 0.0
                ),
                "tag": "ASSUMPTION vs realized frequency on these worlds",
            }
        )
    _write_df(ws_e, pd.DataFrame(realized), 3)
    ws_e["A12"] = (
        "Realized frequencies are NOT a validation of the P90 assumptions. "
        "Contagion and Z_t change the event rate relative to lambda_k0."
    )
    _autosize(ws_e)

    # ---- 90_DAY_PATH ----
    ws_p = wb.create_sheet("90_DAY_PATH")
    ws_p["A1"] = DISCLAIMER
    dates = _dates(cfg)
    bands = hold["bands"]
    df_p = pd.DataFrame(
        {
            "day": np.arange(t_days),
            "date": [d.isoformat() for d in dates],
            "cash_p5": bands["p5"],
            "cash_p25": bands["p25"],
            "cash_p50": bands["p50"],
            "cash_p75": bands["p75"],
            "cash_p95": bands["p95"],
            "ai_p50": result.ai_summary["daily_p50"],
            "ai_p95": result.ai_summary["daily_p95"],
            "ai_mean": result.ai_summary["daily_mean"],
            "frac_worlds_breached_by_day": np.cumsum(
                np.bincount(
                    result.metrics["HOLD"]["first_breach_day"][
                        result.metrics["HOLD"]["first_breach_day"] >= 0
                    ],
                    minlength=t_days,
                )
            )
            / float(n),
        }
    )
    _write_df(ws_p, df_p, 3)
    _autosize(ws_p)
    # native line chart of median cash
    chart = LineChart()
    chart.title = "HOLD cash percentiles"
    chart.y_axis.title = "RUB"
    chart.x_axis.title = "Day"
    data = Reference(ws_p, min_col=3, max_col=7, min_row=3, max_row=3 + t_days)
    cats = Reference(ws_p, min_col=1, min_row=4, max_row=3 + t_days)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    chart.height = 8
    chart.width = 18
    ws_p.add_chart(chart, "L3")

    # ---- MONTE_CARLO ----
    ws_m = wb.create_sheet("MONTE_CARLO")
    ws_m["A1"] = DISCLAIMER
    df_m = pd.DataFrame([_scalar_metrics(hold)])
    _write_df(ws_m, df_m, 3)
    sample_n = min(250, n)
    rng = np.random.default_rng(int(cfg["simulation"]["seed"]))
    idx = rng.choice(n, size=sample_n, replace=False)
    cls = result.metrics["HOLD"]["class"]
    sample = pd.DataFrame(
        {
            "world": idx,
            "class": [CLASS_NAMES[int(c)] for c in cls[idx]],
            "cash90": result.metrics["HOLD"]["cash90"][idx],
            "min_cash": result.cash["HOLD"].min_cash[idx],
            "first_breach_day": result.cash["HOLD"].first_breach_day[idx],
            "max_drawdown": result.cash["HOLD"].max_drawdown[idx],
            "crisis_days": result.cash["HOLD"].crisis_days[idx],
            "max_simultaneous": result.cash["HOLD"].max_simultaneous[idx],
            "max_AI": result.ai_summary["max_ai"][idx],
            "deal_status": result.bundle.deal.status[idx],
            "bank_status": result.bundle.bank.status[idx],
        }
    )
    ws_m.cell(6, 1, f"Sample of {sample_n} worlds (not the full {n})")
    _write_df(ws_m, sample, 7, header_color=GOLD)
    _autosize(ws_m)

    # ---- POLICIES ----
    ws_pol = wb.create_sheet("POLICIES")
    ws_pol["A1"] = DISCLAIMER
    pol_rows = []
    for name, m in result.metrics.items():
        row = _scalar_metrics(m)
        pol_rows.append(row)
    _write_df(ws_pol, pd.DataFrame(pol_rows), 3)
    rank = pd.DataFrame(escape.get("policy_ranking") or [])
    if not rank.empty:
        ws_pol.cell(6 + len(pol_rows), 1, "Ranking by P(SURVIVAL) on identical worlds")
        _write_df(ws_pol, rank, 7 + len(pol_rows), header_color=GREEN)
    _autosize(ws_pol)

    # ---- SENSITIVITY ----
    ws_s = wb.create_sheet("SENSITIVITY")
    ws_s["A1"] = DISCLAIMER
    if sensitivity_rows:
        df_s = pd.DataFrame(sensitivity_rows)
        _write_df(ws_s, df_s, 3)
        chart_b = BarChart()
        chart_b.type = "bar"
        chart_b.title = "abs delta P(SURVIVAL)"
        # use a compact ranked table starting at column 20
        from sensitivity import tornado_table

        torn = tornado_table(sensitivity_rows)
        df_t = pd.DataFrame(
            [{"parameter": r["parameter"], "abs_delta": r["abs_delta"], "delta": r["delta_survival"]} for r in torn]
        )
        start = 3
        col0 = 16
        ws_s.cell(2, col0, "Ranked |delta|")
        for j, name in enumerate(df_t.columns, start=col0):
            ws_s.cell(start, j, name)
        for i, rec in enumerate(df_t.to_dict("records"), start=start + 1):
            ws_s.cell(i, col0, rec["parameter"])
            ws_s.cell(i, col0 + 1, rec["abs_delta"])
            ws_s.cell(i, col0 + 2, rec["delta"])
        data_ref = Reference(ws_s, min_col=col0 + 1, min_row=start, max_row=start + len(df_t))
        cats_ref = Reference(ws_s, min_col=col0, min_row=start + 1, max_row=start + len(df_t))
        chart_b.add_data(data_ref, titles_from_data=True)
        chart_b.set_categories(cats_ref)
        chart_b.shape = 4
        chart_b.height = 10
        chart_b.width = 16
        ws_s.add_chart(chart_b, "A" + str(6 + len(df_s)))
    _autosize(ws_s)

    # ---- ESCAPE_SOLVER ----
    ws_x = wb.create_sheet("ESCAPE_SOLVER")
    ws_x["A1"] = DISCLAIMER
    ws_x["A3"] = "Target P(SURVIVAL)"
    ws_x["B3"] = escape.get("target_survival")
    ws_x["A4"] = "Best policy by P(SURVIVAL)"
    ws_x["B4"] = escape.get("best_policy_by_survival")
    ws_x["A5"] = "Min bridge"
    ws_x["B5"] = str(escape.get("min_bridge_for_target"))
    ws_x["A6"] = "Max burn"
    ws_x["B6"] = str(escape.get("max_burn_for_target"))
    ws_x["A7"] = "Min contract inflow (material lift)"
    ws_x["B7"] = str(escape.get("min_contract_inflow_material"))
    ws_x["A8"] = "Bank vs immediate FREEZE crossover P(approval)"
    ws_x["B8"] = str((escape.get("bank_vs_immediate_freeze") or {}).get("crossover_p_bank_approval"))
    ws_x["A9"] = "Strongest sensitivity factor"
    ws_x["B9"] = str(escape.get("strongest_sensitivity"))
    ws_x["A10"] = "Minimal escape set"
    ws_x["B10"] = str(escape.get("minimal_escape_set"))
    ws_x["A12"] = "Financing timing curve"
    timing = pd.DataFrame(escape.get("financing_timing_curve") or [])
    if not timing.empty:
        _write_df(ws_x, timing, 13)
    curve = pd.DataFrame((escape.get("bank_vs_immediate_freeze") or {}).get("curve") or [])
    if not curve.empty:
        ws_x.cell(15 + len(timing), 1, "HOLD vs immediate FREEZE_90 by P(bank approval)")
        _write_df(ws_x, curve, 16 + len(timing), header_color=GOLD)
    cands = pd.DataFrame((escape.get("minimal_escape_set") or {}).get("candidates") or [])
    if not cands.empty:
        ws_x.cell(20 + len(timing) + len(curve), 1, "Escape candidates")
        _write_df(ws_x, cands, 21 + len(timing) + len(curve), header_color=GREEN)
    for col in ws_x.column_dimensions:
        ws_x.column_dimensions[col].width = 28
    ws_x.column_dimensions["B"].width = 80
    ws_x["A2"] = (
        "These are model-implied thresholds under PLACEHOLDER assumptions, "
        "not an objective recommendation and not a forecast."
    )

    # ---- NULLXES ----
    ws_n = wb.create_sheet("NULLXES")
    ws_n["A1"] = "NULLXES — model identity"
    ws_n["A1"].font = _title_font()
    facts = [
        ("Company", cfg["company"]["name"]),
        ("Status", cfg["company"]["status"]),
        ("Sector", cfg["company"]["sector"]),
        ("Cash0 (RUB)", cfg["company"]["cash_initial"]),
        ("Cmin (RUB)", cfg["company"]["cash_minimum"]),
        ("External founder income", cfg["company"]["founder_external_income_daily"]),
        ("External liquidity now", cfg["company"]["external_liquidity"]),
        ("Bank funding available now", cfg["company"]["bank_funding_available_now"]),
        ("Burn monthly (PLACEHOLDER)", cfg["company"]["burn_monthly"]),
        ("Deal target (NOT received)", cfg["deal"]["target_capital"]),
        ("Bank checkpoint", cfg["bank"]["checkpoint_date"]),
        ("Model start", cfg["simulation"]["start_date"]),
        ("Horizon days", cfg["simulation"]["days"]),
        ("Production worlds", hold["n_worlds"]),
        ("GPU", "Not used. Vectorized NumPy on CPU is the correct backend."),
        ("Disclaimer", DISCLAIMER),
    ]
    ws_n["A3"] = "Field"
    ws_n["B3"] = "Value"
    _style_header_row(ws_n, 3, 2)
    for i, (k, v) in enumerate(facts, start=4):
        ws_n.cell(i, 1, k)
        ws_n.cell(i, 2, v)
    ws_n.column_dimensions["A"].width = 36
    ws_n.column_dimensions["B"].width = 88
    ws_n["A22"] = "Outcome class definitions (formal)"
    ws_n["A23"] = (
        "FAIL: exists t Cash_t < Cmin. "
        "RU_EXIT: not FAIL and EXIT_RU fired. "
        "ESCAPE: not FAIL, not RU_EXIT, material financing, Cash_T >= Cmin + escape_runway*burn. "
        "FREEZE: not FAIL/RU_EXIT/ESCAPE and FREEZE_90 was active. "
        "TRAP: survived but no material financing or runway below trap_runway_days. "
        "SURVIVE: residual non-FAIL trajectory. "
        "Metric P(SURVIVAL) = 1 - P(FAIL), which is not P(class=SURVIVE)."
    )
    ws_n["A23"].alignment = Alignment(wrap_text=True)
    ws_n.merge_cells("A23:B26")
    ws_n.row_dimensions[23].height = 80

    for sheet in wb.worksheets:
        sheet.freeze_panes = "A4"
        sheet.page_setup.fitToPage = True
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        sheet.print_title_rows = "1:2"

    wb.save(path)
    return path
