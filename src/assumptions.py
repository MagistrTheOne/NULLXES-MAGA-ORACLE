"""Assumption catalog, YAML loading, nested overrides, disclaimer.

Every unknown probability in this project is an ASSUMPTION / PLACEHOLDER.
None of the priors below are empirical frequencies.
"""

from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path
from typing import Any, Mapping

import yaml

DISCLAIMER = (
    "Результаты являются условными исходами модели при заданных допущениях "
    "и не являются прогнозом реальных геополитических, военных или экономических событий."
)

CATEGORIES: tuple[str, ...] = ("SEC", "FIN", "MACRO", "GEO", "BIZ", "BLACK_SWAN")
REGULAR_CATEGORIES: tuple[str, ...] = ("SEC", "FIN", "MACRO", "GEO", "BIZ")

POLICY_NAMES: tuple[str, ...] = (
    "HOLD",
    "FREEZE_50",
    "FREEZE_90",
    "CAPITAL_FIRST",
    "EXIT_RU",
)

# Formal class codes (mutually exclusive, priority order).
CLASS_FAIL = 0
CLASS_RU_EXIT = 1
CLASS_ESCAPE = 2
CLASS_FREEZE = 3
CLASS_TRAP = 4
CLASS_SURVIVE = 5
CLASS_NAMES = {
    CLASS_FAIL: "FAIL",
    CLASS_RU_EXIT: "RU_EXIT",
    CLASS_ESCAPE: "ESCAPE",
    CLASS_FREEZE: "FREEZE",
    CLASS_TRAP: "TRAP",
    CLASS_SURVIVE: "SURVIVE",
}

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs"

# Keys whose values are unknown in reality and must be treated as placeholders.
PLACEHOLDER_KEYS: tuple[tuple[str, str], ...] = (
    ("company.cash_minimum", "Cmin default 0 RUB"),
    ("company.burn_monthly", "Monthly burn is unknown; default 200000 RUB"),
    ("company.days_per_month", "30-day month conversion"),
    ("company.discretionary_rd_share", "Share of burn eligible for FREEZE"),
    ("company.bridge_capital", "Optional t=0 bridge; default 0"),
    ("bank.approval_probability", "P(BANK_APPROVE | decision). Not automatic after checkpoint"),
    ("bank.reduced_limit_probability", "P(BANK_REDUCED_LIMIT | approve)"),
    ("bank.reduced_limit_factor", "Limit haircut if reduced"),
    ("bank.fin_stress_approval_decay", "Approval decay per pre-decision FIN event"),
    ("bank.decision_delay_distribution", "Days from checkpoint to decision"),
    ("bank.limit_distribution", "Random bank limit given approval"),
    ("bank.financing_cost.annual_rate", "Cost of drawn bank funds"),
    ("deal.close_probability", "P(close path | no block). Deal is NOT closed"),
    ("deal.failure_probability", "P(DEAL_FAIL path | no block)"),
    ("deal.sanctions_block_probability", "P(SANCTIONS_BLOCK) at t=0"),
    ("deal.legal_block_probability", "P(LEGAL_BLOCK) at t=0"),
    ("deal.counterparty_failure_probability", "P(COUNTERPARTY_FAILURE) at t=0"),
    ("deal.delay_distribution", "Days to close if on close path"),
    ("deal.geo_delay_probability", "P(extra delay | GEO event, pending deal)"),
    ("deal.geo_fail_probability", "P(fail | GEO event, pending deal)"),
    ("deal.tranches", "Staged working-capital schedule if close"),
    ("contract.daily_inflow", "No predictable contract cash-flow; default 0"),
    ("contract.unexpected_contract_probability_per_biz_event", "P(contract | BIZ event)"),
    ("revenue.unexpected_revenue_probability_per_biz_event", "P(revenue | BIZ event)"),
    ("hazards.SEC.p90", "90-day base P(at least one SEC event), Z=0, no contagion"),
    ("hazards.FIN.p90", "90-day base P(at least one FIN event)"),
    ("hazards.MACRO.p90", "90-day base P(at least one MACRO event)"),
    ("hazards.GEO.p90", "90-day base P(at least one GEO event)"),
    ("hazards.BIZ.p90", "90-day base P(at least one BIZ event)"),
    ("hazards.BLACK_SWAN.p_daily", "Separate daily black-swan probability"),
    ("hazards.BLACK_SWAN.pareto_xmin", "Pareto xmin RUB given black swan"),
    ("hazards.BLACK_SWAN.pareto_alpha", "Pareto alpha (tail index)"),
    ("hazards.*.exposure_rub", "RUB exposure used to scale ordinary shock loss"),
    ("hazards.*.severity_median", "LogNormal median multiplier on exposure | event"),
    ("hazards.*.severity_sigma", "LogNormal sigma of log-severity | event"),
    ("hazards.*.beta", "Sensitivity of hazard k to latent Z_t"),
    ("hazards.*.theta", "Hawkes excitation decay horizon (days)"),
    ("contagion.A", "Cross-category contagion matrix"),
    ("contagion.rho", "AR(1) coefficient of latent stress Z"),
    ("contagion.sigma", "Innovation scale of latent stress Z"),
    ("macro_pass_through.burn_kappa", "Burn inflation vs cumulative MACRO severity"),
    ("outcomes.material_financing_rub", "Inflow size that counts as financed"),
    ("outcomes.trap_runway_days", "Runway below which a survivor is TRAP"),
    ("outcomes.escape_runway_days", "Runway for ESCAPE given financing"),
    ("absurdity.weights", "Category weights in raw absurdity score"),
    ("absurdity.gamma", "Weight on number of simultaneous active crises"),
    ("policies.*.trigger", "Policy activation thresholds"),
    ("policies.EXIT_RU.one_off_cost", "One-off cost of RU-contour exit"),
    ("escape.target_survival", "Target P(SURVIVAL) for escape solver (80%)"),
)


def deep_merge(base: dict, overlay: Mapping[str, Any]) -> dict:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if k == "extends":
            continue
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        candidate = CONFIGS_DIR / path
        if candidate.exists():
            path = candidate
        elif not path.exists():
            path = CONFIGS_DIR / Path(path).name
    raw = load_yaml(path)
    extends = raw.get("extends")
    if extends:
        parent_path = (path.parent / extends).resolve()
        parent = load_config(parent_path)
        cfg = deep_merge(parent, raw)
    else:
        cfg = copy.deepcopy(raw)
    cfg["_config_path"] = str(path)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict) -> None:
    sim = cfg["simulation"]
    if int(sim["worlds"]) <= 0:
        raise ValueError("simulation.worlds must be positive")
    if int(sim["days"]) <= 0:
        raise ValueError("simulation.days must be positive")
    cats = list(cfg["contagion"]["categories"])
    if tuple(cats) != CATEGORIES:
        raise ValueError(f"contagion.categories must be {CATEGORIES}")
    A = cfg["contagion"]["A"]
    for src in CATEGORIES:
        row = A[src]
        if len(row) != len(CATEGORIES):
            raise ValueError(f"A[{src}] must have {len(CATEGORIES)} columns")
    p = float(cfg["bank"]["approval_probability"])
    if not (0.0 <= p <= 1.0):
        raise ValueError("bank.approval_probability must be in [0,1]")


def get_path(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_path(cfg: dict, dotted: str, value: Any) -> dict:
    """Return a deep-copied config with a dotted key replaced."""
    out = copy.deepcopy(cfg)
    parts = dotted.split(".")
    cur: Any = out
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    return out


def parse_date(value: str | dt.date | dt.datetime) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def checkpoint_day(cfg: Mapping[str, Any]) -> int:
    start = parse_date(cfg["simulation"]["start_date"])
    chk = parse_date(cfg["bank"]["checkpoint_date"])
    return (chk - start).days


def daily_burn(cfg: Mapping[str, Any]) -> float:
    return float(cfg["company"]["burn_monthly"]) / float(cfg["company"]["days_per_month"])


def assumption_table(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, note in PLACEHOLDER_KEYS:
        rows.append(
            {
                "key": key,
                "value": get_path(cfg, key, "<composite / wildcard>"),
                "tag": "ASSUMPTION / PLACEHOLDER",
                "note": note,
            }
        )
    return rows


def flatten_leaves(cfg: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    skip = {"_config_path", "disclaimer", "notes"}
    for k, v in cfg.items():
        if k in skip:
            continue
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.extend(flatten_leaves(v, key))
        else:
            out.append((key, v))
    return out
