"""One-at-a-time sensitivity with common random numbers.

Each spec replaces one parameter, reuses the baseline RNG streams, and
reports baseline survival, new survival, and delta.

Event-layer keys rebuild the WorldBundle from the same streams.
Financing-layer keys rebuild deal/bank on the same events.
Cash-layer keys only re-run the cash overlay.
"""

from __future__ import annotations

from typing import Any, Mapping

from assumptions import get_path, set_path
from financing import simulate_aux_inflows, simulate_bank, simulate_deal
from simulation import OracleResult, WorldBundle, run_oracle, survival_prob


def _kind(key: str) -> str:
    if key.startswith("hazards.") or key.startswith("contagion."):
        return "events"
    if "shock_severity_multiplier" in key:
        return "events"
    if key.startswith("bank.") or key.startswith("deal."):
        return "financing"
    return "cash"


def _eval_policy(spec_name: str) -> str:
    if "freeze" in spec_name.lower():
        return "FREEZE_90"
    return "HOLD"


def _rebuild_financing(cfg: Mapping, bundle: WorldBundle, streams) -> WorldBundle:
    events = bundle.events
    deal = simulate_deal(
        cfg,
        events,
        streams.deal_u_sanc,
        streams.deal_u_legal,
        streams.deal_u_cp,
        streams.deal_u_fail,
        streams.deal_u_close,
        streams.deal_u_delay,
        streams.deal_u_geo_delay,
        streams.deal_u_geo_fail,
    )
    bank = simulate_bank(
        cfg,
        events,
        streams.bank_u_approve,
        streams.bank_u_reduced,
        streams.bank_u_delay,
        streams.bank_u_limit_g,
        streams.bank_u_fin_delay,
    )
    revenue, contract = simulate_aux_inflows(
        cfg,
        events,
        streams.u_rev_kind,
        streams.u_contract_kind,
        streams.g_rev,
        streams.g_contract,
    )
    return WorldBundle(
        z=bundle.z,
        events=bundle.events,
        severity=bundle.severity,
        lam=bundle.lam,
        shock_loss=bundle.shock_loss,
        deal=deal,
        bank=bank,
        revenue=revenue,
        contract=contract,
        score=bundle.score,
    )


def evaluate_override(
    cfg: Mapping,
    baseline: OracleResult,
    dotted_key: str,
    value: Any,
    policy: str,
) -> float:
    cfg2 = set_path(cfg, dotted_key, value)
    k = _kind(dotted_key)
    streams = baseline.streams
    n = baseline.bundle.events.shape[0]
    seed = int(cfg["simulation"]["seed"])
    if k == "events":
        r = run_oracle(cfg2, n_worlds=n, seed=seed, policies=[policy], streams=streams)
        return survival_prob(r, policy)
    if k == "financing":
        bundle = _rebuild_financing(cfg2, baseline.bundle, streams)
        r = run_oracle(
            cfg2, n_worlds=n, seed=seed, policies=[policy], streams=streams, bundle=bundle
        )
        return survival_prob(r, policy)
    # cash-only: reuse bundle
    r = run_oracle(
        cfg2,
        n_worlds=n,
        seed=seed,
        policies=[policy],
        streams=streams,
        bundle=baseline.bundle,
    )
    return survival_prob(r, policy)


def run_sensitivity(cfg: Mapping, baseline: OracleResult) -> list[dict]:
    rows: list[dict] = []
    specs = list(cfg.get("sensitivity", {}).get("specs", []))
    for spec in specs:
        name = spec["name"]
        key = spec["key"]
        mode = spec["mode"]
        policy = _eval_policy(name)
        if policy not in baseline.metrics:
            # compute baseline under that policy with same streams
            b_s = survival_prob(
                run_oracle(
                    cfg,
                    n_worlds=baseline.bundle.events.shape[0],
                    seed=int(cfg["simulation"]["seed"]),
                    policies=[policy],
                    streams=baseline.streams,
                    bundle=baseline.bundle if _kind(key) != "events" else None,
                ),
                policy,
            )
        else:
            b_s = float(baseline.metrics[policy]["P(SURVIVAL)"])
        cur = get_path(cfg, key)
        if cur is None and key == "company.bridge_capital":
            cur = 0.0
        for side, alt in (("low", spec["low"]), ("high", spec["high"])):
            if mode == "multiplier":
                new_val = float(cur) * float(alt)
            else:
                new_val = alt
            new_s = evaluate_override(cfg, baseline, key, new_val, policy)
            rows.append(
                {
                    "parameter": name,
                    "key": key,
                    "side": side,
                    "policy": policy,
                    "baseline_value": cur,
                    "new_value": new_val,
                    "baseline_survival": b_s,
                    "new_survival": new_s,
                    "delta_survival": new_s - b_s,
                    "abs_delta": abs(new_s - b_s),
                    "tag": "ASSUMPTION / PLACEHOLDER",
                }
            )
    rows.sort(key=lambda r: r["abs_delta"], reverse=True)
    return rows


def tornado_table(rows: list[dict]) -> list[dict]:
    """One row per parameter: max |delta| across low/high."""
    by: dict[str, dict] = {}
    for r in rows:
        name = r["parameter"]
        cur = by.get(name)
        if cur is None or r["abs_delta"] > cur["abs_delta"]:
            by[name] = r
    ranked = sorted(by.values(), key=lambda r: r["abs_delta"], reverse=True)
    return ranked
