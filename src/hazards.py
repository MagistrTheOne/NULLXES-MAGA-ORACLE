"""Hazard intensities, latent factor, daily events, severities.

Base intensity from a 90-day probability (Z=0, no contagion):

    lambda_k0 = -ln(1 - P90_k) / 90

BLACK_SWAN is a separate process parameterized by a daily probability:

    lambda_bs0 = -ln(1 - p_daily)

Daily event probability:

    P(E_{k,t}) = 1 - exp(-lambda_{k,t})

Ordinary severity | event ~ LogNormal.
BLACK_SWAN severity | event ~ Pareto Type I (heavy tail). Mean is not a target metric.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from assumptions import CATEGORIES, REGULAR_CATEGORIES
from contagion import category_vector, contagion_matrix, decay_factors, update_excitation


def base_intensity(p90: np.ndarray, days: int) -> np.ndarray:
    p = np.clip(np.asarray(p90, dtype=np.float64), 0.0, 1.0 - 1e-15)
    with np.errstate(divide="ignore"):
        lam = -np.log1p(-p) / float(days)
    lam = np.where(p <= 0.0, 0.0, lam)
    return lam


def black_swan_base_intensity(p_daily: float) -> float:
    p = float(np.clip(p_daily, 0.0, 1.0 - 1e-15))
    if p <= 0.0:
        return 0.0
    return float(-np.log1p(-p))


def simulate_latent_z(eps: np.ndarray, rho: float, sigma: float) -> np.ndarray:
    """Z_t = rho Z_{t-1} + sigma * eps_t, Z shape (N, T)."""
    n, t = eps.shape
    z = np.empty((n, t), dtype=np.float64)
    z[:, 0] = sigma * eps[:, 0]
    r = float(rho)
    s = float(sigma)
    for i in range(1, t):
        z[:, i] = r * z[:, i - 1] + s * eps[:, i]
    return z


def p90_vector(cfg: Mapping, days: int) -> np.ndarray:
    p = np.zeros(len(CATEGORIES), dtype=np.float64)
    for i, name in enumerate(CATEGORIES):
        h = cfg["hazards"][name]
        if name == "BLACK_SWAN":
            p_daily = float(h["p_daily"])
            p[i] = 1.0 - (1.0 - np.clip(p_daily, 0.0, 1.0)) ** days
        else:
            p[i] = float(h["p90"])
    return np.clip(p, 0.0, 1.0 - 1e-15)


def lambda0_vector(cfg: Mapping, days: int) -> np.ndarray:
    lam = np.zeros(len(CATEGORIES), dtype=np.float64)
    for i, name in enumerate(CATEGORIES):
        h = cfg["hazards"][name]
        if name == "BLACK_SWAN":
            lam[i] = black_swan_base_intensity(float(h["p_daily"]))
        else:
            lam[i] = base_intensity(np.array([h["p90"]]), days)[0]
    return lam


def simulate_hazards(
    cfg: Mapping,
    event_u: np.ndarray,
    severity_g: np.ndarray,
    pareto_u: np.ndarray,
    z: np.ndarray,
    overlay=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized across worlds; Python loop only over days.

    overlay: optional HazardOverlay (live events). None = no overlay.
    """
    n, t_days, k = event_u.shape
    assert z.shape == (n, t_days)
    assert severity_g.shape == (n, t_days, k)

    lam0 = lambda0_vector(cfg, t_days)
    beta = category_vector(cfg, "beta")
    theta = category_vector(cfg, "theta")
    decay = decay_factors(theta)
    a = contagion_matrix(cfg)
    sev_mult = float(cfg["simulation"].get("shock_severity_multiplier", 1.0))

    log_med = np.log(np.maximum(category_vector(cfg, "severity_median"), 1e-18))
    sig = np.maximum(category_vector(cfg, "severity_sigma"), 0.0)

    bs_idx = CATEGORIES.index("BLACK_SWAN")
    xmin = float(cfg["hazards"]["BLACK_SWAN"]["pareto_xmin"])
    alpha = float(cfg["hazards"]["BLACK_SWAN"]["pareto_alpha"])
    inv_alpha = 1.0 / max(alpha, 1e-12)

    events = np.zeros((n, t_days, k), dtype=np.bool_)
    severity = np.zeros((n, t_days, k), dtype=np.float64)
    lam_path = np.zeros((n, t_days, k), dtype=np.float64)
    exc = np.zeros((n, k), dtype=np.float64)

    z_use = z
    if overlay is not None and getattr(overlay, "z_add", None) is not None:
        z_use = z + overlay.z_add[None, :]

    for t in range(t_days):
        boost = exc @ a
        scale = np.maximum(1.0 + boost, 0.0)
        z_t = z_use[:, t][:, None]
        lam = lam0[None, :] * np.exp(beta[None, :] * z_t) * scale
        if overlay is not None:
            lam = lam + overlay.lambda_add[t][None, :]
        np.maximum(lam, 0.0, out=lam)
        lam_path[:, t, :] = lam
        p = 1.0 - np.exp(-lam)
        if overlay is not None:
            p = p + overlay.p_boost[t][None, :]
        np.clip(p, 0.0, 1.0, out=p)
        ev = event_u[:, t, :] < p
        if overlay is not None:
            ev = ev | overlay.force_event[t][None, :]
        events[:, t, :] = ev

        sev = np.exp(log_med[None, :] + sig[None, :] * severity_g[:, t, :])
        u = np.clip(pareto_u[:, t], 1e-12, 1.0 - 1e-12)
        sev[:, bs_idx] = xmin * np.power(u, -inv_alpha)
        sev *= sev_mult
        severity[:, t, :] = np.where(ev, sev, 0.0)

        exc = update_excitation(exc, ev, decay)

    return events, severity, lam_path


def shock_loss_rub(cfg: Mapping, events: np.ndarray, severity: np.ndarray) -> np.ndarray:
    """(N, T) RUB losses. Ordinary: exposure * lognormal. Swan: Pareto RUB * exposure (1)."""
    exposure = category_vector(cfg, "exposure_rub")
    loss_kt = events.astype(np.float64) * severity * exposure[None, None, :]
    return loss_kt.sum(axis=2)


def active_severity(events: np.ndarray, severity: np.ndarray) -> np.ndarray:
    """ActiveSeverity_k,t = severity on event days, else 0. Shape (N,T,K)."""
    return np.where(events, severity, 0.0)


def n_active_crises(events: np.ndarray) -> np.ndarray:
    return events.sum(axis=2).astype(np.float64)


def implied_p90_note() -> str:
    return (
        "P90_k is an ASSUMPTION for the 90-day at-least-one-event probability "
        "under lambda_k0 only (Z=0, A=0). Realized frequencies in the simulation "
        "differ because of the latent factor and Hawkes contagion."
    )
