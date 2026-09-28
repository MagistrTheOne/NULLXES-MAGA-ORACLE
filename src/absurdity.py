"""Absurdity Index.

Raw world-day score:
    score = sum_k w_k * ActiveSeverity_k + gamma * n_active_crises

AI = 100 * F_baseline(score)

F_baseline(s) := empirical P(S < s) on the baseline HOLD daily scores
(left-continuous ECDF so that the mass at score=0 maps to AI=0).

Bands (config):
    0-20 NORMAL
    20-40 🗿
    40-60 БЛЯДЬ
    60-80 КАКОГО ХУЯ
    80-95 MGS
    95-100 ПОНЕДЕЛЬНИК
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from assumptions import CATEGORIES


def raw_score(
    cfg: Mapping,
    events: np.ndarray,
    severity: np.ndarray,
) -> np.ndarray:
    w = np.array(
        [float(cfg["absurdity"]["weights"][name]) for name in CATEGORIES],
        dtype=np.float64,
    )
    gamma = float(cfg["absurdity"]["gamma"])
    active_sev = np.where(events, severity, 0.0)
    n_active = events.sum(axis=2).astype(np.float64)
    return (active_sev * w[None, None, :]).sum(axis=2) + gamma * n_active


def ecdf_left(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """F(s) = P_emp(S < s) using a sorted reference sample."""
    ref = np.sort(np.asarray(reference, dtype=np.float64).ravel())
    q = np.asarray(query, dtype=np.float64)
    # searchsorted left: count of ref < s
    idx = np.searchsorted(ref, q.ravel(), side="left")
    n = max(ref.size, 1)
    return (idx.astype(np.float64) / n).reshape(q.shape)


def absurdity_index(
    score: np.ndarray,
    baseline_scores: np.ndarray,
) -> np.ndarray:
    f = ecdf_left(baseline_scores, score)
    return 100.0 * np.clip(f, 0.0, 1.0)


def band_label(cfg: Mapping, ai_value: float) -> str:
    for b in cfg["absurdity"]["bands"]:
        if float(b["lo"]) <= ai_value < float(b["hi"]):
            return str(b["label"])
    return str(cfg["absurdity"]["bands"][-1]["label"])


def summarize_ai(ai: np.ndarray, cfg: Mapping) -> dict:
    """ai: (N, T)"""
    daily_mean = ai.mean(axis=0)
    daily_p50 = np.quantile(ai, 0.50, axis=0)
    daily_p95 = np.quantile(ai, 0.95, axis=0)
    max_ai = ai.max(axis=1)
    terminal = float(np.median(max_ai))
    return {
        "daily_mean": daily_mean,
        "daily_p50": daily_p50,
        "daily_p95": daily_p95,
        "max_ai": max_ai,
        "median_max_ai": terminal,
        "mean_max_ai": float(max_ai.mean()),
        "p95_max_ai": float(np.quantile(max_ai, 0.95)),
        "median_max_band": band_label(cfg, terminal),
    }
