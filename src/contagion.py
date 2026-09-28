"""Contagion matrix helpers and Hawkes excitation recursion.

lambda_{k,t} = lambda_{k0} * exp(beta_k * Z_t) * (1 + sum_j A[j,k] * exc_{j,t})

exc_{j,t} = sum_{tau<t} I(event_j, tau) * exp(-(t-tau)/theta_j)

Recursion (applied at end of day t, for use on day t+1):
    exc <- decay * (exc + I_t)
    decay_j = exp(-1 / theta_j)
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from assumptions import CATEGORIES


def contagion_matrix(cfg: Mapping) -> np.ndarray:
    A_cfg = cfg["contagion"]["A"]
    k = len(CATEGORIES)
    A = np.zeros((k, k), dtype=np.float64)
    for i, src in enumerate(CATEGORIES):
        row = np.asarray(A_cfg[src], dtype=np.float64)
        if row.shape != (k,):
            raise ValueError(f"A[{src}] has shape {row.shape}, expected ({k},)")
        A[i, :] = row
    return A


def category_vector(cfg: Mapping, field: str, default: float | None = None) -> np.ndarray:
    out = np.zeros(len(CATEGORIES), dtype=np.float64)
    for i, name in enumerate(CATEGORIES):
        block = cfg["hazards"][name]
        if field not in block:
            if default is None:
                raise KeyError(f"hazards.{name}.{field}")
            out[i] = default
        else:
            out[i] = float(block[field])
    return out


def decay_factors(theta: np.ndarray) -> np.ndarray:
    th = np.maximum(np.asarray(theta, dtype=np.float64), 1e-12)
    return np.exp(-1.0 / th)


def update_excitation(exc: np.ndarray, events_t: np.ndarray, decay: np.ndarray) -> np.ndarray:
    """Advance excitation after observing today's events.

    exc: (N, K) excitation used at the start of today (tau < t)
    events_t: (N, K) bool/float events at t
    decay: (K,)
    """
    return decay[None, :] * (exc + events_t.astype(np.float64, copy=False))
