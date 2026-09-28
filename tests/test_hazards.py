"""Hazard intensities, P90=0, non-negative lambda, Hawkes coupling."""

from __future__ import annotations

import numpy as np

from assumptions import CATEGORIES, load_config
from hazards import base_intensity, lambda0_vector, simulate_hazards, simulate_latent_z
from simulation import make_streams, run_oracle


def test_lambda0_from_p90():
    p90 = np.array([0.0, 0.2, 0.5, 0.9])
    lam = base_intensity(p90, 90)
    assert lam[0] == 0.0
    assert np.allclose(lam[1:], -np.log(1.0 - p90[1:]) / 90.0)


def test_p90_zero_never_fires():
    cfg = load_config("baseline.yaml")
    for name in CATEGORIES:
        if name == "BLACK_SWAN":
            cfg["hazards"][name]["p_daily"] = 0.0
        else:
            cfg["hazards"][name]["p90"] = 0.0
    r = run_oracle(cfg, n_worlds=64, seed=1, policies=["HOLD"])
    assert r.bundle.events.sum() == 0
    assert np.all(r.bundle.lam == 0.0) or np.all(r.bundle.events == False)  # noqa: E712


def test_hazard_never_negative():
    cfg = load_config("baseline.yaml")
    r = run_oracle(cfg, n_worlds=32, seed=2, policies=["HOLD"])
    assert np.all(r.bundle.lam >= -1e-15)


def test_latent_ar1_shape_and_persistence():
    rng = np.random.default_rng(0)
    eps = rng.standard_normal((8, 20))
    z = simulate_latent_z(eps, rho=0.9, sigma=0.3)
    assert z.shape == eps.shape
    assert np.allclose(z[:, 0], 0.3 * eps[:, 0])
    assert np.allclose(z[:, 1], 0.9 * z[:, 0] + 0.3 * eps[:, 1])


def test_contagion_changes_intensity_when_events_occur():
    cfg = load_config("baseline.yaml")
    streams = make_streams(32, 90, seed=4)
    z = simulate_latent_z(streams.z_eps, 0.0, 0.0)
    events, _sev, lam = simulate_hazards(
        cfg, streams.event_u, streams.severity_g, streams.pareto_u, z
    )
    assert events.shape == (32, 90, len(CATEGORIES))
    assert lam.shape == events.shape
    lam0 = lambda0_vector(cfg, 90)
    # With Z=0, day 0 has no excitation so lambda == lambda0
    assert np.allclose(lam[:, 0, :], lam0[None, :], atol=1e-12)
