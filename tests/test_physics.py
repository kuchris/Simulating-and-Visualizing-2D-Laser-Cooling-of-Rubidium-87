"""Check the Monte-Carlo model against the analytic Doppler theory."""

import numpy as np
import pytest

from mot2d import constants as c
from mot2d import theory
from mot2d.parameters import MOTParameters
from mot2d.physics import mean_force, photon_kicks, scattering_rates
from mot2d.simulation import SimulationSettings, initial_state, run

P = MOTParameters()


def _force_x(x=0.0, vx=0.0):
    return mean_force(np.array([[x, 0.0]]), np.array([[vx, 0.0]]), P)[0, 0]


def test_no_force_at_rest_in_the_centre():
    assert _force_x() == pytest.approx(0.0, abs=1e-30)


def test_friction_matches_analytic_damping_coefficient():
    dv = 1e-4
    beta = -(_force_x(vx=dv) - _force_x(vx=-dv)) / (2 * dv)
    assert beta == pytest.approx(theory.damping_coefficient(P), rel=1e-6)


def test_restoring_force_matches_analytic_spring_constant():
    dx = 1e-7
    kappa = -(_force_x(x=dx) - _force_x(x=-dx)) / (2 * dx)
    assert kappa == pytest.approx(theory.spring_constant(P), rel=1e-6)


def test_scattering_rate_at_centre():
    rates = scattering_rates(np.zeros((1, 2)), np.zeros((1, 2)), P)
    assert rates == pytest.approx(theory.scattering_rate_per_beam(P), rel=1e-12)


def test_photon_kicks_average_to_mean_force():
    rng = np.random.default_rng(0)
    x = np.zeros((200_000, 2))
    v = np.tile([0.5, -0.3], (200_000, 1))
    dt = 1e-6
    measured = photon_kicks(scattering_rates(x, v, P), dt, rng).mean(axis=0) * c.MASS / dt
    expected = mean_force(x[:1], v[:1], P)[0]
    assert measured == pytest.approx(expected, rel=0.05)


def test_initial_velocities_have_the_requested_temperature():
    s = SimulationSettings(n_atoms=200_000, initial_temperature=10e-3)
    _, v = initial_state(s, np.random.default_rng(0))
    assert c.MASS * v.var(axis=0) / c.K_B == pytest.approx([10e-3, 10e-3], rel=0.01)


def test_steady_state_reaches_doppler_temperature_and_cloud_size():
    p = MOTParameters(detuning=-1.0)
    s = SimulationSettings(n_atoms=500, duration=4e-3, initial_temperature=1e-3, initial_radius=0.2e-3)
    steady = run(p, s).steady_state()
    assert steady.temperature().mean() == pytest.approx(theory.doppler_temperature(p), rel=0.08)
    assert steady.rms_size().mean() == pytest.approx(theory.cloud_size(p), rel=0.08)
