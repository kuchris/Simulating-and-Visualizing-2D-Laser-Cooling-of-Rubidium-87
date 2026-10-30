"""Check the Monte-Carlo model against the analytic Doppler theory, in 2D and 3D."""

from dataclasses import replace

import numpy as np
import pytest

from mot import constants as c
from mot import theory
from mot.measurement import absorption_image, time_of_flight
from mot.parameters import MOTParameters
from mot.physics import mean_force, photon_kicks, scattering_rates
from mot.simulation import SimulationResult, SimulationSettings, initial_state, run

P3 = MOTParameters(dimensions=3)
P2 = MOTParameters(dimensions=2)
BOTH = pytest.mark.parametrize("p", [P2, P3], ids=["2D", "3D"])


def _force(p, axis, x=0.0, v=0.0, origin=None):
    """Mean force along `axis` for an atom displaced by x and moving at v along that axis."""
    r = np.zeros((1, p.dimensions)) if origin is None else np.array([origin], dtype=float)
    u = np.zeros((1, p.dimensions))
    r[0, axis] += x
    u[0, axis] += v
    return mean_force(r, u, p)[0, axis]


@BOTH
def test_no_force_at_rest_in_the_centre(p):
    assert mean_force(np.zeros((1, p.dimensions)), np.zeros((1, p.dimensions)), p) == pytest.approx(0, abs=1e-30)


@BOTH
def test_friction_matches_analytic_damping_coefficient(p):
    dv = 1e-4
    for axis in range(p.dimensions):
        beta = -(_force(p, axis, v=dv) - _force(p, axis, v=-dv)) / (2 * dv)
        assert beta == pytest.approx(theory.damping_coefficient(p), rel=1e-6)


@BOTH
def test_restoring_force_matches_analytic_spring_constants(p):
    dx = 1e-7
    kappa = [-(_force(p, i, x=dx) - _force(p, i, x=-dx)) / (2 * dx) for i in range(p.dimensions)]
    assert kappa == pytest.approx(theory.spring_constants(p), rel=1e-6)


def test_axial_trap_is_twice_as_stiff_in_3d():
    kx, ky, kz = theory.spring_constants(P3)
    assert kx == pytest.approx(ky)
    assert kz == pytest.approx(2 * kx)


def test_3d_temperature_is_the_textbook_doppler_formula():
    x = 2 * abs(P3.detuning)
    textbook = c.HBAR * c.GAMMA / 4 * (1 + 6 * P3.saturation + x**2) / x / c.K_B
    assert theory.doppler_temperature(P3) == pytest.approx(textbook, rel=1e-12)


@BOTH
def test_photon_kicks_average_to_mean_force(p):
    rng = np.random.default_rng(0)
    n = 200_000
    x = np.zeros((n, p.dimensions))
    v = np.tile(np.linspace(0.5, -0.3, p.dimensions), (n, 1))
    dt = 1e-6
    measured = photon_kicks(scattering_rates(x, v, p), dt, p, rng).mean(axis=0) * c.MASS / dt
    assert measured == pytest.approx(mean_force(x[:1], v[:1], p)[0], rel=0.05)


@BOTH
def test_initial_state_has_requested_temperature_and_radius(p):
    s = SimulationSettings(n_atoms=200_000, initial_temperature=10e-3, initial_radius=2e-3)
    x, v = initial_state(s, p.dimensions, np.random.default_rng(0))
    assert c.MASS * v.var(axis=0) / c.K_B == pytest.approx([10e-3] * p.dimensions, rel=0.01)
    assert np.linalg.norm(x, axis=1).max() <= 2e-3


@BOTH
def test_steady_state_reaches_doppler_temperature_and_cloud_size(p):
    p = replace(p, detuning=-1.0)
    s = SimulationSettings(n_atoms=500, duration=5e-3, initial_temperature=1e-3, initial_radius=0.2e-3)
    steady = run(p, s).steady_state()
    assert steady.temperature().mean() == pytest.approx(theory.doppler_temperature(p), rel=0.08)
    assert steady.rms_size().mean(axis=0) == pytest.approx(theory.cloud_sizes(p), rel=0.1)


# --- Imperfections: the mean force vanishes where theory puts the cloud centre ---------

def _balance_point(p, axis):
    """Solve F_axis(x) + m g_axis = 0 by Newton iteration."""
    x, dx = 0.0, 1e-7
    for _ in range(20):
        f = _force(p, axis, x=x) + c.MASS * p.gravity_vector[axis]
        slope = (_force(p, axis, x=x + dx) - _force(p, axis, x=x - dx)) / (2 * dx)
        x -= f / slope
    return x


def test_gravity_sag():
    assert _balance_point(P3, 2) == pytest.approx(theory.equilibrium_position(P3)[2], rel=1e-3)


def test_power_imbalance_shift():
    p = replace(P3, power_imbalance=0.01)
    assert _balance_point(p, 0) == pytest.approx(theory.equilibrium_position(p)[0], rel=0.01)


def test_stray_field_moves_the_trap_centre():
    p = replace(P3, stray_field=(0.1, 0.0, 0.0), gravity=False)
    assert _balance_point(p, 0) == pytest.approx(theory.equilibrium_position(p)[0], rel=1e-6)


def test_reversed_field_is_an_anti_trap():
    p = replace(P3, reversed_field=True)
    assert np.all(theory.spring_constants(p) < 0)
    assert _force(p, 0, x=1e-4) > 0


# --- Measurement ----------------------------------------------------------------------

def _thermal_cloud(p, temperature, sigma, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, sigma, (1, n, p.dimensions))
    v = rng.normal(0, np.sqrt(c.K_B * temperature / c.MASS), (1, n, p.dimensions))
    return SimulationResult(p, SimulationSettings(n_atoms=n), np.zeros(1), x, v)


def test_time_of_flight_recovers_temperature_and_size():
    tof = time_of_flight(_thermal_cloud(P3, 300e-6, 200e-6), np.linspace(0, 4e-3, 9))
    assert tof.temperature == pytest.approx([300e-6] * 3, rel=0.02)
    assert tof.initial_size == pytest.approx([200e-6] * 3, rel=0.02)
    assert tof.centers[-1, 2] == pytest.approx(-0.5 * c.G_EARTH * 4e-3**2, abs=5e-6)


def test_absorption_image_counts_every_atom():
    cloud = _thermal_cloud(P3, 300e-6, 200e-6).positions[0]
    image = absorption_image(cloud, n_atoms=1e6, half_width=2e-3)
    assert image.atom_number() == pytest.approx(1e6, rel=0.01)
    peak = 1e6 / (2 * np.pi * (200e-6) ** 2) * c.RESONANT_CROSS_SECTION
    assert image.optical_depth.max() == pytest.approx(peak, rel=0.1)
