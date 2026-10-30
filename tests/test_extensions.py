"""Checks of the extensions: capture, repumper, density limit, 2D-MOT source and molasses."""

from dataclasses import replace

import numpy as np
import pytest

from mot import capture, density, molasses, repumper, source, theory
from mot import constants as c
from mot.parameters import MOTParameters
from mot.simulation import doppler_steady_state

P3 = MOTParameters(dimensions=3)


# --- Capture --------------------------------------------------------------------------

def test_slow_flux_fraction_grows_as_fourth_power():
    v = 0.01 * capture.most_probable_speed()
    assert capture.slow_flux_fraction(v) == pytest.approx(0.5 * 0.01**4, rel=1e-3)


def test_vapour_atoms_follow_the_effusive_flux():
    s = capture.CaptureSettings()
    x, v, speed = capture.sample_vapour_atoms(100_000, 40.0, P3, s, np.random.default_rng(0))
    assert np.linalg.norm(x, axis=1) == pytest.approx(s.start_radius)
    cos_inward = -np.einsum("ij,ij->i", x, v) / (s.start_radius * speed)
    assert cos_inward.min() > 0
    assert cos_inward.mean() == pytest.approx(2 / 3, rel=0.01)  # cos θ law
    expected = capture.slow_flux_fraction(20.0) / capture.slow_flux_fraction(40.0)
    assert np.mean(speed < 20.0) == pytest.approx(expected, rel=0.03)


def test_photon_noise_barely_moves_the_capture_threshold():
    v_c = capture.deterministic_capture_velocity(P3)
    probability = capture.capture_probability(P3, [0.85 * v_c, 1.15 * v_c], n_each=40)
    assert probability[0] > 0.9
    assert probability[1] < 0.1


# --- Repumper -------------------------------------------------------------------------

def test_leak_probability_is_dominated_by_f2_prime():
    s_tot = 6 * P3.saturation
    via_f2 = (1 / 4) / (7 / 10) * (1 + s_tot + (2 * P3.detuning) ** 2) \
        / (1 + s_tot + (2 * (P3.detuning + repumper.SPLITTING_3_2 / c.GAMMA)) ** 2) / 2
    assert repumper.leak_probability(P3) == pytest.approx(via_f2, rel=0.15)
    assert repumper.leak_probability(P3) > via_f2


def test_without_repumper_atoms_go_dark_at_the_leak_rate():
    t = 0.6e-3
    result = repumper.run(P3, 0.0, n_atoms=3000, duration=t, snapshot_interval=t)
    rate = -np.log(result.bright_fraction()[-1]) / t
    assert rate == pytest.approx(repumper.leak_rate(P3), rel=0.05)


def test_bright_fraction_matches_rate_balance():
    result = repumper.run(P3, 1e-3, n_atoms=400, duration=3e-3)
    assert result.bright_fraction()[20:].mean() == pytest.approx(repumper.bright_fraction(P3, 1e-3), rel=0.05)


# --- Density limit --------------------------------------------------------------------

def test_shape_factors_add_up_to_one():
    assert density.depolarisation_factors(P3).sum() == pytest.approx(1)
    assert density._axial_factor(1.0) == pytest.approx(1 / 3)
    assert density._axial_factor(0.999) == pytest.approx(1 / 3, rel=1e-2)
    assert density._axial_factor(1.001) == pytest.approx(1 / 3, rel=1e-2)


def test_density_limited_cloud_holds_n_atoms_at_n_max():
    n = 1e8
    semi_axes = np.sqrt(5) * density.density_limited_sizes(P3, n)
    assert 4 / 3 * np.pi * np.prod(semi_axes) * density.density_limit(P3) == pytest.approx(n)


def _uniform_ball(n_points, radius, rng):
    direction = rng.normal(size=(n_points, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return radius * rng.random(n_points)[:, None] ** (1 / 3) * direction


def test_mean_field_obeys_gauss_and_beer_lambert():
    n_atoms, radius = 1e8, 0.8e-3
    x = _uniform_ball(100_000, radius, np.random.default_rng(1))
    field = density.MeanField(P3, n_atoms, len(x), density.RESCATTER_RATIO)
    rate = 1e6
    field.update(x, np.full(len(x), rate))
    n = n_atoms / (4 / 3 * np.pi * radius**3)

    probe = np.array([[0.5 * radius, 0, 0], [0, 0, -0.5 * radius]])
    # Inside a uniform ball, a = ħk σ_R R n r / (3m), pointing outwards.
    expected = c.HBAR * c.WAVENUMBER * field.sigma_r * rate * n * 0.5 * radius / (3 * c.MASS)
    a = field.rescattering_acceleration(probe)
    assert a[0, 0] == pytest.approx(expected, rel=0.1)
    assert a[1, 2] == pytest.approx(-expected, rel=0.1)

    # The +x beam reaching the centre has crossed one radius of the cloud.
    centre = field.transmission(np.zeros((1, 3)))[0]
    assert centre[0] == pytest.approx(np.exp(-field.sigma_l * n * radius), rel=0.05)
    assert centre == pytest.approx(np.full(6, centre[0]), rel=0.05)


def test_rescattering_alone_leaves_the_doppler_temperature():
    s = density.DensitySettings(n_sim=400, duration=5e-3, shadows=False,
                                rescatter_ratio=density.RESCATTER_RATIO - 1)
    steady = density.run(P3, 1e8, s).steady_state()
    assert steady.temperature().mean() == pytest.approx(theory.doppler_temperature(P3), rel=0.15)
    assert density.central_density(steady, 1e8) == pytest.approx(density.density_limit(P3), rel=0.3)


def test_without_two_body_loss_the_steady_number_is_rate_over_loss():
    assert density.steady_atom_number(P3, 1e8, 2.0, beta=0.0) == pytest.approx(5e7, rel=1e-6)
    assert density.steady_atom_number(P3, 1e8, 2.0) < 5e7


# --- 2D-MOT source --------------------------------------------------------------------

def test_source_atoms_enter_the_box_from_its_surface():
    sp = source.SourceParameters()
    x, v = source.sample_entering_atoms(sp, 20_000, 50.0, np.random.default_rng(2))
    half = np.array([sp.box_half_width, sp.box_half_width, sp.box_half_length])
    on_face = np.isclose(np.abs(x), half)
    assert np.all(on_face.sum(axis=1) >= 1)
    face_axis = np.argmax(on_face, axis=1)
    rows = np.arange(len(x))
    assert np.all(x[rows, face_axis] * v[rows, face_axis] < 0)  # moving inwards
    assert np.all(np.linalg.norm(v, axis=1) < 50.0)


def test_stronger_push_gives_faster_atoms():
    sp = source.SourceParameters()
    speeds = [source.pushed_velocity(replace(sp, push_saturation=s)) for s in (0.2, 0.5, 1.5)]
    assert speeds == sorted(speeds)
    assert source.pushed_velocity(replace(sp, push_saturation=0.0)) == 0.0


# --- Molasses -------------------------------------------------------------------------

def test_clebsch_gordan_coefficients():
    assert molasses.clebsch_gordan(2, 2, 1, 1, 3, 3) == pytest.approx(1)
    assert molasses.clebsch_gordan(2, 0, 1, 0, 3, 0) == pytest.approx(np.sqrt(3 / 5))
    for m in range(-3, 4):
        total = sum(molasses.clebsch_gordan(2, m - q, 1, q, 3, m) ** 2 for q in (-1, 0, 1))
        assert total == pytest.approx(1)  # every excited state decays with rate Γ


def test_steady_state_is_a_density_matrix():
    rho = molasses.steady_state(0.1, -3.0, 0.5)
    assert np.trace(rho).real == pytest.approx(1)
    assert np.allclose(rho, rho.conj().T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(rho) > -1e-10)


def test_fast_atoms_are_pumped_onto_the_stretched_transition():
    detuning, s, v = -2.0, 0.1, 8.0
    force, _ = molasses.force_and_rate([v], detuning, s)
    kv = c.WAVENUMBER * v / c.GAMMA
    stretched = -c.HBAR * c.WAVENUMBER * 0.5 * c.GAMMA * s / (1 + s + 4 * (detuning + kv) ** 2)
    assert force[0] == pytest.approx(stretched, rel=0.03)


def test_sub_doppler_friction_does_not_depend_on_intensity():
    weak = molasses.ForceTable.build(-6.0, 0.05, n=81).friction()
    strong = molasses.ForceTable.build(-6.0, 0.3, n=81).friction()
    assert weak == pytest.approx(strong, rel=0.05)
    doppler = -np.gradient(molasses.doppler_force([-1e-3, 0, 1e-3], -6.0, 0.3), 1e-3)[1]
    assert strong > 100 * doppler


def test_sub_doppler_temperature_scales_as_intensity_over_detuning():
    a = molasses.ForceTable.build(-3.0, 0.6, n=81).core_temperature()
    b = molasses.ForceTable.build(-6.0, 1.2, n=81).core_temperature()
    c_ = molasses.ForceTable.build(-6.0, 2.4, n=81).core_temperature()
    assert a == pytest.approx(b, rel=0.05)
    assert c_ == pytest.approx(2 * b, rel=0.05)


def test_molasses_cools_the_mot_far_below_the_doppler_limit():
    x, v = doppler_steady_state(P3, 300, np.random.default_rng(3))
    table = molasses.ForceTable.build(-6.0, 3.0)
    final = molasses.run([(table, 1.5e-3)], x, v, snapshot_interval=0.5e-3).velocities[-1]
    sigma = 1.4826 * np.median(np.abs(final - np.median(final)))
    core = c.MASS * sigma**2 / c.K_B
    assert core < 0.05 * theory.doppler_temperature(P3)
    assert core == pytest.approx(table.core_temperature(), rel=0.35)
