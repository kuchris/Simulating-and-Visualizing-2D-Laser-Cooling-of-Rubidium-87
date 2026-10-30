"""Analytic predictions of Doppler theory, used to check the simulation.

All results linearise the mean force near the trap centre (k|v| ≪ Γ, small
Zeeman shift), where along each axis i

    F_i = −β v_i − κ_i x_i.

Quantities that differ between axes are returned as arrays of length D.
"""

import numpy as np

from . import constants as c
from .parameters import MOTParameters

EMISSION_FRACTION = 1 / 3  # <ê_i²> for isotropic spontaneous emission in 3D


def _n_beams(p: MOTParameters) -> int:
    return 2 * p.dimensions


def _denominator(p: MOTParameters) -> float:
    """1 + s_tot + (2δ/Γ)² at the trap centre."""
    return 1 + _n_beams(p) * p.saturation + (2 * p.detuning) ** 2


def scattering_rate_per_beam(p: MOTParameters) -> float:
    """Scattering rate R from one beam for an atom at rest at the centre [1/s]."""
    return 0.5 * c.GAMMA * p.saturation / _denominator(p)


def damping_coefficient(p: MOTParameters) -> float:
    """Friction coefficient β [kg/s]: β = −8ħk² s0 (δ/Γ) / (1 + s_tot + (2δ/Γ)²)²."""
    return -8 * c.HBAR * c.WAVENUMBER**2 * p.saturation * p.detuning / _denominator(p) ** 2


def spring_constants(p: MOTParameters) -> np.ndarray:
    """Spring constant along each axis, κ_i = β μ'B'_i / (ħk) [N/m]. Negative for an anti-trap."""
    return damping_coefficient(p) * p.zeeman_gradients / c.WAVENUMBER


def velocity_damping_time(p: MOTParameters) -> float:
    """Time m/β for the velocity to relax [s]."""
    return c.MASS / damping_coefficient(p)


def trap_frequencies(p: MOTParameters) -> np.ndarray:
    """Trap frequency ω_i/2π along each axis [Hz]."""
    return np.sqrt(spring_constants(p) / c.MASS) / (2 * np.pi)


def damping_ratios(p: MOTParameters) -> np.ndarray:
    """ζ_i = β / (2√(κ_i m)); ζ < 1 means atoms oscillate, ζ > 1 means overdamped."""
    return damping_coefficient(p) / (2 * np.sqrt(spring_constants(p) * c.MASS))


def momentum_diffusion(p: MOTParameters) -> float:
    """Momentum diffusion constant D_p along one axis [kg² m² s⁻³].

    Random kicks of size ħk raise <p_i²> at the rate 2 D_p, from
      - absorption from the 2 beams on that axis:      2 R
      - emission of all 2D·R photons, projected:       2D·R · <ê_i²>
    In 3D this is 4R, the same as the textbook 1D result.
    """
    rate = scattering_rate_per_beam(p)
    return 0.5 * (c.HBAR * c.WAVENUMBER) ** 2 * rate * (2 + _n_beams(p) * EMISSION_FRACTION)


def doppler_temperature(p: MOTParameters) -> float:
    """Steady-state temperature k_B T = D_p / β, the same along every axis [K]."""
    return momentum_diffusion(p) / damping_coefficient(p) / c.K_B


def cloud_sizes(p: MOTParameters) -> np.ndarray:
    """RMS cloud size along each axis, σ_i = √(k_B T / κ_i) [m]."""
    return np.sqrt(c.K_B * doppler_temperature(p) / spring_constants(p))


def optimal_detuning(p: MOTParameters) -> float:
    """Detuning (in Γ) that minimises the Doppler temperature: δ = −√(1 + s_tot) / 2."""
    return -0.5 * np.sqrt(1 + _n_beams(p) * p.saturation)


def ideal_doppler_limit() -> float:
    """Textbook Doppler limit ħΓ / 2k_B (δ = −Γ/2, s → 0) [K]."""
    return c.HBAR * c.GAMMA / (2 * c.K_B)


# --- Imperfections ----------------------------------------------------------------

def equilibrium_position(p: MOTParameters) -> np.ndarray:
    """Centre of the cloud, where the mean force balances gravity [m].

    Three effects move it away from the origin:
      - a stray field moves the field zero to r_0
      - unequal ±x beams leave a net push 2εħkR along x
      - gravity pulls the cloud down by mg/κ_z
    """
    kappa = spring_constants(p)
    push = np.zeros(p.dimensions)
    push[0] = 2 * p.power_imbalance * c.HBAR * c.WAVENUMBER * scattering_rate_per_beam(p)
    return p.field_zero + (push + c.MASS * p.gravity_vector) / kappa


def escape_rates(p: MOTParameters) -> np.ndarray:
    """Growth rate λ_i of the cloud size in an anti-trap (κ_i < 0) [1/s].

    From m ẍ + β ẋ − |κ| x = 0:  λ = −γ/2 + √(γ²/4 + |κ|/m), with γ = β/m.
    """
    gamma = damping_coefficient(p) / c.MASS
    return -gamma / 2 + np.sqrt(gamma**2 / 4 - spring_constants(p) / c.MASS)


def expanded_size(sigma0, temperature, t):
    """Cloud size after free expansion for a time t: σ² = σ0² + (k_B T / m) t² [m]."""
    return np.sqrt(sigma0**2 + c.K_B * temperature / c.MASS * t**2)


def relaxation_rate(p: MOTParameters) -> float:
    """Slowest rate at which the cloud's position settles, over all axes [1/s].

    Solutions of m ẍ + β ẋ + κ x = 0 decay as exp(−λt) with λ = γ/2 − Re√(γ²/4 − ω²).
    """
    gamma = damping_coefficient(p) / c.MASS
    omega_sq = spring_constants(p) / c.MASS
    return float(np.min(gamma / 2 - np.sqrt(np.maximum(gamma**2 / 4 - omega_sq, 0))))
