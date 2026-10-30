"""Analytic predictions of Doppler theory, used to check the simulation.

All results linearise the mean force near the centre (k|v| ≪ Γ, small
Zeeman shift), where along each axis

    F = −β v − κ x.
"""

import numpy as np

from . import constants as c
from .parameters import MOTParameters

N_BEAMS = 4
EMISSION_FRACTION = 1 / 3  # <ê_x²> for isotropic spontaneous emission in 3D


def _denominator(p: MOTParameters) -> float:
    """1 + s_tot + (2δ/Γ)² at the trap centre."""
    return 1 + N_BEAMS * p.saturation + (2 * p.detuning) ** 2


def scattering_rate_per_beam(p: MOTParameters) -> float:
    """Scattering rate from one beam for an atom at rest at the centre [1/s]."""
    return 0.5 * c.GAMMA * p.saturation / _denominator(p)


def damping_coefficient(p: MOTParameters) -> float:
    """Friction coefficient β [kg/s]: β = −8ħk² s0 (δ/Γ) / (1 + s_tot + (2δ/Γ)²)²."""
    return -8 * c.HBAR * c.WAVENUMBER**2 * p.saturation * p.detuning / _denominator(p) ** 2


def spring_constant(p: MOTParameters) -> float:
    """Trap spring constant κ = β μ'B' / (ħk) [N/m]."""
    return damping_coefficient(p) * p.zeeman_gradient / c.WAVENUMBER


def velocity_damping_time(p: MOTParameters) -> float:
    """Time m/β for the velocity to relax [s]."""
    return c.MASS / damping_coefficient(p)


def trap_frequency(p: MOTParameters) -> float:
    """Trap frequency ω/2π [Hz]."""
    return np.sqrt(spring_constant(p) / c.MASS) / (2 * np.pi)


def damping_ratio(p: MOTParameters) -> float:
    """ζ = β / (2√(κm)); ζ < 1 means atoms oscillate, ζ > 1 means overdamped."""
    return damping_coefficient(p) / (2 * np.sqrt(spring_constant(p) * c.MASS))


def momentum_diffusion(p: MOTParameters) -> float:
    """Momentum diffusion constant D_p along one axis [kg² m² s⁻³].

    Random kicks of size ħk raise <p_x²> at the rate 2 D_p, from
      - absorption from the 2 beams on that axis:   2 R
      - emission of all 4 R photons, projected:     4 R · <ê_x²>
    """
    rate = scattering_rate_per_beam(p)
    return 0.5 * (c.HBAR * c.WAVENUMBER) ** 2 * rate * (2 + N_BEAMS * EMISSION_FRACTION)


def doppler_temperature(p: MOTParameters) -> float:
    """Steady-state temperature k_B T = D_p / β, per axis [K]."""
    return momentum_diffusion(p) / damping_coefficient(p) / c.K_B


def cloud_size(p: MOTParameters) -> float:
    """RMS cloud radius along one axis, σ = √(k_B T / κ) [m]."""
    return np.sqrt(c.K_B * doppler_temperature(p) / spring_constant(p))


def optimal_detuning(saturation: float) -> float:
    """Detuning (in Γ) that minimises the Doppler temperature: δ = −√(1 + s_tot) / 2."""
    return -0.5 * np.sqrt(1 + N_BEAMS * saturation)


def ideal_doppler_limit() -> float:
    """Textbook 1D Doppler limit ħΓ / 2k_B (δ = −Γ/2, s → 0) [K]."""
    return c.HBAR * c.GAMMA / (2 * c.K_B)
