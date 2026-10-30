"""Light-atom interaction: scattering rates, mean force and photon recoil kicks.

Model
-----
Two-level atoms in 2·D Gaussian beams (D = 2 or 3 dimensions).
Beam j (unit vector e_j) is scattered at the rate

    R_j = (Γ/2) · s_j / (1 + s_tot + (2 δ_j / Γ)²)

with
    s_j   = s0 · P_j · exp(−2 r⊥² / w²)              local saturation of beam j (P_j: relative power)
    s_tot = Σ_j s_j                                  all beams saturate the same transition
    δ_j   = δ − e_j · (k v + G ∘ (r − r_0))          laser detuning seen by the atom

where G_i = μ'B'_i/ħ is the Zeeman gradient along axis i and r_0 the zero of
the magnetic field.

The Doppler shift (k·v) makes the atom scatter more from the beam it moves
against, which gives friction. The Zeeman shift in the quadrupole field, with
σ± polarisations chosen accordingly, makes it scatter more from the beam
pointing back to the field zero, which gives the trap.
"""

import numpy as np

from . import constants as c
from .parameters import MOTParameters


def scattering_rates(positions: np.ndarray, velocities: np.ndarray, p: MOTParameters,
                     transmission: np.ndarray | None = None) -> np.ndarray:
    """Photon scattering rate [1/s] from each beam, shape (n_atoms, n_beams).

    `transmission` (same shape) is the fraction of each beam left after the
    atoms upstream have absorbed from it; by default the cloud is transparent.
    """
    directions = p.beam_directions
    r_along = positions @ directions.T
    v_along = velocities @ directions.T
    zeeman = ((positions - p.field_zero) * p.zeeman_gradients) @ directions.T
    r_perp_sq = np.sum(positions**2, axis=1, keepdims=True) - r_along**2

    s = p.saturation * p.beam_powers * np.exp(-2 * r_perp_sq / p.beam_waist**2)
    if transmission is not None:
        s = s * transmission
    s_total = s.sum(axis=1, keepdims=True)
    delta_eff = p.delta - c.WAVENUMBER * v_along - zeeman

    return lorentzian_rates(s, s_total, delta_eff)


def lorentzian_rates(s: np.ndarray, s_total: np.ndarray, delta_eff: np.ndarray) -> np.ndarray:
    """Two-level scattering rate (Γ/2) s / (1 + s_tot + (2δ/Γ)²) [1/s], element by element."""
    return 0.5 * c.GAMMA * s / (1 + s_total + (2 * delta_eff / c.GAMMA) ** 2)


def mean_force(positions: np.ndarray, velocities: np.ndarray, p: MOTParameters) -> np.ndarray:
    """Average radiation-pressure force [N], shape (n_atoms, dimensions). Gravity not included."""
    return c.HBAR * c.WAVENUMBER * scattering_rates(positions, velocities, p) @ p.beam_directions


def photon_kicks(rates: np.ndarray, dt: float, p: MOTParameters, rng: np.random.Generator) -> np.ndarray:
    """Random velocity change [m/s] from one time step of absorption and emission.

    Each beam's absorptions are Poisson distributed. Each absorbed photon kicks
    the atom by ħk along its beam and is re-emitted in an isotropic 3D direction,
    which gives a second kick (in 2D only its in-plane part matters). Both are
    random, so this one function produces the friction, the trap and the heating.
    """
    return recoil_kicks(rng.poisson(rates * dt), p.beam_directions, rng)


def recoil_kicks(n_absorbed: np.ndarray, directions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Velocity change [m/s] from n_absorbed[i, j] photons taken from beam j and re-emitted isotropically.

    `directions` has shape (n_beams, D); only the first D components of each emission are kept.
    """
    n_atoms, dims = n_absorbed.shape[0], directions.shape[1]
    kick = n_absorbed @ directions

    emitter = np.repeat(np.arange(n_atoms), n_absorbed.sum(axis=1))
    for axis, component in enumerate(_isotropic_unit_vectors(emitter.size, rng)[:dims]):
        kick[:, axis] += np.bincount(emitter, component, minlength=n_atoms)

    return c.RECOIL_VELOCITY * kick


def _isotropic_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    """n random directions, uniform on the sphere, shape (3, n)."""
    cos_theta = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2 * np.pi, n)
    sin_theta = np.sqrt(1 - cos_theta**2)
    return np.stack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta))
