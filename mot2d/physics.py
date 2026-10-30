"""Light-atom interaction: scattering rates, mean force and photon recoil kicks.

Model
-----
Two-level atoms in four Gaussian beams travelling along +x, −x, +y and −y.
Beam j (unit vector e_j) is scattered at the rate

    R_j = (Γ/2) · s_j / (1 + s_tot + (2 δ_j / Γ)²)

with
    s_j   = s0 · exp(−2 r⊥² / w²)            local saturation of beam j
    s_tot = Σ_j s_j                          all beams saturate the same transition
    δ_j   = δ − e_j · (k v + (μ'B'/ħ) r)     laser detuning seen by the atom

The Doppler shift (k·v) makes the atom scatter more from the beam it moves
against, which gives friction. The Zeeman shift in the quadrupole field
(μ'B'/ħ · r), with σ± polarisations chosen accordingly, makes it scatter more
from the beam pointing back to the centre, which gives the trap. Both enter δ_j
the same way, so the MOT force depends only on v + (μ'B'/ħk) r.
"""

import numpy as np

from . import constants as c
from .parameters import MOTParameters

BEAM_DIRECTIONS = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


def scattering_rates(positions: np.ndarray, velocities: np.ndarray, p: MOTParameters) -> np.ndarray:
    """Photon scattering rate [1/s] from each beam, shape (n_atoms, 4)."""
    r_along = positions @ BEAM_DIRECTIONS.T
    v_along = velocities @ BEAM_DIRECTIONS.T
    r_perp_sq = np.sum(positions**2, axis=1, keepdims=True) - r_along**2

    s = p.saturation * np.exp(-2 * r_perp_sq / p.beam_waist**2)
    s_total = s.sum(axis=1, keepdims=True)
    delta_eff = p.delta - c.WAVENUMBER * v_along - p.zeeman_gradient * r_along

    return 0.5 * c.GAMMA * s / (1 + s_total + (2 * delta_eff / c.GAMMA) ** 2)


def mean_force(positions: np.ndarray, velocities: np.ndarray, p: MOTParameters) -> np.ndarray:
    """Average radiation-pressure force [N], shape (n_atoms, 2)."""
    return c.HBAR * c.WAVENUMBER * scattering_rates(positions, velocities, p) @ BEAM_DIRECTIONS


def photon_kicks(rates: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
    """Random velocity change [m/s] from one time step of absorption and emission.

    Each beam's absorptions are Poisson distributed. Each absorbed photon kicks
    the atom by ħk along its beam and is re-emitted in an isotropic 3D direction,
    whose projection onto the xy plane gives a second kick. Both are random, so
    this one function produces the friction, the trap and the heating.
    """
    n_atoms = rates.shape[0]
    n_absorbed = rng.poisson(rates * dt)
    kick = n_absorbed @ BEAM_DIRECTIONS

    n_emitted = n_absorbed.sum(axis=1)
    emitter = np.repeat(np.arange(n_atoms), n_emitted)
    cos_theta = rng.uniform(-1.0, 1.0, emitter.size)
    phi = rng.uniform(0.0, 2 * np.pi, emitter.size)
    sin_theta = np.sqrt(1 - cos_theta**2)
    kick[:, 0] += np.bincount(emitter, sin_theta * np.cos(phi), minlength=n_atoms)
    kick[:, 1] += np.bincount(emitter, sin_theta * np.sin(phi), minlength=n_atoms)

    return c.RECOIL_VELOCITY * kick
