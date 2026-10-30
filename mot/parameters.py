"""Experimental parameters of the MOT, in 2D or 3D."""

from dataclasses import dataclass

import numpy as np

from . import constants as c

GAUSS_PER_CM = 1e-2  # 1 G/cm expressed in T/m
GAUSS = 1e-4         # 1 G expressed in T


@dataclass(frozen=True)
class MOTParameters:
    """The knobs of the experiment.

    Geometry
      3D: six beams along ±x, ±y, ±z and an anti-Helmholtz quadrupole field
          B = B'(x/2, y/2, −z). The axial gradient B' (along z) is twice the radial one.
      2D: four beams along ±x, ±y and a 2D quadrupole B = B'(x, −y); z is free.

    The last three fields are deliberate imperfections, all zero in a perfect MOT.
    """

    dimensions: int = 3
    detuning: float = -2.0          # laser detuning δ in units of Γ (negative = red)
    saturation: float = 0.5         # peak saturation parameter s0 = I/I_sat of *each* beam
    beam_waist: float = 5e-3        # 1/e² intensity radius w [m]
    field_gradient: float = 10.0    # strongest gradient B' [G/cm]: axial in 3D, in-plane in 2D
    magnetic_moment: float = 1.0    # μ' = (g_F' m_F' − g_F m_F) in units of μ_B; 1 for |2,2> -> |3',3'>
    gravity: bool = True            # gravity along −z (3D only)

    power_imbalance: float = 0.0    # ε: the +x beam carries s0(1+ε), the −x beam s0(1−ε)
    stray_field: tuple = (0.0, 0.0, 0.0)  # uniform bias field [G] along x, y, z
    reversed_field: bool = False    # coil current (or σ± handedness) reversed: an anti-trap

    def __post_init__(self):
        if self.dimensions not in (2, 3):
            raise ValueError("dimensions must be 2 or 3")

    @property
    def delta(self) -> float:
        """Detuning δ in rad/s."""
        return self.detuning * c.GAMMA

    @property
    def beam_directions(self) -> np.ndarray:
        """Unit vectors of the beams, shape (2·dimensions, dimensions): +x, −x, +y, −y[, +z, −z]."""
        axes = np.eye(self.dimensions)
        return np.stack([sign * axis for axis in axes for sign in (1.0, -1.0)])

    @property
    def beam_powers(self) -> np.ndarray:
        """Relative power of each beam."""
        powers = np.ones(2 * self.dimensions)
        powers[:2] = 1 + self.power_imbalance, 1 - self.power_imbalance
        return powers

    @property
    def axis_gradients(self) -> np.ndarray:
        """Magnitude of the field gradient along each axis [T/m]."""
        relative = np.ones(2) if self.dimensions == 2 else np.array([0.5, 0.5, 1.0])
        return self.field_gradient * GAUSS_PER_CM * relative

    @property
    def zeeman_gradients(self) -> np.ndarray:
        """Zeeman shift per unit displacement along each axis, μ'B'_i/ħ [rad s⁻¹ m⁻¹].

        Negative when the field is reversed: the shift then favours the beam pushing outwards.
        """
        sign = -1.0 if self.reversed_field else 1.0
        return sign * self.magnetic_moment * c.MU_B * self.axis_gradients / c.HBAR

    @property
    def field_zero(self) -> np.ndarray:
        """Where the stray field moves the zero of the quadrupole field [m]."""
        stray = np.asarray(self.stray_field[: self.dimensions], dtype=float) * GAUSS
        return -stray / self.axis_gradients

    @property
    def gravity_vector(self) -> np.ndarray:
        """Gravitational acceleration [m/s²]."""
        g = np.zeros(self.dimensions)
        if self.gravity and self.dimensions == 3:
            g[2] = -c.G_EARTH
        return g
