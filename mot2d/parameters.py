"""Experimental parameters of the 2D MOT."""

from dataclasses import dataclass

from . import constants as c

GAUSS_PER_CM = 1e-2  # 1 G/cm expressed in T/m


@dataclass(frozen=True)
class MOTParameters:
    """The knobs of the experiment.

    All four beams share the same detuning, peak intensity and waist.
    """

    detuning: float = -2.0          # laser detuning δ in units of Γ (negative = red)
    saturation: float = 0.5         # peak saturation parameter s0 = I/I_sat of *each* beam
    beam_waist: float = 5e-3        # 1/e² intensity radius w [m]
    field_gradient: float = 10.0    # quadrupole field gradient B' [G/cm]
    magnetic_moment: float = 1.0    # μ' = (g_F' m_F' − g_F m_F) in units of μ_B; 1 for |2,2> -> |3',3'>

    @property
    def delta(self) -> float:
        """Detuning δ in rad/s."""
        return self.detuning * c.GAMMA

    @property
    def zeeman_gradient(self) -> float:
        """Zeeman shift per unit displacement, μ'B'/ħ [rad s⁻¹ m⁻¹]."""
        return self.magnetic_moment * c.MU_B * self.field_gradient * GAUSS_PER_CM / c.HBAR
