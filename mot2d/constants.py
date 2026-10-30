"""Physical constants (CODATA 2018) and Rb-87 D2-line data.

Atomic data from D. A. Steck, "Rubidium 87 D Line Data" (rev. 2.3.3).
"""

import numpy as np

HBAR = 1.054571817e-34      # reduced Planck constant [J s]
K_B = 1.380649e-23          # Boltzmann constant [J/K]
MU_B = 9.2740100783e-24     # Bohr magneton [J/T]

# Rb-87, D2 line (5S1/2 F=2 -> 5P3/2 F'=3 cycling transition)
MASS = 1.443160648e-25              # atomic mass [kg]
WAVELENGTH = 780.241209686e-9       # vacuum wavelength [m]
GAMMA = 2 * np.pi * 6.0666e6        # natural linewidth Γ [rad/s]

WAVENUMBER = 2 * np.pi / WAVELENGTH      # k [1/m]
RECOIL_VELOCITY = HBAR * WAVENUMBER / MASS  # velocity change per photon, ħk/m [m/s]
