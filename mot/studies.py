"""Parameter scans: many short simulations, each reduced to its steady state."""

from dataclasses import dataclass, replace

import numpy as np

from . import theory
from .parameters import MOTParameters
from .simulation import SimulationResult, SimulationSettings, run


def _settle(p: MOTParameters, n_atoms: int, seed: int) -> SimulationResult:
    """Start a small, hot cloud at the origin and return the steady-state half of the run.

    The run lasts long enough for both the velocity (m/β) and the position
    (theory.relaxation_rate) to settle, and at least 6 ms.
    """
    settle_time = max(theory.velocity_damping_time(p), 1 / theory.relaxation_rate(p))
    duration = max(6e-3, 10 * settle_time)
    settings = SimulationSettings(n_atoms=n_atoms, duration=duration, snapshot_interval=duration / 100,
                                  initial_temperature=1e-3, initial_radius=0.3e-3, seed=seed)
    return run(p, settings).steady_state()


def _mean_and_error(samples: np.ndarray, n_blocks: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Mean over snapshots, with the error from the scatter of block averages."""
    blocks = np.array([block.mean(axis=0) for block in np.array_split(samples, n_blocks)])
    return samples.mean(axis=0), blocks.std(axis=0, ddof=1) / np.sqrt(n_blocks)


@dataclass
class Scan:
    values: np.ndarray        # the scanned parameter
    temperature: np.ndarray   # (n, D) [K]
    temperature_error: np.ndarray
    size: np.ndarray          # RMS size, (n, D) [m]
    center: np.ndarray        # centre of mass, (n, D) [m]
    center_error: np.ndarray


def scan(params: list[MOTParameters], values, n_atoms: int = 500, seed: int = 1) -> Scan:
    """Steady-state temperature, size and position for each parameter set."""
    T, T_err, size, center, center_err = [], [], [], [], []
    for p in params:
        steady = _settle(p, n_atoms, seed)
        mean, error = _mean_and_error(steady.temperature())
        T.append(mean)
        T_err.append(error)
        size.append(steady.rms_size().mean(axis=0))
        mean, error = _mean_and_error(steady.center())
        center.append(mean)
        center_err.append(error)
    return Scan(np.asarray(values, dtype=float), *map(np.array, (T, T_err, size, center, center_err)))


def scan_detuning(base: MOTParameters, detunings, n_atoms: int = 500) -> Scan:
    return scan([replace(base, detuning=d) for d in detunings], detunings, n_atoms)


def scan_power_imbalance(base: MOTParameters, imbalances, n_atoms: int = 1000) -> Scan:
    return scan([replace(base, power_imbalance=e) for e in imbalances], imbalances, n_atoms)


def scan_stray_field(base: MOTParameters, fields_x, n_atoms: int = 1000) -> Scan:
    """Uniform stray field along x [G]."""
    return scan([replace(base, stray_field=(b, 0.0, 0.0)) for b in fields_x], fields_x, n_atoms)


def scan_gradient(base: MOTParameters, gradients, n_atoms: int = 2000) -> Scan:
    """Field gradient [G/cm]; in 3D this shows the gravitational sag mg/κ_z."""
    return scan([replace(base, field_gradient=g) for g in gradients], gradients, n_atoms)


def anti_trap(base: MOTParameters, duration: float = 6e-3,
              n_atoms: int = 1000) -> tuple[SimulationResult, SimulationResult]:
    """The same small cloud released into a normal MOT and into one with the field reversed."""
    settings = SimulationSettings(n_atoms=n_atoms, duration=duration, snapshot_interval=0.05e-3,
                                  initial_temperature=theory.doppler_temperature(base),
                                  initial_radius=0.2e-3, seed=3)
    return run(base, settings), run(replace(base, reversed_field=True), settings)
