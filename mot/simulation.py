"""Monte-Carlo time evolution of atoms in the MOT."""

from dataclasses import dataclass

import numpy as np

from . import constants as c
from . import theory
from .fast import advance
from .parameters import MOTParameters


@dataclass(frozen=True)
class SimulationSettings:
    n_atoms: int = 1000
    duration: float = 20e-3             # [s]
    dt: float = 1e-6                    # time step [s]; R·dt ≲ 1 photon per beam
    snapshot_interval: float = 0.1e-3   # how often the state is stored [s]
    initial_temperature: float = 10e-3  # [K]
    initial_radius: float = 3e-3        # atoms start uniformly inside this disc/ball [m]
    seed: int | None = 0


@dataclass
class SimulationResult:
    params: MOTParameters
    settings: SimulationSettings
    times: np.ndarray        # (n_snapshots,) [s]
    positions: np.ndarray    # (n_snapshots, n_atoms, D) [m]
    velocities: np.ndarray   # (n_snapshots, n_atoms, D) [m/s]

    def temperature(self) -> np.ndarray:
        """Kinetic temperature along each axis, T = m Var(v) / k_B, shape (n_snapshots, D) [K]."""
        return c.MASS * self.velocities.var(axis=1) / c.K_B

    def rms_size(self) -> np.ndarray:
        """RMS cloud size along each axis, shape (n_snapshots, D) [m]."""
        return self.positions.std(axis=1)

    def center(self) -> np.ndarray:
        """Centre of mass of the cloud, shape (n_snapshots, D) [m]."""
        return self.positions.mean(axis=1)

    def steady_state(self, fraction: float = 0.5) -> "SimulationResult":
        """The last `fraction` of the run, where the cloud has reached equilibrium."""
        keep = self.times >= (1 - fraction) * self.times[-1]
        return SimulationResult(self.params, self.settings, self.times[keep],
                                self.positions[keep], self.velocities[keep])


def initial_state(s: SimulationSettings, dimensions: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Uniform disc (2D) or ball (3D) in position, Maxwell-Boltzmann in velocity."""
    direction = rng.normal(size=(s.n_atoms, dimensions))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = s.initial_radius * rng.random(s.n_atoms) ** (1 / dimensions)
    positions = radius[:, None] * direction

    sigma_v = np.sqrt(c.K_B * s.initial_temperature / c.MASS)  # per velocity component
    velocities = rng.normal(0.0, sigma_v, (s.n_atoms, dimensions))
    return positions, velocities


def doppler_steady_state(p: MOTParameters, n_atoms: int,
                         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Atoms drawn from the steady state predicted by Doppler theory: a Gaussian cloud at T_D."""
    x = theory.equilibrium_position(p) + rng.normal(0, theory.cloud_sizes(p), (n_atoms, p.dimensions))
    v = rng.normal(0, np.sqrt(c.K_B * theory.doppler_temperature(p) / c.MASS), (n_atoms, p.dimensions))
    return x, v


def run(p: MOTParameters, s: SimulationSettings = SimulationSettings()) -> SimulationResult:
    """Evolve the atoms with a semi-implicit Euler step: kick v, then drift x."""
    rng = np.random.default_rng(s.seed)
    x, v = initial_state(s, p.dimensions, rng)
    steps_per_snapshot = max(1, round(s.snapshot_interval / s.dt))
    n_snapshots = max(1, round(s.duration / (steps_per_snapshot * s.dt)))

    positions = np.empty((n_snapshots + 1, s.n_atoms, p.dimensions))
    velocities = np.empty_like(positions)
    for i in range(n_snapshots):
        positions[i], velocities[i] = x, v
        advance(x, v, p, steps_per_snapshot, s.dt, rng)
    positions[-1], velocities[-1] = x, v

    times = np.arange(n_snapshots + 1) * steps_per_snapshot * s.dt
    return SimulationResult(p, s, times, positions, velocities)
