"""Monte-Carlo time evolution of atoms in the 2D MOT."""

from dataclasses import dataclass, replace

import numpy as np

from . import constants as c
from . import theory
from .parameters import MOTParameters
from .physics import photon_kicks, scattering_rates


@dataclass(frozen=True)
class SimulationSettings:
    n_atoms: int = 1000
    duration: float = 20e-3             # [s]
    dt: float = 1e-6                    # time step [s]; R·dt ≲ 1 photon per beam
    snapshot_interval: float = 0.1e-3   # how often the state is stored [s]
    initial_temperature: float = 10e-3  # [K]
    initial_radius: float = 3e-3        # atoms start uniformly inside this disc [m]
    seed: int | None = 0


@dataclass
class SimulationResult:
    params: MOTParameters
    settings: SimulationSettings
    times: np.ndarray        # (n_snapshots,) [s]
    positions: np.ndarray    # (n_snapshots, n_atoms, 2) [m]
    velocities: np.ndarray   # (n_snapshots, n_atoms, 2) [m/s]

    def temperature(self) -> np.ndarray:
        """Kinetic temperature along x and y, T = m Var(v) / k_B, shape (n_snapshots, 2) [K]."""
        return c.MASS * self.velocities.var(axis=1) / c.K_B

    def rms_size(self) -> np.ndarray:
        """RMS cloud size along x and y, shape (n_snapshots, 2) [m]."""
        return self.positions.std(axis=1)

    def steady_state(self, fraction: float = 0.5) -> "SimulationResult":
        """The last `fraction` of the run, where the cloud has reached equilibrium."""
        keep = self.times >= (1 - fraction) * self.times[-1]
        return SimulationResult(self.params, self.settings, self.times[keep],
                                self.positions[keep], self.velocities[keep])


def initial_state(s: SimulationSettings, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Uniform disc in position, Maxwell-Boltzmann in velocity."""
    radius = s.initial_radius * np.sqrt(rng.random(s.n_atoms))
    angle = rng.uniform(0, 2 * np.pi, s.n_atoms)
    positions = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))

    sigma_v = np.sqrt(c.K_B * s.initial_temperature / c.MASS)  # per velocity component
    velocities = rng.normal(0.0, sigma_v, (s.n_atoms, 2))
    return positions, velocities


def run(p: MOTParameters, s: SimulationSettings = SimulationSettings()) -> SimulationResult:
    """Evolve the atoms with a semi-implicit Euler step: kick v, then drift x."""
    rng = np.random.default_rng(s.seed)
    x, v = initial_state(s, rng)

    steps_per_snapshot = max(1, round(s.snapshot_interval / s.dt))
    n_snapshots = max(1, round(s.duration / (steps_per_snapshot * s.dt)))

    positions = np.empty((n_snapshots + 1, s.n_atoms, 2))
    velocities = np.empty_like(positions)
    for i in range(n_snapshots):
        positions[i], velocities[i] = x, v
        for _ in range(steps_per_snapshot):
            v += photon_kicks(scattering_rates(x, v, p), s.dt, rng)
            x += v * s.dt
    positions[-1], velocities[-1] = x, v

    times = np.arange(n_snapshots + 1) * steps_per_snapshot * s.dt
    return SimulationResult(p, s, times, positions, velocities)


@dataclass
class DetuningScan:
    detuning: np.ndarray            # [Γ]
    temperature: np.ndarray         # steady-state temperature, mean of x and y [K]
    temperature_error: np.ndarray   # statistical uncertainty of the mean [K]
    cloud_size: np.ndarray          # steady-state RMS size, mean of x and y [m]


def scan_detuning(base: MOTParameters, detunings, n_atoms: int = 500, seed: int = 1) -> DetuningScan:
    """Steady-state temperature and cloud size for each detuning.

    Every run starts from a small, hot cloud and lasts ten velocity-damping times
    (at least 6 ms), so the second half is well into the steady state.
    """
    temperature, error, size = [], [], []
    for detuning in detunings:
        p = replace(base, detuning=detuning)
        tau = theory.velocity_damping_time(p)
        duration = max(6e-3, 10 * tau)
        settings = SimulationSettings(n_atoms=n_atoms, duration=duration, snapshot_interval=duration / 100,
                                      initial_temperature=1e-3, initial_radius=0.3e-3, seed=seed)
        steady = run(p, settings).steady_state()

        # <v²> decorrelates after ~τ/2, so the steady half holds about 2 × length/(τ/2)
        # independent temperature readings (x and y).
        samples = steady.temperature()
        n_independent = 2 * (steady.times[-1] - steady.times[0]) / (tau / 2)
        temperature.append(samples.mean())
        error.append(samples.std() / np.sqrt(n_independent))
        size.append(steady.rms_size().mean())

    return DetuningScan(np.asarray(detunings, dtype=float), np.array(temperature),
                        np.array(error), np.array(size))
