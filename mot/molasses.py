"""Sub-Doppler cooling in σ+σ− optical molasses, from the optical Bloch equations.

After the MOT, experiments switch off the field, detune further and lower the
intensity: an optical molasses. A real Rb atom then cools far below the
Doppler limit. This needs the full Zeeman structure of F=2 → F'=3, which the
two-level model leaves out.

1D model. Two counter-propagating beams along z, σ+ along +z and σ− along −z,
add up to a linear polarisation that rotates along z. An atom moving at v
sees it rotate at the rate kv. In the frame that rotates with it, the
Hamiltonian is time-independent with an extra term kv·J_z (Dalibard &
Cohen-Tannoudji, JOSA B 6, 2023 (1989)):

    H = −δ Σ_e |e⟩⟨e| + kv M + (Ω/2) Σ_{q=±1} (A_q + A_q†),

where A_q = Σ_m ⟨2 m; 1 q|3 m+q⟩ |3' m+q⟩⟨2 m| and M is the magnetic quantum
number. With spontaneous emission (rate Γ, jump operators A_q†), the steady
state of the 12-level density matrix ρ gives the force and the scattering
rate at each velocity:

    F(v) = −⟨∂H/∂z⟩,    R(v) = Γ Σ_e ρ_ee.

The motion uses this force and the photon noise of R(v); the fluctuations of
the dipole force between the beams are left out.
"""

from dataclasses import dataclass
from math import factorial, sqrt

import numpy as np

from . import constants as c
from .parameters import MOTParameters
from .physics import recoil_kicks
from .simulation import SimulationResult, SimulationSettings

F_GROUND, F_EXCITED = 2, 3
N_G, N_E = 2 * F_GROUND + 1, 2 * F_EXCITED + 1
DIM = N_G + N_E
RECOIL_TEMPERATURE = c.MASS * c.RECOIL_VELOCITY**2 / c.K_B  # (ħk)²/(m k_B)


def clebsch_gordan(j1: int, m1: int, j2: int, m2: int, j: int, m: int) -> float:
    """⟨j1 m1; j2 m2 | j m⟩ for integer angular momenta (Racah's formula)."""
    if m1 + m2 != m or abs(m1) > j1 or abs(m2) > j2 or abs(m) > j or not abs(j1 - j2) <= j <= j1 + j2:
        return 0.0
    prefactor = sqrt((2 * j + 1) * factorial(j + j1 - j2) * factorial(j - j1 + j2) * factorial(j1 + j2 - j)
                     / factorial(j1 + j2 + j + 1))
    prefactor *= sqrt(factorial(j + m) * factorial(j - m) * factorial(j1 - m1) * factorial(j1 + m1)
                      * factorial(j2 - m2) * factorial(j2 + m2))
    total = 0.0
    for k in range(0, j1 + j2 - j + 1):
        args = (k, j1 + j2 - j - k, j1 - m1 - k, j2 + m2 - k, j - j2 + m1 + k, j - j1 - m2 + k)
        if min(args) >= 0:
            total += (-1) ** k / np.prod([factorial(a) for a in args])
    return prefactor * total


def _ground(m):
    return m + F_GROUND


def _excited(m):
    return N_G + m + F_EXCITED


def _operators():
    """Raising operators A_q (q = −1, 0, +1), magnetic number M, excited-state projector."""
    raising = {}
    for q in (-1, 0, 1):
        a = np.zeros((DIM, DIM))
        for m in range(-F_GROUND, F_GROUND + 1):
            if abs(m + q) <= F_EXCITED:
                a[_excited(m + q), _ground(m)] = clebsch_gordan(F_GROUND, m, 1, q, F_EXCITED, m + q)
        raising[q] = a
    magnetic = np.diag([m for m in range(-F_GROUND, F_GROUND + 1)] + [m for m in range(-F_EXCITED, F_EXCITED + 1)])
    excited = np.diag([0.0] * N_G + [1.0] * N_E)
    return raising, magnetic.astype(float), excited


_RAISING, _M, _PE = _operators()


def _superoperator(h: np.ndarray, jumps) -> np.ndarray:
    """Liouvillian for column-stacked ρ: vec(XρY) = (Yᵀ ⊗ X) vec(ρ)."""
    eye = np.eye(DIM)
    lv = -1j * (np.kron(eye, h) - np.kron(h.T, eye))
    for jump in jumps:
        jj = jump.conj().T @ jump
        lv += np.kron(jump.conj(), jump) - 0.5 * np.kron(eye, jj) - 0.5 * np.kron(jj.T, eye)
    return lv


def steady_state(kv: float, detuning: float, rabi: float) -> np.ndarray:
    """Steady-state ρ in the rotating frame; kv, δ and Ω in units of Γ."""
    coupling = 0.5 * rabi * (_RAISING[1] + _RAISING[-1])
    h = -detuning * _PE + kv * _M + coupling + coupling.T
    lv = _superoperator(h.astype(complex), [_RAISING[q].T for q in (-1, 0, 1)])
    trace_row = np.eye(DIM).reshape(-1)  # Tr ρ = Σ ρ_ii
    lv[0] = trace_row
    rhs = np.zeros(DIM * DIM, dtype=complex)
    rhs[0] = 1
    return np.linalg.solve(lv, rhs).reshape(DIM, DIM, order="F")


def force_and_rate(velocities, detuning: float, saturation: float):
    """Mean force [N] and total scattering rate [1/s] for atoms moving at v along the beams.

    `saturation` is s0 of each beam for the stretched transition, Ω = Γ√(s0/2).
    """
    rabi = np.sqrt(saturation / 2)
    gradient = 0.5j * rabi * (_RAISING[1] - _RAISING[-1])  # ∂H/∂z in units of ħk Γ
    gradient = gradient + gradient.conj().T
    force, rate = [], []
    for v in np.atleast_1d(velocities):
        rho = steady_state(c.WAVENUMBER * v / c.GAMMA, detuning, rabi)
        force.append(-np.trace(rho @ gradient).real)
        rate.append(np.trace(rho @ _PE).real)
    unit = c.HBAR * c.WAVENUMBER * c.GAMMA
    return np.array(force) * unit, np.array(rate) * c.GAMMA


def doppler_force(velocities, detuning: float, saturation: float) -> np.ndarray:
    """Two-level rate-equation force of the same beam pair, as in `physics` [N]."""
    v = np.asarray(velocities, dtype=float)
    kv = c.WAVENUMBER * v / c.GAMMA

    def push(shift):
        return saturation / (1 + 2 * saturation + 4 * (detuning - shift) ** 2)

    return c.HBAR * c.WAVENUMBER * 0.5 * c.GAMMA * (push(kv) - push(-kv))


@dataclass
class ForceTable:
    """F(v) and R(v) of one beam pair on a grid dense near v = 0, interpolated in between."""
    detuning: float
    saturation: float
    velocity: np.ndarray
    force: np.ndarray
    rate: np.ndarray

    @classmethod
    def build(cls, detuning: float, saturation: float, v_max: float = 6.0, n: int = 241):
        u = np.linspace(0, np.arcsinh(v_max / 2e-3), n // 2 + 1)
        positive = 2e-3 * np.sinh(u)
        force, rate = force_and_rate(positive, detuning, saturation)
        velocity = np.concatenate([-positive[:0:-1], positive])
        force = np.concatenate([-force[:0:-1], force])  # odd in v
        rate = np.concatenate([rate[:0:-1], rate])      # even in v
        return cls(detuning, saturation, velocity, force, rate)

    def __call__(self, v: np.ndarray):
        return np.interp(v, self.velocity, self.force), np.interp(v, self.velocity, self.rate)

    def friction(self) -> float:
        """β = −dF/dv at v = 0 [kg/s]."""
        i = len(self.velocity) // 2
        return -(self.force[i + 1] - self.force[i - 1]) / (self.velocity[i + 1] - self.velocity[i - 1])

    def capture_velocity(self) -> float:
        """Velocity of the first maximum of the friction force: the edge of the sub-Doppler peak [m/s]."""
        i = len(self.velocity) // 2
        drag = -self.force[i:]
        peaks = np.flatnonzero((drag[1:-1] > drag[:-2]) & (drag[1:-1] >= drag[2:])) + 1
        return float(self.velocity[i + peaks[0]]) if peaks.size else float(self.velocity[-1])

    def core_temperature(self) -> float:
        """k_B T = D(0)/β for atoms inside the sub-Doppler peak (3D, all axes alike) [K]."""
        rate = self.rate[len(self.velocity) // 2]
        diffusion = 0.5 * (c.HBAR * c.WAVENUMBER) ** 2 * rate * (1 + 3 / 3)
        return diffusion / self.friction() / c.K_B


def stationary_distribution(table: ForceTable, other_axes_rate: float | None = None):
    """Velocity distribution P(v) along one axis from the Fokker-Planck equation.

    With momentum diffusion 2D(v) = (ħk)² [R(v) + R_all/3] (absorption from this
    pair, isotropic emission of all photons, R_all = R(v) + 2R̄ in 3D) the
    stationary solution is P ∝ exp(∫ m F/D dv) / D. The rate of the other two
    axes, R̄, is found self-consistently unless given.
    """
    v = np.linspace(-table.velocity.max(), table.velocity.max(), 40001)
    force, rate = table(v)
    mean_other = other_axes_rate if other_axes_rate is not None else rate.min()
    for _ in range(20 if other_axes_rate is None else 1):
        diffusion = 0.5 * (c.HBAR * c.WAVENUMBER) ** 2 * (rate + (rate + 2 * mean_other) / 3)
        integrand = c.MASS * force / diffusion
        exponent = np.concatenate([[0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(v))])
        exponent -= exponent[len(v) // 2]
        density = np.exp(exponent) / diffusion
        density /= np.trapezoid(density, v)
        if other_axes_rate is None:
            mean_other = np.trapezoid(density * rate, v)
    return v, density


def fokker_planck_temperature(table: ForceTable) -> float:
    """k_B T = m⟨v²⟩ of the stationary distribution [K]."""
    v, density = stationary_distribution(table)
    return c.MASS * np.trapezoid(density * v**2, v) / c.K_B


# --- 3D molasses: every axis feels the 1D force ------------------------------------

@dataclass
class MolassesResult:
    times: np.ndarray
    positions: np.ndarray    # (n_snapshots, n_atoms, 3)
    velocities: np.ndarray

    def temperature(self) -> np.ndarray:
        return c.MASS * self.velocities.var(axis=1) / c.K_B

    def as_simulation_result(self) -> SimulationResult:
        """The last snapshot, as a cloud that `measurement.time_of_flight` can release."""
        n = self.positions.shape[1]
        return SimulationResult(MOTParameters(dimensions=3), SimulationSettings(n_atoms=n),
                                self.times[-1:], self.positions[-1:], self.velocities[-1:])


def run(stages, x: np.ndarray, v: np.ndarray, dt: float = 1e-6, snapshot_interval: float = 0.05e-3,
        seed: int = 8) -> MolassesResult:
    """Evolve a cloud through molasses stages [(ForceTable, duration), ...] with the field off.

    Along each axis the atom feels the 1D force F(v_i) of that beam pair and
    scatters R(v_i) photons per second. Each photon is absorbed from one of
    the two beams at random (the mean is already in F) and re-emitted
    isotropically. Gravity acts throughout.
    """
    rng = np.random.default_rng(seed)
    x, v = x.copy(), v.copy()
    directions = MOTParameters(dimensions=3).beam_directions
    gravity_kick = MOTParameters(dimensions=3).gravity_vector * dt
    times, xs, vs = [0.0], [x.copy()], [v.copy()]
    t = 0.0
    steps = max(1, round(snapshot_interval / dt))
    for table, duration in stages:
        for k in range(round(duration / dt)):
            force, rate = table(v)
            n_axis = rng.poisson(rate * dt)
            plus = rng.binomial(n_axis, 0.5)
            photons = np.stack([plus, n_axis - plus], axis=2).reshape(len(v), 6)
            v += force / c.MASS * dt + recoil_kicks(photons, directions, rng) + gravity_kick
            x += v * dt
            t += dt
            if (k + 1) % steps == 0:
                times.append(t)
                xs.append(x.copy())
                vs.append(v.copy())
    return MolassesResult(np.array(times), np.array(xs), np.array(vs))
