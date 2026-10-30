"""Loading the MOT from the background vapour: capture velocity and loading rate.

A MOT is filled from the room-temperature vapour in the cell. Only the slowest
atoms crossing the beams are stopped before they leave: those slower than the
capture velocity v_c. The flux of such atoms through a surface of area A is

    Φ(v < v_c) = Φ_tot · F(v_c / v_p),   Φ_tot = n v̄ A / 4,   F(x) = 1 − (1 + x²) e^{−x²}

with v_p = √(2k_BT/m) and v̄ = 2v_p/√π. Since F(x) ≈ x⁴/2 for slow atoms, the
loading rate grows as v_c⁴ (Monroe et al., PRL 65, 1571 (1990)).

Here the same photon-by-photon model as in `simulation` decides whether an atom
is captured; nothing beyond the beams and the field is added. The beams are
taken to fill a sphere of radius 2w (the windows of the cell); atoms enter it
from the vapour.
"""

from dataclasses import dataclass

import numpy as np

from . import constants as c
from . import theory
from .fast import advance
from .parameters import MOTParameters

ROOM_TEMPERATURE = 295.0  # vapour temperature [K]
VAPOUR_DENSITY = 1e15     # Rb-87 background density [m⁻³] (1e9 cm⁻³, ~3e-8 mbar)
LOSS_CROSS_SECTION = 2e-17  # trap loss in collisions with hot vapour atoms [m²] (order of magnitude)


@dataclass(frozen=True)
class CaptureSettings:
    start_radius: float = 10e-3   # atoms enter the sphere of this radius (2w: beyond the beams) [m]
    escape_radius: float = 20e-3  # atoms leaving this sphere are lost; slow ones may overshoot and return
    duration: float | None = None  # [s]; by default long enough for a captured atom to settle
    dt: float = 1e-6              # [s]
    trapped_radius: float = 2e-3  # captured: inside this radius ...
    trapped_speed: float = 1.0    # ... and slower than this at the end [m/s]
    seed: int = 5


def most_probable_speed(temperature: float = ROOM_TEMPERATURE) -> float:
    """v_p = √(2k_BT/m) [m/s]."""
    return np.sqrt(2 * c.K_B * temperature / c.MASS)


def slow_flux_fraction(v_max, temperature: float = ROOM_TEMPERATURE):
    """Fraction F of the flux through a surface carried by atoms slower than v_max."""
    x2 = (np.asarray(v_max) / most_probable_speed(temperature)) ** 2
    return 1 - (1 + x2) * np.exp(-x2)


def total_flux(area: float, density: float = VAPOUR_DENSITY, temperature: float = ROOM_TEMPERATURE) -> float:
    """Atoms per second crossing a surface of area A inwards, Φ = n v̄ A / 4."""
    return density * 2 * most_probable_speed(temperature) / np.sqrt(np.pi) * area / 4


def settle_time(p: MOTParameters, s: CaptureSettings) -> float:
    """Run time: 3 ms to cross the beams plus five relaxation times of the trap [s]."""
    return s.duration if s.duration is not None else 3e-3 + 5 / theory.relaxation_rate(p)


def evolve(x: np.ndarray, v: np.ndarray, p: MOTParameters, s: CaptureSettings,
           rng: np.random.Generator | None) -> np.ndarray:
    """Follow incoming atoms photon by photon; return which of them end up trapped.

    With rng=None the photon kicks are replaced by the mean force: the same
    trajectories without the noise. Atoms that leave the escape sphere moving
    outwards are dropped, since the beams can no longer bring them back.
    """
    x, v = x.copy(), v.copy()
    alive = np.arange(len(x))
    chunk = 20  # steps between checks for escaped atoms
    for _ in range(round(settle_time(p, s) / s.dt / chunk)):
        xa, va = x[alive], v[alive]
        advance(xa, va, p, chunk, s.dt, rng)
        x[alive], v[alive] = xa, va
        leaving = (np.einsum("ij,ij->i", xa, xa) > s.escape_radius**2) & (np.einsum("ij,ij->i", xa, va) > 0)
        alive = alive[~leaving]
        if alive.size == 0:
            break

    trapped = np.zeros(len(x), dtype=bool)
    r, u = np.linalg.norm(x[alive], axis=1), np.linalg.norm(v[alive], axis=1)
    trapped[alive] = (r < s.trapped_radius) & (u < s.trapped_speed)
    return trapped


def _head_on(speeds, n_each: int, p: MOTParameters, s: CaptureSettings, axis: int):
    """Atoms entering at −R along `axis`, moving straight at the centre."""
    speeds = np.repeat(np.asarray(speeds, dtype=float), n_each)
    x = np.zeros((speeds.size, p.dimensions))
    v = np.zeros_like(x)
    x[:, axis] = -s.start_radius
    v[:, axis] = speeds
    return x, v, speeds


def capture_probability(p: MOTParameters, speeds, n_each: int = 200, axis: int = 0,
                        s: CaptureSettings = CaptureSettings()) -> np.ndarray:
    """Fraction of head-on atoms captured, for each launch speed."""
    x, v, _ = _head_on(speeds, n_each, p, s, axis)
    trapped = evolve(x, v, p, s, np.random.default_rng(s.seed))
    return trapped.reshape(len(speeds), n_each).mean(axis=1)


def deterministic_capture_velocity(p: MOTParameters, axis: int = 0, s: CaptureSettings = CaptureSettings(),
                                   speeds=np.linspace(0.25, 80, 320)) -> float:
    """Fastest head-on speed stopped by the mean force alone (no photon noise) [m/s]."""
    x, v, speeds = _head_on(speeds, 1, p, s, axis)
    escaped = np.flatnonzero(~evolve(x, v, p, s, rng=None))
    return speeds[-1] if escaped.size == 0 else speeds[max(escaped[0] - 1, 0)]


def half_capture_speed(speeds, probability) -> float:
    """Speed at which the capture probability falls through 1/2 [m/s]."""
    speeds, probability = np.asarray(speeds), np.asarray(probability)
    below = np.flatnonzero(probability < 0.5)
    if below.size == 0 or below[0] == 0:
        return float("nan")
    i = below[0]
    return float(np.interp(0.5, [probability[i], probability[i - 1]], [speeds[i], speeds[i - 1]]))


# --- Loading from an isotropic vapour ------------------------------------------------

def sample_vapour_atoms(n: int, v_max: float, p: MOTParameters, s: CaptureSettings,
                        rng: np.random.Generator, temperature: float = ROOM_TEMPERATURE):
    """Atoms crossing the starting sphere inwards, drawn from the vapour flux with v < v_max.

    Positions are uniform on the sphere, directions follow the cos θ law of an
    effusive flux, and the speed density is ∝ v³ exp(−v²/v_p²). With y = (v/v_p)²
    the cumulative distribution is 1 − (1 + y) e^{−y}, inverted from a table.
    """
    if p.dimensions != 3:
        raise ValueError("vapour loading is modelled for the 3D MOT")
    normal = rng.normal(size=(n, 3))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    x = s.start_radius * normal

    y = np.linspace(0, (v_max / most_probable_speed(temperature)) ** 2, 4001)
    cdf = 1 - (1 + y) * np.exp(-y)
    speed = most_probable_speed(temperature) * np.sqrt(np.interp(rng.uniform(0, cdf[-1], n), cdf, y))

    cos_theta = np.sqrt(rng.random(n))  # angle to the inward normal −r̂
    phi = rng.uniform(0, 2 * np.pi, n)
    helper = np.where(np.abs(normal[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    e1 = np.cross(normal, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(normal, e1)
    sin_theta = np.sqrt(1 - cos_theta**2)
    direction = (-cos_theta[:, None] * normal + sin_theta[:, None] * (np.cos(phi)[:, None] * e1
                                                                       + np.sin(phi)[:, None] * e2))
    return x, speed[:, None] * direction, speed


@dataclass
class Loading:
    speeds: np.ndarray            # speed of each sampled atom [m/s]
    trapped: np.ndarray           # whether it was captured, with photon noise
    trapped_mean_force: np.ndarray  # whether the same atom is captured by the mean force alone
    v_max: float                  # sampled atoms are slower than this [m/s]
    weight: float                 # real atoms per second represented by the whole sample
    capture_velocity: float       # head-on deterministic v_c [m/s]
    beam_waist: float

    @property
    def rate(self) -> float:
        """Loading rate R_L [atoms/s]."""
        return self.weight * self.trapped.mean()

    @property
    def rate_error(self) -> float:
        f = self.trapped.mean()
        return self.weight * np.sqrt(f * (1 - f) / self.trapped.size)

    @property
    def rate_mean_force(self) -> float:
        return self.weight * self.trapped_mean_force.mean()

    def textbook_rate(self, density: float = VAPOUR_DENSITY, temperature: float = ROOM_TEMPERATURE) -> float:
        """R = ½ n V^{2/3} v_c⁴ (m/2k_BT)^{3/2} (Monroe 1990), V = (2w)³, head-on v_c."""
        volume = (2 * self.beam_waist) ** 3
        return 0.5 * density * volume ** (2 / 3) * self.capture_velocity**4 / most_probable_speed(temperature) ** 3

    def steady_atom_number(self, density: float = VAPOUR_DENSITY,
                           temperature: float = ROOM_TEMPERATURE) -> float:
        """N = R_L τ, with τ⁻¹ = n σ_loss v̄ the loss rate from collisions with the vapour."""
        loss_rate = density * LOSS_CROSS_SECTION * 2 * most_probable_speed(temperature) / np.sqrt(np.pi)
        return self.rate / loss_rate


def loading_rate(p: MOTParameters, n_atoms: int = 20000, s: CaptureSettings = CaptureSettings(),
                 density: float = VAPOUR_DENSITY, temperature: float = ROOM_TEMPERATURE) -> Loading:
    """Monte-Carlo loading rate: simulate only the slow tail of the vapour, then reweight it.

    Atoms faster than v_max = 1.6 v_c (head-on) are never captured, so the
    rate is Φ_tot · F(v_max) · (captured fraction of the sampled atoms).
    """
    v_c = deterministic_capture_velocity(p, s=s)
    v_max = 1.6 * v_c
    rng = np.random.default_rng(s.seed)
    x, v, speed = sample_vapour_atoms(n_atoms, v_max, p, s, rng, temperature)
    trapped = evolve(x, v, p, s, rng)
    trapped_mean_force = evolve(x, v, p, s, rng=None)

    flux = total_flux(4 * np.pi * s.start_radius**2, density, temperature)
    return Loading(speed, trapped, trapped_mean_force, v_max, flux * slow_flux_fraction(v_max, temperature),
                   v_c, p.beam_waist)
