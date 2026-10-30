"""A 2D MOT as a source of cold atoms, with a push beam.

A 2D MOT cools and traps atoms in x and y but leaves them free along z. In
experiments it fills a line of slow atoms from the vapour of a source cell;
a weak "push" beam along +z then blows them through a small differential-
pumping hole into the science chamber, where they load a 3D MOT. The
output is a beam of atoms whose speed is set by the push beam.

Geometry (all in 3D):
  - four transverse beams along ±x, ±y, elongated along z: s0 exp(−2r_t²/w² − 2z²/L²),
    with r_t the distance from the beam axis in the transverse plane
  - a 2D quadrupole field B = B'(x, −y, 0), zero along the whole z axis
  - a push beam along +z with its own saturation, detuning and waist
  - the hole: radius r_h at z = z_h, beyond the end of the cooling beams

Atoms enter a box around the beams from the vapour; those that leave the box
through its +z face are followed in a straight line to the hole. The useful
output is the atoms slow enough for a 3D MOT to capture them.
"""

from dataclasses import dataclass

import numpy as np

from . import constants as c
from .capture import ROOM_TEMPERATURE, most_probable_speed, slow_flux_fraction, total_flux
from .fast import USE_NUMBA, Beams, advance_beams
from .parameters import GAUSS_PER_CM
from .physics import lorentzian_rates, recoil_kicks

SOURCE_VAPOUR_DENSITY = 3e15  # Rb-87 in a room-temperature source cell [m⁻³] (natural Rb at 1e-7 mbar)
CAPTURABLE_SPEED = 30.0      # atoms faster than this along z pass the 3D MOT [m/s]
DIRECTIONS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], dtype=float)


@dataclass(frozen=True)
class SourceParameters:
    detuning: float = -2.0          # transverse beams, in Γ
    saturation: float = 2.0         # s0 of each transverse beam
    beam_waist: float = 5e-3        # 1/e² radius across the beams [m]
    beam_length: float = 12e-3      # 1/e² half-length along z [m]
    field_gradient: float = 15.0    # B' [G/cm]
    push_saturation: float = 0.5
    push_detuning: float = 0.0      # in Γ
    push_waist: float = 1e-3        # [m]
    hole_distance: float = 30e-3    # z of the differential-pumping hole [m]
    hole_radius: float = 0.75e-3    # [m]
    box_half_width: float = 10e-3   # atoms enter the box |x|, |y| < this ... [m]
    box_half_length: float = 20e-3  # ... and |z| < this [m]

    def saturations(self, x: np.ndarray) -> np.ndarray:
        """Local saturation of the five beams, shape (n, 5)."""
        along = np.exp(-2 * x[:, 2] ** 2 / self.beam_length**2)
        s = np.empty((len(x), 5))
        s[:, 0:2] = (self.saturation * along * np.exp(-2 * x[:, 1] ** 2 / self.beam_waist**2))[:, None]
        s[:, 2:4] = (self.saturation * along * np.exp(-2 * x[:, 0] ** 2 / self.beam_waist**2))[:, None]
        s[:, 4] = self.push_saturation * np.exp(-2 * (x[:, 0] ** 2 + x[:, 1] ** 2) / self.push_waist**2)
        return s

    def scattering_rates(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Photon scattering rate from each beam [1/s], shape (n, 5)."""
        s = self.saturations(x)
        gradient = 1.0 * c.MU_B * self.field_gradient * GAUSS_PER_CM / c.HBAR
        zeeman = (x * np.array([gradient, gradient, 0.0])) @ DIRECTIONS.T
        detuning = c.GAMMA * np.array([self.detuning] * 4 + [self.push_detuning])
        delta_eff = detuning - c.WAVENUMBER * v @ DIRECTIONS.T - zeeman
        return lorentzian_rates(s, s.sum(axis=1, keepdims=True), delta_eff)

    def beams(self) -> Beams:
        """The five beams in the form of the compiled loop."""
        w, l, wp = self.beam_waist**-2, self.beam_length**-2, self.push_waist**-2
        inverse = np.array([[0, w, l], [0, w, l], [w, 0, l], [w, 0, l], [wp, wp, 0]])
        gradient = c.MU_B * self.field_gradient * GAUSS_PER_CM / c.HBAR
        return Beams(DIRECTIONS, np.array([self.saturation] * 4 + [self.push_saturation]), inverse,
                     c.GAMMA * np.array([self.detuning] * 4 + [self.push_detuning]),
                     np.array([gradient, gradient, 0.0]), np.zeros(3))

    def box_area(self) -> float:
        a, l = self.box_half_width, self.box_half_length
        return 4 * (2 * a) * (2 * l) + 2 * (2 * a) ** 2


def sample_entering_atoms(sp: SourceParameters, n: int, v_max: float, rng: np.random.Generator,
                          temperature: float = ROOM_TEMPERATURE):
    """Vapour atoms crossing the box surface inwards with speed below v_max."""
    a, l = sp.box_half_width, sp.box_half_length
    faces = np.array([2 * a * 2 * l] * 4 + [(2 * a) ** 2] * 2)
    face = rng.choice(6, size=n, p=faces / faces.sum())
    # Face k: outward normal axis and sign; ±x, ±y, ±z.
    axis = np.array([0, 0, 1, 1, 2, 2])[face]
    sign = np.array([1, -1, 1, -1, 1, -1])[face]
    half = np.array([a, a, l])
    x = rng.uniform(-1, 1, (n, 3)) * half
    x[np.arange(n), axis] = sign * half[axis]

    y = np.linspace(0, (v_max / most_probable_speed(temperature)) ** 2, 4001)
    cdf = 1 - (1 + y) * np.exp(-y)
    speed = most_probable_speed(temperature) * np.sqrt(np.interp(rng.uniform(0, cdf[-1], n), cdf, y))
    cos_theta = np.sqrt(rng.random(n))
    phi = rng.uniform(0, 2 * np.pi, n)
    sin_theta = np.sqrt(1 - cos_theta**2)
    v = np.empty((n, 3))
    others = np.array([[1, 2], [0, 2], [0, 1]])[axis]
    v[np.arange(n), axis] = -sign * cos_theta * speed
    v[np.arange(n), others[:, 0]] = sin_theta * np.cos(phi) * speed
    v[np.arange(n), others[:, 1]] = sin_theta * np.sin(phi) * speed
    return x, v


@dataclass
class SourceResult:
    params: SourceParameters
    n_sampled: int
    weight: float              # atoms per second represented by the whole sample
    out_velocity: np.ndarray   # (n_out, 3) velocity of the atoms passing the hole [m/s]
    out_time: np.ndarray       # time each spent in the source [s]
    tracks: list               # (path (n, 3) [m], velocity on leaving [m/s]) of a few useful atoms

    def flux(self, below: float = CAPTURABLE_SPEED) -> float:
        """Atoms per second through the hole slower than `below` along z."""
        return self.weight * np.sum(self.out_velocity[:, 2] < below) / self.n_sampled

    def flux_error(self, below: float = CAPTURABLE_SPEED) -> float:
        return self.weight * np.sqrt(np.sum(self.out_velocity[:, 2] < below)) / self.n_sampled


def run(sp: SourceParameters, n_atoms: int = 200000, v_max: float = 70.0, duration: float = 20e-3,
        dt: float = 1e-6, density: float = SOURCE_VAPOUR_DENSITY, n_tracks: int = 12,
        track_interval: float = 50e-6, seed: int = 9) -> SourceResult:
    """Follow vapour atoms through the source until they leave the box; count those through the hole.

    Only atoms slower than CAPTURABLE_SPEED along z are useful, and those
    entering faster than v_max cannot end up so slow, so the useful flux is
    Φ_tot · F(v_max) · (fraction of the sample through the hole). Faster
    atoms that fly straight through the hole without being cooled are part of
    the sample but not of the useful flux.
    """
    rng = np.random.default_rng(seed)
    x, v = sample_entering_atoms(sp, n_atoms, v_max, rng)
    alive = np.arange(n_atoms)
    half = np.array([sp.box_half_width, sp.box_half_width, sp.box_half_length])
    out_index, out_v, out_t = [], [], []
    tracked = np.arange(min(n_atoms, 4000))
    track_every = max(1, round(track_interval / dt))
    snapshots = [x[tracked].copy()]
    beams = sp.beams()
    chunk = 10 if USE_NUMBA and track_every % 10 == 0 else 1  # steps between checks for atoms leaving

    for k in range(chunk, round(duration / dt) + 1, chunk):
        xa, va = x[alive], v[alive]
        if USE_NUMBA:
            advance_beams(xa, va, beams, chunk, dt, rng)
        else:
            n_absorbed = rng.poisson(sp.scattering_rates(xa, va) * dt)
            va += recoil_kicks(n_absorbed, DIRECTIONS, rng)
            xa += va * dt
        x[alive], v[alive] = xa, va
        if k % track_every == 0:
            snapshots.append(x[tracked].copy())

        outside = np.any(np.abs(xa) > half, axis=1)
        through_end = outside & (xa[:, 2] > sp.box_half_length) & (va[:, 2] > 0)
        # Straight flight from the end of the box to the plane of the hole.
        flight = (sp.hole_distance - xa[:, 2]) / np.where(va[:, 2] > 0, va[:, 2], np.inf)
        at_hole = xa[:, :2] + va[:, :2] * flight[:, None]
        hits = through_end & (np.hypot(at_hole[:, 0], at_hole[:, 1]) < sp.hole_radius)
        out_index.extend(alive[hits])
        out_v.extend(va[hits])
        out_t.extend([k * dt] * hits.sum())
        alive = alive[~outside]
        if alive.size == 0:
            break

    # Positions stop changing once an atom has left; cut each track there.
    snapshots = np.array(snapshots)
    tracks = []
    for i, velocity in zip(out_index, out_v):
        if i >= len(tracked) or velocity[2] >= CAPTURABLE_SPEED or len(tracks) == n_tracks:
            continue
        path = snapshots[:, i]
        moving = np.flatnonzero(np.any(np.diff(path, axis=0) != 0, axis=1))
        path = path[: moving[-1] + 2] if moving.size else path[:1]
        tracks.append((np.vstack([path, x[i]]), velocity))

    weight = total_flux(sp.box_area(), density) * slow_flux_fraction(v_max)
    return SourceResult(sp, n_atoms, weight, np.array(out_v).reshape(-1, 3), np.array(out_t), tracks)


def pushed_velocity(sp: SourceParameters, z_start: float = 0.0, dt: float = 1e-6) -> float:
    """Speed reached by an atom starting at rest on the axis, pushed to the end of the box [m/s].

    The mean force of the push beam along z, with the transverse beams
    saturating the transition; no photon noise.
    """
    x = np.array([[0.0, 0.0, z_start]])
    v = np.zeros((1, 3))
    while x[0, 2] < sp.box_half_length:
        a = c.HBAR * c.WAVENUMBER * sp.scattering_rates(x, v)[0, 4] / c.MASS
        v[0, 2] += a * dt
        x[0, 2] += v[0, 2] * dt
        if v[0, 2] <= 0 and a <= 0:
            return 0.0
    return float(v[0, 2])
