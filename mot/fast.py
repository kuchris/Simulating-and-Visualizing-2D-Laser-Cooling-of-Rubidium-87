"""Advance the atoms by many time steps at once: the inner loop of every simulation.

`advance` does exactly what this loop does with the functions of `physics`:

    for _ in range(n_steps):
        rates = scattering_rates(x, v, p, transmission) * brightness[:, None]
        n_absorbed = rng.poisson(rates * dt)
        v += recoil_kicks(n_absorbed, p.beam_directions, rng) + (gravity + acceleration) * dt
        x += v * dt

With Numba installed it runs as compiled code, one atom per thread, and is
much faster. Each atom then draws from its own random stream (xorshift64*),
seeded from `rng`, so a run gives the same result on any number of threads;
the numbers differ from the NumPy path, but not their statistics.

The compiled loop takes any set of Gaussian beams (`Beams`), so the 2D-MOT
source uses it too, and with noise=False it follows the mean force instead.

Set the environment variable MOT_NUMBA=0 to use the NumPy path.
"""

import os
from dataclasses import dataclass

import numpy as np

from . import constants as c
from .parameters import MOTParameters
from .physics import mean_force, recoil_kicks, scattering_rates

try:
    import numba
except ImportError:  # pragma: no cover - the NumPy path below is the fallback
    numba = None

USE_NUMBA = numba is not None and os.environ.get("MOT_NUMBA", "1") != "0"


@dataclass(frozen=True)
class Beams:
    """Gaussian beams along the coordinate axes, in the form the compiled loop reads.

    Beam j has the local saturation s_j exp(−2 Σ_a x_a² / w_ja²), with 1/w_ja² = 0
    along its own axis, and the detuning δ_j − e_j·(k v + G ∘ (r − r_0)).
    """
    directions: np.ndarray        # (n_beams, D)
    saturation: np.ndarray        # (n_beams,)
    inverse_waist_sq: np.ndarray  # (n_beams, D) [1/m²]
    detuning: np.ndarray          # (n_beams,) [rad/s]
    zeeman: np.ndarray            # (D,) Zeeman shift per unit displacement [rad s⁻¹ m⁻¹]
    field_zero: np.ndarray        # (D,) [m]

    @classmethod
    def of_mot(cls, p: MOTParameters) -> "Beams":
        directions = p.beam_directions
        inverse = (1 - np.abs(directions)) / p.beam_waist**2
        return cls(directions, p.saturation * p.beam_powers, inverse, np.full(len(directions), p.delta),
                   p.zeeman_gradients, p.field_zero)


def advance(x: np.ndarray, v: np.ndarray, p: MOTParameters, n_steps: int, dt: float,
            rng: np.random.Generator | None, transmission: np.ndarray | None = None,
            acceleration: np.ndarray | None = None, brightness: np.ndarray | None = None) -> np.ndarray:
    """Evolve x and v in place for n_steps; return the photons each atom scattered.

    rng           random generator; None follows the mean force, without photon noise
    transmission  (n, n_beams) fraction of each beam reaching each atom (default 1)
    acceleration  (n, D) extra acceleration besides gravity [m/s²] (default 0)
    brightness    (n,) 1 for atoms that scatter, 0 for dark ones (default 1)
    """
    n, dims = x.shape
    transmission = np.ones((n, 2 * dims)) if transmission is None else transmission
    acceleration = np.zeros((n, dims)) if acceleration is None else acceleration
    brightness = np.ones(n) if brightness is None else brightness
    acceleration = acceleration + p.gravity_vector

    if USE_NUMBA:
        return advance_beams(x, v, Beams.of_mot(p), n_steps, dt, rng, transmission, acceleration, brightness)

    photons = np.zeros(n, dtype=np.int64)
    for _ in range(n_steps):
        if rng is None:
            v += mean_force(x, v, p) / c.MASS * dt + acceleration * dt
        else:
            rates = scattering_rates(x, v, p, transmission) * brightness[:, None]
            n_absorbed = rng.poisson(rates * dt)
            v += recoil_kicks(n_absorbed, p.beam_directions, rng) + acceleration * dt
            photons += n_absorbed.sum(axis=1)
        x += v * dt
    return photons


def advance_beams(x: np.ndarray, v: np.ndarray, beams: Beams, n_steps: int, dt: float,
                  rng: np.random.Generator | None, transmission: np.ndarray | None = None,
                  acceleration: np.ndarray | None = None, brightness: np.ndarray | None = None) -> np.ndarray:
    """The compiled loop for any `Beams` (Numba only). Arguments as for `advance`."""
    n, dims = x.shape
    transmission = np.ones((n, len(beams.directions))) if transmission is None else transmission
    acceleration = np.zeros((n, dims)) if acceleration is None else acceleration
    brightness = np.ones(n) if brightness is None else brightness
    seeds = (np.ones(n, dtype=np.uint64) if rng is None
             else rng.integers(1, 2**63, size=n, dtype=np.uint64))
    photons = np.zeros(n, dtype=np.int64)
    _advance(x, v, n_steps, dt, beams.directions, beams.saturation, beams.inverse_waist_sq, beams.detuning,
             beams.zeeman, beams.field_zero, transmission, acceleration, brightness, rng is not None,
             seeds, photons)
    return photons


def interpolate(grid: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Trilinear interpolation of a 3D grid at fractional cell coordinates u (n, 3); zero outside."""
    if USE_NUMBA:
        return _interpolate(grid, np.ascontiguousarray(u))
    lo = np.floor(u).astype(int)
    f = u - lo
    value = np.zeros(len(u))
    for corner in np.ndindex(2, 2, 2):
        i = lo + np.array(corner)
        ok = np.all((i >= 0) & (i < np.array(grid.shape)), axis=1)
        w = np.prod(np.where(corner, f, 1 - f), axis=1)
        value[ok] += w[ok] * grid[i[ok, 0], i[ok, 1], i[ok, 2]]
    return value


def warm_up():
    """Compile the loops (or load them from Numba's cache) with one-atom runs."""
    if USE_NUMBA:
        for rng in (np.random.default_rng(0), None):
            advance(np.zeros((1, 3)), np.zeros((1, 3)), MOTParameters(), 1, 1e-6, rng)
        interpolate(np.zeros((2, 2, 2)), np.zeros((1, 3)))


def set_threads(n: int):
    """Number of threads the compiled loop may use (call with 1 inside worker processes)."""
    if USE_NUMBA:
        numba.set_num_threads(max(1, min(n, numba.config.NUMBA_NUM_THREADS)))


if USE_NUMBA:
    _GAMMA, _K, _RECOIL = c.GAMMA, c.WAVENUMBER, c.RECOIL_VELOCITY

    @numba.njit(inline="always")
    def _uniform(state, i):
        """Next number in [0, 1) of atom i's xorshift64* stream."""
        s = state[i]
        s ^= s >> np.uint64(12)
        s ^= s << np.uint64(25)
        s ^= s >> np.uint64(27)
        state[i] = s
        return ((s * np.uint64(0x2545F4914F6CDD1D)) >> np.uint64(11)) * (1.0 / 9007199254740992.0)

    @numba.njit(inline="always")
    def _poisson(mean, state, i):
        """Poisson draw by multiplying uniforms (Knuth); the means here are a few photons at most."""
        if mean <= 0.0:
            return 0
        limit = np.exp(-mean)
        k = 0
        product = _uniform(state, i)
        while product > limit:
            k += 1
            product *= _uniform(state, i)
        return k

    @numba.njit(parallel=True, cache=True)
    def _advance(x, v, n_steps, dt, directions, s_beam, inverse_waist_sq, detuning, zeeman, field_zero,
                 transmission, acceleration, brightness, noise, state, photons):
        n, dims = x.shape
        n_beams = directions.shape[0]
        for i in numba.prange(n):
            s = np.empty(n_beams)
            kick = np.empty(3)
            for _ in range(n_steps):
                s_total = 0.0
                for j in range(n_beams):
                    exponent = 0.0
                    for a in range(dims):
                        exponent += x[i, a] * x[i, a] * inverse_waist_sq[j, a]
                    s[j] = s_beam[j] * np.exp(-2.0 * exponent) * transmission[i, j]
                    s_total += s[j]

                kick[:] = 0.0
                for j in range(n_beams):
                    shift = 0.0
                    for a in range(dims):
                        shift += directions[j, a] * (_K * v[i, a] + (x[i, a] - field_zero[a]) * zeeman[a])
                    off = 2.0 * (detuning[j] - shift) / _GAMMA
                    mean = 0.5 * _GAMMA * s[j] / (1.0 + s_total + off * off) * brightness[i] * dt
                    if not noise:
                        for a in range(dims):
                            kick[a] += mean * directions[j, a]
                        continue
                    count = _poisson(mean, state, i)
                    photons[i] += count
                    for _photon in range(count):
                        for a in range(dims):
                            kick[a] += directions[j, a]
                        cos_theta = 2.0 * _uniform(state, i) - 1.0
                        phi = 2.0 * np.pi * _uniform(state, i)
                        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
                        kick[0] += sin_theta * np.cos(phi)
                        kick[1] += sin_theta * np.sin(phi)
                        kick[2] += cos_theta

                for a in range(dims):
                    v[i, a] += _RECOIL * kick[a] + acceleration[i, a] * dt
                    x[i, a] += v[i, a] * dt

    @numba.njit(cache=True)
    def _interpolate(grid, u):
        nx, ny, nz = grid.shape
        value = np.zeros(u.shape[0])
        for k in range(u.shape[0]):
            i0, j0, l0 = int(np.floor(u[k, 0])), int(np.floor(u[k, 1])), int(np.floor(u[k, 2]))
            fx, fy, fz = u[k, 0] - i0, u[k, 1] - j0, u[k, 2] - l0
            total = 0.0
            for di in range(2):
                i = i0 + di
                if i < 0 or i >= nx:
                    continue
                wx = fx if di else 1.0 - fx
                for dj in range(2):
                    j = j0 + dj
                    if j < 0 or j >= ny:
                        continue
                    wy = fy if dj else 1.0 - fy
                    for dl in range(2):
                        l = l0 + dl
                        if l < 0 or l >= nz:
                            continue
                        total += wx * wy * (fz if dl else 1.0 - fz) * grid[i, j, l]
            value[k] = total
        return value
