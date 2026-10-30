"""Many atoms together: shadowing, rescattered light and light-assisted collisions (3D MOT).

A small MOT is a gas of independent atoms. A large one is not:

- Shadowing. Atoms upstream absorb from each beam, so the atoms behind them
  see less light. Pairs of opposing beams then push the cloud inwards.
- Rescattering. A photon scattered by one atom can be absorbed by another,
  pushing them apart with a force that falls as 1/r², like a Coulomb force.

With σ_L the absorption cross-section for laser light and σ_R that for
rescattered light, the two add up to a mean-field force of divergence
(Walker, Sesko & Wieman, PRL 64, 408 (1990))

    ∇·F = 6 ħk R n (σ_R − σ_L),

where R is the scattering rate per beam. When σ_R > σ_L the cloud repels
itself. Once this outweighs the trap, ∇·F_trap = −Σκ_i, adding atoms makes the
cloud larger instead of denser: the density saturates at

    n_max = Σκ_i / (6 ħk R (σ_R − σ_L)).

Each simulated atom stands for N/n_sim real atoms. Both forces come from the
density of the cloud on a grid: the shadows from column densities along each
beam, the rescattering force from a convolution with the 1/r² law.

The rescattering force is conservative, and on its own leaves the temperature
at the Doppler value. The shadow force is not: its curl does not vanish in a
non-uniform cloud, so with friction it drives circulating flows, and the
velocity spread of a large cloud grows (Sesko et al. saw such flows as
orbiting rings in large MOTs).
"""

from dataclasses import dataclass

import numpy as np

from . import constants as c
from . import theory
from .fast import advance, interpolate
from .parameters import MOTParameters
from .physics import scattering_rates
from .simulation import SimulationResult, SimulationSettings, doppler_steady_state

RESCATTER_RATIO = 1.3    # σ_R / σ_L (Sesko, Walker & Wieman, JOSA B 8, 946 (1991): 1.2–1.4)
TWO_BODY_LOSS = 5e-18    # light-assisted collision rate coefficient β [m³/s] (5e-12 cm³/s, typical for Rb)


def absorption_cross_section(p: MOTParameters) -> float:
    """σ_L = σ0 / (1 + s_tot + (2δ/Γ)²) for laser light, at the trap centre [m²]."""
    return c.RESONANT_CROSS_SECTION / (1 + 2 * p.dimensions * p.saturation + (2 * p.detuning) ** 2)


def density_limit(p: MOTParameters, ratio: float = RESCATTER_RATIO) -> float:
    """Maximum density n_max = Σκ_i / (2D ħk R σ_L (σ_R/σ_L − 1)) [m⁻³]."""
    push = 2 * p.dimensions * c.HBAR * c.WAVENUMBER * theory.scattering_rate_per_beam(p)
    return theory.spring_constants(p).sum() / (push * absorption_cross_section(p) * (ratio - 1))


def depolarisation_factors(p: MOTParameters, ratio: float = RESCATTER_RATIO) -> np.ndarray:
    """Shape factors N_i of the uniform cloud at the density limit (Σ N_i = 1).

    Inside a uniform cloud of density n, with a = ħk R n:
      shadows      F_i = −2σ_L a x_i                  (each pair of beams separately)
      rescattering F_i = 2D σ_R a N_i x_i             (field inside a uniform ellipsoid)
    Balancing the trap −κ_i x_i on every axis fixes N_i.
    """
    kappa = theory.spring_constants(p)
    d2 = 2 * p.dimensions
    return (d2 * (ratio - 1) * kappa / kappa.sum() + 2) / (d2 * ratio)


def _axial_factor(aspect: float) -> float:
    """Depolarisation factor along the symmetry axis of a spheroid with c/a = aspect."""
    if np.isclose(aspect, 1):
        return 1 / 3
    if aspect < 1:  # oblate
        e = np.sqrt(1 - aspect**2)
        return (1 - np.sqrt(1 - e**2) * np.arcsin(e) / e) / e**2
    e = np.sqrt(1 - 1 / aspect**2)  # prolate
    return (1 - e**2) / e**2 * (np.arctanh(e) / e - 1)


def density_limited_sizes(p: MOTParameters, atom_number, ratio: float = RESCATTER_RATIO) -> np.ndarray:
    """RMS sizes of the uniform spheroid holding N atoms at n_max, shape (len(N), 3) [m].

    The spheroid's aspect ratio c/a is the one whose axial factor is N_z; its
    volume is N/n_max, and a uniform ellipsoid has RMS size (semi-axis)/√5.
    """
    if p.dimensions != 3:
        raise ValueError("the density limit is modelled for the 3D MOT")
    target = depolarisation_factors(p, ratio)[2]
    lo, hi = 1e-3, 1e3
    for _ in range(100):  # N_z falls as the spheroid gets longer
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if _axial_factor(mid) > target else (lo, mid)
    aspect = np.sqrt(lo * hi)
    volume = np.asarray(atom_number, dtype=float) / density_limit(p, ratio)
    a = (volume / (4 / 3 * np.pi * aspect)) ** (1 / 3)
    return np.stack([a, a, aspect * a], axis=-1) / np.sqrt(5)


def temperature_limited_density(p: MOTParameters, atom_number) -> np.ndarray:
    """Peak density N / ((2π)^{3/2} σ_x σ_y σ_z) of the Gaussian cloud of Doppler theory [m⁻³]."""
    return np.asarray(atom_number, dtype=float) / ((2 * np.pi) ** 1.5 * np.prod(theory.cloud_sizes(p)))


# --- Mean-field forces ---------------------------------------------------------------

class MeanField:
    """Density of the cloud on a fixed grid, and the forces it exerts.

    The simulated atoms are a sample of the density; binned and blurred by
    one cell, they give a smooth density. The grid spans ±5 times the larger
    of the Doppler and the density-limited cloud size.
    """

    def __init__(self, p: MOTParameters, atom_number: float, n_sim: int, ratio: float,
                 shadows: bool = True, cells: int = 40, blur_cells: float = 1.0):
        net_ratio = ratio if shadows else ratio + 1  # same total repulsion without the shadows
        size = np.maximum(theory.cloud_sizes(p), density_limited_sizes(p, atom_number, net_ratio))
        center = theory.equilibrium_position(p)
        self.edges = [np.linspace(m - 5 * h, m + 5 * h, cells + 1) for m, h in zip(center, size)]
        self.widths = np.array([e[1] - e[0] for e in self.edges])
        self.origin = np.array([e[0] for e in self.edges]) + 0.5 * self.widths  # first cell centre
        self.weight = atom_number / n_sim
        self.sigma_l = absorption_cross_section(p)
        self.sigma_r = ratio * self.sigma_l
        self.shadows = shadows
        self.density = self.source = None

        offsets = np.arange(2 * cells)
        offsets = np.where(offsets < cells, offsets, offsets - 2 * cells)
        d = np.meshgrid(*[offsets * w for w in self.widths], indexing="ij")
        r3 = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 1.5
        r3[0, 0, 0] = np.inf  # a cell exerts no net force on itself
        self._kernels = [np.fft.rfftn(component / r3) for component in d]
        index = np.arange(cells)
        self._blur = np.exp(-0.5 * ((index[:, None] - index[None, :]) / blur_cells) ** 2)
        self._blur /= self._blur.sum(axis=0, keepdims=True)

    def _grid(self, x, weights=None):
        grid, _ = np.histogramdd(x, bins=self.edges, weights=weights)
        for axis in range(3):
            grid = np.moveaxis(np.tensordot(self._blur, np.moveaxis(grid, axis, 0), axes=1), 0, axis)
        return grid

    def update(self, x: np.ndarray, total_rates: np.ndarray):
        """Bin the atoms (for the shadows) and their scattering rates (for the rescattered light)."""
        self.density, self.source = self._grid(x), self._grid(x, total_rates)

    def _at(self, grid, x):
        return interpolate(grid, (x - self.origin) / self.widths)

    def transmission(self, x: np.ndarray) -> np.ndarray:
        """Fraction of each beam reaching each atom, exp(−σ_L · column density upstream), (n, 6)."""
        result = np.ones((len(x), 6))
        if not self.shadows:
            return result
        for axis in range(3):
            area = np.prod(np.delete(self.widths, axis))
            forward = np.cumsum(self.density, axis=axis) - 0.5 * self.density  # light along +axis
            backward = np.flip(np.cumsum(np.flip(self.density, axis), axis=axis), axis) - 0.5 * self.density
            for column, beam in ((forward, 2 * axis), (backward, 2 * axis + 1)):
                result[:, beam] = np.exp(-self.sigma_l * self.weight * self._at(column, x) / area)
        return result

    def rescattering_acceleration(self, x: np.ndarray) -> np.ndarray:
        """a = Σ_k ħk σ_R R_k (r − r_k) / (4π m |r − r_k|³) over the real atoms k [m/s²].

        The sum over grid cells is a convolution, done by FFT.
        """
        n = self.source.shape[0]
        source = np.fft.rfftn(self.source, s=(2 * n,) * 3, axes=(0, 1, 2))
        scale = c.HBAR * c.WAVENUMBER * self.sigma_r * self.weight / (4 * np.pi * c.MASS)
        field = [np.fft.irfftn(source * k, s=(2 * n,) * 3, axes=(0, 1, 2))[:n, :n, :n] for k in self._kernels]
        return scale * np.stack([self._at(f, x) for f in field], axis=1)


@dataclass(frozen=True)
class DensitySettings:
    n_sim: int = 1500
    duration: float = 14e-3
    dt: float = 1e-6
    update_interval: float = 50e-6   # how often the mean field is recomputed [s]
    snapshot_interval: float = 0.1e-3
    rescatter_ratio: float = RESCATTER_RATIO
    shadows: bool = True             # False: rescattering only, a conservative force
    seed: int = 7


def run(p: MOTParameters, atom_number: float, s: DensitySettings = DensitySettings()) -> SimulationResult:
    """N real atoms, represented by n_sim simulated ones, starting from the Doppler steady state."""
    if p.dimensions != 3:
        raise ValueError("density effects are modelled for the 3D MOT")
    rng = np.random.default_rng(s.seed)
    x, v = doppler_steady_state(p, s.n_sim, rng)
    field = MeanField(p, atom_number, s.n_sim, s.rescatter_ratio, s.shadows)

    update_every = max(1, round(s.update_interval / s.dt))
    steps = max(1, round(s.snapshot_interval / s.dt))
    n_snapshots = max(1, round(s.duration / (steps * s.dt)))
    positions = np.empty((n_snapshots + 1, s.n_sim, 3))
    velocities = np.empty_like(positions)
    for i in range(n_snapshots):
        positions[i], velocities[i] = x, v
        done = 0
        while done < steps:  # the mean field is held fixed for update_every steps
            if field.density is None:
                field.update(x, scattering_rates(x, v, p).sum(axis=1))
            field.update(x, scattering_rates(x, v, p, field.transmission(x)).sum(axis=1))
            chunk = min(update_every, steps - done)
            advance(x, v, p, chunk, s.dt, rng, transmission=field.transmission(x),
                    acceleration=field.rescattering_acceleration(x))
            done += chunk
    positions[-1], velocities[-1] = x, v

    times = np.arange(n_snapshots + 1) * steps * s.dt
    settings = SimulationSettings(n_atoms=s.n_sim, duration=s.duration, dt=s.dt,
                                  snapshot_interval=s.snapshot_interval, seed=s.seed)
    return SimulationResult(p, settings, times, positions, velocities)


def central_density(steady: SimulationResult, atom_number: float) -> float:
    """Density of real atoms inside the ellipsoid of half the RMS size around the centre [m⁻³]."""
    x = steady.positions
    scaled = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    inside = (np.einsum("tij,tij->ti", scaled, scaled) < 0.25).mean()
    volume = 4 / 3 * np.pi * 0.5**3 * np.prod(x.std(axis=1), axis=-1).mean()
    return atom_number * inside / volume


# --- Number of atoms: loading against losses ------------------------------------------

def mean_density(p: MOTParameters, atom_number, ratio: float = RESCATTER_RATIO) -> np.ndarray:
    """∫n² d³r / N, the density felt by the average atom, from small to large N [m⁻³].

    A Gaussian cloud gives n_peak / 2^{3/2}; a uniform one n_max. The two limits
    are joined harmonically.
    """
    gaussian = temperature_limited_density(p, atom_number) / 2**1.5
    return 1 / (1 / gaussian + 1 / density_limit(p, ratio))


def loading_curve(p: MOTParameters, rate: float, background_loss: float, times,
                  beta: float = TWO_BODY_LOSS, ratio: float = RESCATTER_RATIO) -> np.ndarray:
    """N(t) from dN/dt = R_L − Γ_bg N − β N n̄(N), starting empty."""
    times = np.asarray(times, dtype=float)
    n = np.zeros_like(times)
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        loss = background_loss + beta * mean_density(p, max(n[i - 1], 1.0), ratio)
        n[i] = rate / loss + (n[i - 1] - rate / loss) * np.exp(-loss * dt)
    return n


def steady_atom_number(p: MOTParameters, rate: float, background_loss: float,
                       beta: float = TWO_BODY_LOSS, ratio: float = RESCATTER_RATIO) -> float:
    """N where loading balances losses, R_L = (Γ_bg + β n̄(N)) N."""
    lo, hi = 0.0, rate / background_loss
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rate > (background_loss + beta * mean_density(p, max(mid, 1.0), ratio)) * mid:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
