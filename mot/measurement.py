"""What an experiment would measure: time-of-flight expansion and absorption images."""

from dataclasses import dataclass

import numpy as np

from . import constants as c
from .simulation import SimulationResult


def free_expansion(positions: np.ndarray, velocities: np.ndarray, gravity: np.ndarray,
                   t: float) -> np.ndarray:
    """Positions after the light and field are switched off for a time t."""
    return positions + velocities * t + 0.5 * gravity * t**2


@dataclass
class TimeOfFlight:
    times: np.ndarray          # expansion times [s]
    sizes: np.ndarray          # RMS size at each time, (n_times, D) [m]
    centers: np.ndarray        # cloud centre at each time, (n_times, D) [m]
    mean_velocity: np.ndarray  # mean velocity of the sample at release, (D,) [m/s]
    temperature: np.ndarray    # fitted temperature per axis [K]
    initial_size: np.ndarray   # fitted in-trap size per axis [m]


def time_of_flight(steady: SimulationResult, times) -> TimeOfFlight:
    """Release the steady-state cloud and fit σ²(t) = σ0² + (k_B T / m) t² on each axis.

    This is how temperatures are measured in the lab: the slope of σ² against t²
    gives the velocity spread, independently of the trap.
    """
    dims = steady.params.dimensions
    x0 = steady.positions.reshape(-1, dims)  # pool all steady-state snapshots
    v0 = steady.velocities.reshape(-1, dims)
    times = np.asarray(times, dtype=float)

    clouds = [free_expansion(x0, v0, steady.params.gravity_vector, t) for t in times]
    sizes = np.array([cloud.std(axis=0) for cloud in clouds])
    centers = np.array([cloud.mean(axis=0) for cloud in clouds])

    slope, intercept = np.polyfit(times**2, sizes**2, 1)
    return TimeOfFlight(times, sizes, centers, v0.mean(axis=0), c.MASS * slope / c.K_B, np.sqrt(intercept))


@dataclass
class AbsorptionImage:
    optical_depth: np.ndarray  # OD = −ln(I/I0), shape (n_rows, n_cols); rows run along the vertical axis
    extent: tuple              # (left, right, bottom, top) [m]
    pixel_size: float          # [m]

    def atom_number(self) -> float:
        """Atoms in the image, N = Σ OD · A_pixel / σ0."""
        return self.optical_depth.sum() * self.pixel_size**2 / c.RESONANT_CROSS_SECTION


def absorption_image(positions: np.ndarray, n_atoms: float, axes=(0, 2), center=(0.0, 0.0),
                     half_width: float = 3e-3, pixel_size: float = 20e-6, resolution: float = 15e-6,
                     probe_counts: float | None = None, rng: np.random.Generator | None = None
                     ) -> AbsorptionImage:
    """Image of the cloud taken with a resonant probe beam.

    The simulated atoms are samples of the density: each stands for
    n_atoms / len(positions) real atoms. Their column density along the probe,
    blurred by the imaging resolution, gives the optical depth OD = σ0 · n_column.
    With `probe_counts` photons per pixel, camera shot noise is added to both the
    image with atoms and the reference image without.
    """
    n_pixels = int(round(2 * half_width / pixel_size))
    edges_h = center[0] + np.linspace(-half_width, half_width, n_pixels + 1)
    edges_v = center[1] + np.linspace(-half_width, half_width, n_pixels + 1)
    counts, _, _ = np.histogram2d(positions[:, axes[1]], positions[:, axes[0]], bins=(edges_v, edges_h))

    column_density = counts * (n_atoms / len(positions)) / pixel_size**2
    od = c.RESONANT_CROSS_SECTION * _gaussian_blur(column_density, resolution / pixel_size)

    if probe_counts is not None:
        rng = rng or np.random.default_rng()
        with_atoms = rng.poisson(probe_counts * np.exp(-od))
        reference = rng.poisson(probe_counts, od.shape)
        od = -np.log(np.maximum(with_atoms, 1) / np.maximum(reference, 1))

    extent = (edges_h[0], edges_h[-1], edges_v[0], edges_v[-1])
    return AbsorptionImage(od, extent, pixel_size)


def _gaussian_blur(image: np.ndarray, sigma_pixels: float) -> np.ndarray:
    """Separable Gaussian blur that conserves the total."""
    if sigma_pixels <= 0:
        return image
    half = int(np.ceil(4 * sigma_pixels))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_pixels) ** 2)
    kernel /= kernel.sum()
    blurred = np.apply_along_axis(np.convolve, 0, image, kernel, mode="same")
    return np.apply_along_axis(np.convolve, 1, blurred, kernel, mode="same")
