"""Figures and animations."""

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import constants as c
from . import theory
from .measurement import AbsorptionImage, TimeOfFlight
from .parameters import MOTParameters
from .physics import mean_force, scattering_rates
from .simulation import SimulationResult
from .studies import Scan

# --- Palette -----------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8b8a85"
GRID = "#e6e5e1"
BLUE = "#2a78d6"     # x axis
ORANGE = "#eb6834"   # y axis
AQUA = "#1baf7a"     # z axis
RED = "#e34948"
NEUTRAL = "#f0efec"

AXIS_COLORS = (BLUE, ORANGE, AQUA)
AXIS_NAMES = ("x", "y", "z")

FORCE_CMAP = LinearSegmentedColormap.from_list("force", [BLUE, NEUTRAL, RED])
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "density", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

F_MAX = c.HBAR * c.WAVENUMBER * c.GAMMA / 2  # maximum force from one saturated beam

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "text.color": INK,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
})


# --- Helpers ----------------------------------------------------------------

def _theory_line(ax, value, label, side="right", color=INK):
    """Thin horizontal reference line with a direct label above one end."""
    ax.axhline(value, color=color, lw=1, zorder=1)
    x, dx = (1, -4) if side == "right" else (0, 4)
    ax.annotate(label, (x, value), xycoords=("axes fraction", "data"), xytext=(dx, 4),
                textcoords="offset points", ha=side, va="bottom", color=INK, fontsize=9)


def _distinct_axes(p: MOTParameters) -> list[int]:
    """Axes with different physics: x only in 2D; x (radial) and z (axial) in 3D."""
    return [0] if p.dimensions == 2 else [0, 2]


def _axis_label(p: MOTParameters, axis: int) -> str:
    if p.dimensions == 2:
        return AXIS_NAMES[axis]
    return {0: "x, y (radial)", 1: "y", 2: "z (axial)"}[axis]


def _along(p: MOTParameters, axis: int, values: np.ndarray) -> np.ndarray:
    """Vectors of length D that are zero except along `axis`."""
    vectors = np.zeros((len(values), p.dimensions))
    vectors[:, axis] = values
    return vectors


def _force_map(p: MOTParameters, x_range, v_range, n=161):
    """Mean force F_x on a grid of (x, v_x), atoms on the x axis."""
    x = np.linspace(*x_range, n)
    v = np.linspace(*v_range, n)
    X, V = np.meshgrid(x, v)
    F = mean_force(_along(p, 0, X.ravel()), _along(p, 0, V.ravel()), p)[:, 0]
    return x, v, F.reshape(X.shape)


def _draw_phase_space(ax, p: MOTParameters, x_mm, v_ms):
    x, v, F = _force_map(p, (-x_mm * 1e-3, x_mm * 1e-3), (-v_ms, v_ms))
    limit = np.abs(F).max() / F_MAX
    mesh = ax.pcolormesh(x * 1e3, v, F / F_MAX, cmap=FORCE_CMAP, shading="auto",
                         norm=TwoSlopeNorm(0, -limit, limit), rasterized=True)
    # Flow of the deterministic motion: dx/dt = v (in mm/s), dv/dt = F/m.
    X, V = np.meshgrid(x * 1e3, v)
    ax.streamplot(X, V, V * 1e3, F / c.MASS, color=INK_SECONDARY, linewidth=0.6,
                  density=1.1, arrowsize=0.8)
    ax.set_xlim(-x_mm, x_mm)
    ax.set_ylim(-v_ms, v_ms)
    ax.grid(False)
    ax.set_xlabel("position x (mm)")
    ax.set_ylabel("velocity $v_x$ (m/s)")
    return mesh


def _parameter_caption(p: MOTParameters) -> str:
    gradient = "axial " if p.dimensions == 3 else ""
    return (f"Rb-87 D2 line  ·  {p.dimensions}D MOT  ·  δ = {p.detuning:g} Γ  ·  "
            f"s0 = {p.saturation:g} per beam  ·  w = {p.beam_waist * 1e3:g} mm  ·  "
            f"{gradient}B' = {p.field_gradient:g} G/cm")


def _side_view_axes(p: MOTParameters) -> tuple[int, int]:
    """Horizontal and vertical axis of the picture: x–z (side view) in 3D, x–y in 2D."""
    return (0, 2) if p.dimensions == 3 else (0, 1)


def _ellipse(ax, center, radii, label):
    angle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(center[0] + radii[0] * np.cos(angle), center[1] + radii[1] * np.sin(angle), color=INK, lw=1)
    ax.annotate(label, (center[0] + radii[0] / np.sqrt(2), center[1] + radii[1] / np.sqrt(2)),
                xytext=(3, 3), textcoords="offset points", fontsize=9, color=INK)


# --- Figure: the force --------------------------------------------------------------

def plot_force_model(p: MOTParameters, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    # (a) Doppler friction: force against velocity at the centre.
    ax = axes[0]
    v = np.linspace(-12, 12, 600)
    pos, vel = np.zeros((v.size, p.dimensions)), _along(p, 0, v)
    push = c.HBAR * c.WAVENUMBER * scattering_rates(pos, vel, p)
    total = mean_force(pos, vel, p)[:, 0]
    ax.plot(v, push[:, 0] / F_MAX, color=BLUE, lw=1.5, label="beam along +x")
    ax.plot(v, -push[:, 1] / F_MAX, color=ORANGE, lw=1.5, label="beam along −x")
    ax.plot(v, total / F_MAX, color=INK, label="total")
    ax.plot(v, -theory.damping_coefficient(p) * v / F_MAX, color=INK_MUTED, lw=1, label=r"linear: $-\beta v$")
    ax.set_ylim(1.4 * total.min() / F_MAX, 1.4 * total.max() / F_MAX)
    ax.set_title("a  Doppler cooling: friction from the Doppler shift")
    ax.set_xlabel("velocity $v_x$ (m/s)")
    ax.set_ylabel(r"force $F_x$ / ($\hbar k\Gamma/2$)")
    ax.legend(loc="upper right", fontsize=9)

    # (b) MOT trapping: force against displacement along each distinct axis.
    ax = axes[1]
    x = np.linspace(-15e-3, 15e-3, 600)
    kappa = theory.spring_constants(p)
    extreme = 0
    for axis in _distinct_axes(p):
        force = mean_force(_along(p, axis, x), np.zeros((x.size, p.dimensions)), p)[:, axis]
        extreme = max(extreme, np.abs(force).max())
        ax.plot(x * 1e3, force / F_MAX, color=AXIS_COLORS[axis], label=f"along {_axis_label(p, axis)}")
        ax.plot(x * 1e3, -kappa[axis] * x / F_MAX, color=AXIS_COLORS[axis], lw=1, alpha=0.6)
    ax.set_ylim(-1.4 * extreme / F_MAX, 1.4 * extreme / F_MAX)
    ax.set_title("b  MOT trap: restoring force from the Zeeman shift")
    ax.set_xlabel("displacement from the centre (mm)")
    ax.set_ylabel(r"force / ($\hbar k\Gamma/2$)")
    ax.legend(loc="upper right", fontsize=9, title="thin lines: linear −κx", title_fontsize=8.5)

    # (c) Both together: phase-space flow.
    ax = axes[2]
    mesh = _draw_phase_space(ax, p, x_mm=12, v_ms=12)
    fig.colorbar(mesh, ax=ax, label=r"$F_x$ / ($\hbar k\Gamma/2$)", shrink=0.9)
    ax.set_title("c  Phase space: every path spirals into the centre")

    fig.suptitle(_parameter_caption(p), x=0.01, ha="left", color=INK_SECONDARY, fontsize=9.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Figure: cooling dynamics -------------------------------------------------------

def plot_cooling_dynamics(result: SimulationResult, path: Path):
    p = result.params
    dims = range(p.dimensions)
    T_theory = theory.doppler_temperature(p)
    sigma_theory = theory.cloud_sizes(p)
    t_ms = result.times * 1e3

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    T = result.temperature() * 1e6
    for i in dims:
        ax.plot(t_ms, T[:, i], color=AXIS_COLORS[i], lw=1.6, label=f"$T_{AXIS_NAMES[i]}$")
    ax.set_yscale("log")
    _theory_line(ax, T_theory * 1e6, f"Doppler theory  {T_theory * 1e6:.0f} µK")
    ax.set_title("a  Temperature falls to the Doppler temperature")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("temperature (µK)")
    ax.legend(loc="upper center", ncol=p.dimensions)

    ax = axes[0, 1]
    size = result.rms_size() * 1e6
    for i in dims:
        ax.plot(t_ms, size[:, i], color=AXIS_COLORS[i], lw=1.6, label=rf"$\sigma_{AXIS_NAMES[i]}$")
    for axis in _distinct_axes(p):
        _theory_line(ax, sigma_theory[axis] * 1e6,
                     f"theory, {_axis_label(p, axis)}:  {sigma_theory[axis] * 1e6:.0f} µm")
    ax.set_yscale("log")
    ax.set_title("b  The cloud is compressed by the trap")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("RMS cloud size (µm)")
    ax.legend(loc="upper center", ncol=p.dimensions)

    steady = result.steady_state()
    h, v = _side_view_axes(p)
    ax = axes[1, 0]
    points = steady.positions.reshape(-1, p.dimensions) * 1e6
    center = theory.equilibrium_position(p) * 1e6
    half = 4 * sigma_theory.max() * 1e6
    ax.hist2d(points[:, h], points[:, v], bins=70,
              range=[[center[h] - half, center[h] + half], [center[v] - half, center[v] + half]],
              cmap=DENSITY_CMAP, rasterized=True)
    for n in (1, 2):
        _ellipse(ax, (center[h], center[v]), n * sigma_theory[[h, v]] * 1e6, f"{n}σ")
    ax.set_aspect("equal")
    ax.grid(False)
    view = "side view, " if p.dimensions == 3 else ""
    ax.set_title(f"c  Steady-state density ({view}theory ellipses)")
    ax.set_xlabel(f"{AXIS_NAMES[h]} (µm)")
    ax.set_ylabel(f"{AXIS_NAMES[v]} (µm)")

    ax = axes[1, 1]
    v0 = result.velocities[0].ravel()
    v1 = steady.velocities.ravel()
    bins = np.linspace(-3, 3, 121)
    ax.hist(v0, bins=bins, density=True, histtype="step", color=ORANGE, lw=1.5,
            label=f"start ({result.settings.initial_temperature * 1e3:.0f} mK)")
    ax.hist(v1, bins=bins, density=True, color=BLUE, alpha=0.85, label="steady state")
    sigma_v = np.sqrt(c.K_B * T_theory / c.MASS)
    grid = np.linspace(-3, 3, 500)
    ax.plot(grid, np.exp(-grid**2 / (2 * sigma_v**2)) / np.sqrt(2 * np.pi * sigma_v**2), color=INK, lw=1,
            label=f"Maxwell-Boltzmann, {T_theory * 1e6:.0f} µK")
    ax.set_title("d  Velocity distribution (all components)")
    ax.set_xlabel("velocity (m/s)")
    ax.set_ylabel("probability density (s/m)")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(_parameter_caption(p) + f"  ·  {result.settings.n_atoms} atoms",
                 x=0.01, ha="left", color=INK_SECONDARY, fontsize=9.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Figure: detuning scan ----------------------------------------------------------

def plot_detuning_scan(scan: Scan, base: MOTParameters, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    curve = np.linspace(scan.values.min() * 1.05, scan.values.max() * 0.95, 300)
    curve_params = [replace(base, detuning=d) for d in curve]

    ax = axes[0]
    ax.plot(curve, [theory.doppler_temperature(q) * 1e6 for q in curve_params],
            color=INK, lw=1.2, label="Doppler theory")
    T = scan.temperature.mean(axis=1)
    T_err = np.sqrt((scan.temperature_error**2).sum(axis=1)) / base.dimensions
    ax.errorbar(scan.values, T * 1e6, yerr=T_err * 1e6, fmt="o", ms=7, color=BLUE, mec=SURFACE,
                mew=1.5, elinewidth=1.2, label="simulation")
    d_opt = theory.optimal_detuning(base)
    T_opt = theory.doppler_temperature(replace(base, detuning=d_opt))
    ax.annotate(f"minimum {T_opt * 1e6:.0f} µK\nat δ = {d_opt:.2f} Γ", (d_opt, T_opt * 1e6),
                xytext=(0, -22), textcoords="offset points", ha="center", va="top", fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8))
    T_ideal = theory.ideal_doppler_limit() * 1e6
    _theory_line(ax, T_ideal, f"ideal limit ħΓ/2k$_B$ = {T_ideal:.0f} µK", side="left")
    ax.set_ylim(0, None)
    ax.set_title("a  Temperature against detuning")
    ax.set_xlabel("detuning δ (Γ)")
    ax.set_ylabel("steady-state temperature (µK)")
    ax.legend(loc="lower right")

    ax = axes[1]
    sizes = np.array([theory.cloud_sizes(q) for q in curve_params])
    for axis in _distinct_axes(base):
        color = AXIS_COLORS[axis]
        ax.plot(curve, sizes[:, axis] * 1e6, color=color, lw=1.2)
        ax.plot(scan.values, scan.size[:, axis] * 1e6, "o", ms=7, color=color, mec=SURFACE, mew=1.5,
                label=_axis_label(base, axis))
    ax.set_ylim(0, None)
    ax.set_title("b  Cloud size against detuning")
    ax.set_xlabel("detuning δ (Γ)")
    ax.set_ylabel("RMS cloud size (µm)")
    ax.legend(loc="upper right", title="dots: simulation, lines: theory", title_fontsize=8.5)

    caption = (f"{base.dimensions}D MOT  ·  s0 = {base.saturation:g} per beam  ·  B' = {base.field_gradient:g} G/cm"
               "  ·  each point: 500 atoms, averaged over the steady-state half of the run")
    fig.suptitle(caption, x=0.01, ha="left", color=INK_SECONDARY, fontsize=9.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Figure: time of flight ---------------------------------------------------------

def plot_time_of_flight(tof: TimeOfFlight, images: list[AbsorptionImage], p: MOTParameters, path: Path):
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.15])
    h, v = _side_view_axes(p)

    image_axes = top.subplots(1, len(images), sharey=True)
    for ax, image, t in zip(image_axes, images, tof.times):
        peak = np.percentile(image.optical_depth, 99.9)
        ax.imshow(image.optical_depth, origin="lower", extent=np.array(image.extent) * 1e3,
                  cmap=DENSITY_CMAP, vmin=0, vmax=peak)
        ax.grid(False)
        ax.set_title(f"t = {t * 1e3:g} ms", fontsize=10)
        ax.text(0.04, 0.04, f"peak OD ≈ {peak:.2f}", transform=ax.transAxes, fontsize=8.5, color=INK_SECONDARY)
        ax.set_xlabel(f"{AXIS_NAMES[h]} (mm)")
    image_axes[0].set_ylabel(f"{AXIS_NAMES[v]} (mm)")
    top.suptitle("a  Simulated absorption images after release (10⁶ atoms, camera shot noise, "
                 "colour scaled per image)", x=0.005, ha="left", fontsize=11, weight="bold")

    panels = bottom.subplots(1, 2 if p.dimensions == 3 else 1, squeeze=False)[0]
    ax = panels[0]
    t = np.linspace(0, tof.times.max(), 200)
    for axis in range(p.dimensions):
        ax.plot(tof.times**2 * 1e6, tof.sizes[:, axis] ** 2 * 1e6, "o", ms=7, color=AXIS_COLORS[axis],
                mec=SURFACE, mew=1.5, label=f"{AXIS_NAMES[axis]}:  fit T = {tof.temperature[axis] * 1e6:.0f} µK")
        fit = theory.expanded_size(tof.initial_size[axis], tof.temperature[axis], t)
        ax.plot(t**2 * 1e6, fit**2 * 1e6, color=AXIS_COLORS[axis], lw=1.2)
    ax.set_title(r"b  Temperature from the slope of $\sigma^2$ against $t^2$")
    ax.set_xlabel("$t^2$ (ms²)")
    ax.set_ylabel(r"$\sigma^2$ (mm²)")
    ax.legend(loc="upper left", title=f"Doppler theory: {theory.doppler_temperature(p) * 1e6:.0f} µK",
              title_fontsize=9)

    if p.dimensions == 3:
        # A real cloud of 10⁶ atoms has zero mean velocity; our 1000 samples do not quite,
        # so their small common drift v̄t is removed to show the free fall alone.
        ax = panels[1]
        drop = tof.centers[:, 2] - tof.centers[0, 2] - tof.mean_velocity[2] * tof.times
        ax.plot(t * 1e3, -0.5 * c.G_EARTH * t**2 * 1e6, color=INK, lw=1.2, label=r"free fall $-\frac{1}{2}gt^2$")
        ax.plot(tof.times * 1e3, drop * 1e6, "o", ms=7, color=AQUA, mec=SURFACE, mew=1.5,
                label=r"cloud centre z (sample drift $\bar v_z t$ removed)")
        ax.set_title("c  Gravity: the released cloud falls")
        ax.set_xlabel("time of flight (ms)")
        ax.set_ylabel("vertical displacement (µm)")
        ax.legend(loc="lower left")

    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- Figure: imperfections ----------------------------------------------------------

def plot_imperfections(imbalance: Scan, stray: Scan, gradient: Scan,
                       anti_trap: tuple[SimulationResult, SimulationResult], base: MOTParameters, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    def scan_panel(ax, scan, axis, make_params, scale, unit, xlabel, title, theory_x=None):
        x = np.linspace(scan.values.min(), scan.values.max(), 200) if theory_x is None else theory_x
        predicted = [theory.equilibrium_position(make_params(value))[axis] * scale for value in x]
        ax.plot(x, predicted, color=INK, lw=1.2, label="theory")
        ax.errorbar(scan.values, scan.center[:, axis] * scale, yerr=scan.center_error[:, axis] * scale,
                    fmt="o", ms=7, color=AXIS_COLORS[axis], mec=SURFACE, mew=1.5, elinewidth=1.2,
                    label="simulation")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"cloud centre {AXIS_NAMES[axis]} ({unit})")
        ax.legend(loc="best")

    scan_panel(axes[0, 0], imbalance, 0, lambda e: replace(base, power_imbalance=e), 1e3, "mm",
               "power imbalance ε  (+x beam: 1+ε, −x beam: 1−ε)",
               "a  Unequal beams push the cloud: Δx = 2εħkR/κ")
    scan_panel(axes[0, 1], stray, 0, lambda b: replace(base, stray_field=(b, 0.0, 0.0)), 1e3, "mm",
               "stray field along x (G)", "b  A stray field moves the field zero: Δx = −B/B'ₓ")
    gradients = np.geomspace(gradient.values.min(), gradient.values.max(), 200)
    scan_panel(axes[1, 0], gradient, 2, lambda g: replace(base, field_gradient=g), 1e6, "µm",
               "axial field gradient B' (G/cm)", "c  Gravity sags the cloud: Δz = −mg/κ_z",
               theory_x=gradients)

    ax = axes[1, 1]
    normal, reversed_ = anti_trap
    t_ms = normal.times * 1e3
    rates = theory.escape_rates(reversed_.params)
    for axis in _distinct_axes(base):
        color = AXIS_COLORS[axis]
        name = AXIS_NAMES[axis]
        ax.plot(t_ms, normal.rms_size()[:, axis] * 1e3, color=color, lw=1.2, alpha=0.5,
                label=rf"$\sigma_{name}$, correct field")
        grown = reversed_.rms_size()[:, axis] * 1e3
        ax.plot(t_ms, grown, color=color, lw=2, label=rf"$\sigma_{name}$, reversed field")
        anchor = len(t_ms) - 1
        ax.plot(t_ms, grown[anchor] * np.exp(rates[axis] * (normal.times - normal.times[anchor])),
                color=INK, lw=1)
    ax.set_yscale("log")
    ax.set_title("d  Reversed field: the trap becomes an anti-trap")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("RMS cloud size (mm)")
    ax.legend(loc="upper left", fontsize=9, title="black: theory growth $e^{\\lambda t}$", title_fontsize=8.5)

    fig.suptitle(_parameter_caption(base) + "  ·  each point: steady state of one simulation",
                 x=0.01, ha="left", color=INK_SECONDARY, fontsize=9.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Animations -----------------------------------------------------------------------

def make_animation(result: SimulationResult, path: Path, max_time=12e-3, n_shown=500, fps=10, stride=2):
    """Three panels: the atoms, a zooming view that follows the shrinking cloud, and T(t)."""
    p = result.params
    frames = np.flatnonzero(result.times <= max_time)[::stride]
    shown = slice(0, min(n_shown, result.settings.n_atoms))
    T = result.temperature() * 1e6
    T_theory = theory.doppler_temperature(p) * 1e6
    t_ms = result.times * 1e3
    lim_mm = 1.3 * result.settings.initial_radius * 1e3

    fig = plt.figure(figsize=(14, 4.6), constrained_layout=True)
    if p.dimensions == 3:
        ax_atoms = fig.add_subplot(1, 3, 1, projection="3d")
        _setup_3d_axes(ax_atoms, lim_mm)
        dots = ax_atoms.scatter([], [], [], s=3, color=BLUE, alpha=0.6, linewidths=0, depthshade=False)
    else:
        ax_atoms = fig.add_subplot(1, 3, 1)
        _setup_2d_axes(ax_atoms, lim_mm)
        dots = ax_atoms.scatter([], [], s=4, color=BLUE, alpha=0.6, linewidths=0)

    # Zoomed view (±4σ): phase space x–v_x in 2D, side view x–z in 3D.
    ax_zoom = fig.add_subplot(1, 3, 2)
    h, v = _side_view_axes(p)
    if p.dimensions == 3:
        ax_zoom.set_title("Side view, zooming with the cloud")
        ax_zoom.set_xlabel("x (mm)")
        ax_zoom.set_ylabel("z (mm)")
        ax_zoom.set_aspect("equal")
        span = 4 * result.rms_size().max(axis=1) * 1e3
        force_image = None
    else:
        ax_zoom.set_title("Phase space, zooming with the cloud (colour = mean force)")
        ax_zoom.set_xlabel("position x (mm)")
        ax_zoom.set_ylabel("velocity $v_x$ (m/s)")
        x_span = 4 * result.rms_size()[:, 0] * 1e3
        v_span = 4 * result.velocities[:, :, 0].std(axis=1)
        force_image = ax_zoom.imshow(np.zeros((2, 2)), origin="lower", aspect="auto", cmap=FORCE_CMAP,
                                     alpha=0.55, interpolation="bilinear")
        ax_zoom.grid(False)
    zoom_dots = ax_zoom.scatter([], [], s=4, color=INK, alpha=0.6, linewidths=0)

    ax_T = fig.add_subplot(1, 3, 3)
    ax_T.set_xlim(0, t_ms[frames[-1]])
    ax_T.set_ylim(0.5 * T_theory, 1.5 * T.max())
    ax_T.set_yscale("log")
    ax_T.set_xlabel("time (ms)")
    ax_T.set_ylabel("temperature (µK)")
    ax_T.set_title("Temperature")
    _theory_line(ax_T, T_theory, f"Doppler theory {T_theory:.0f} µK")
    lines_T = [ax_T.plot([], [], color=AXIS_COLORS[i], lw=1.5, label=f"$T_{AXIS_NAMES[i]}$")[0]
               for i in range(p.dimensions)]
    ax_T.legend(loc="upper right", ncol=p.dimensions)
    clock = fig.suptitle("", x=0.01, ha="left", fontsize=11)

    def update(k):
        frame = frames[k]
        pos = result.positions[frame, shown] * 1e3
        if p.dimensions == 3:
            dots._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
            ax_atoms.view_init(elev=18, azim=-60 + 0.6 * k)
            center = result.center()[frame] * 1e3
            ax_zoom.set_xlim(center[h] - span[frame], center[h] + span[frame])
            ax_zoom.set_ylim(center[v] - span[frame], center[v] + span[frame])
            zoom_dots.set_offsets(pos[:, [h, v]])
        else:
            dots.set_offsets(pos)
            xs, vs = x_span[frame], v_span[frame]
            _, _, F = _force_map(p, (-xs * 1e-3, xs * 1e-3), (-vs, vs), n=81)
            limit = np.abs(F).max()
            force_image.set_data(F)
            force_image.set_norm(TwoSlopeNorm(0, -limit, limit))
            force_image.set_extent((-xs, xs, -vs, vs))
            ax_zoom.set_xlim(-xs, xs)
            ax_zoom.set_ylim(-vs, vs)
            zoom_dots.set_offsets(np.column_stack((pos[:, 0], result.velocities[frame, shown, 0])))
        for i, line in enumerate(lines_T):
            line.set_data(t_ms[: frame + 1], T[: frame + 1, i])
        clock.set_text(f"t = {t_ms[frame]:5.1f} ms     T = {T[frame].mean():7.0f} µK")

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


def _setup_2d_axes(ax, lim_mm):
    ax.set_xlim(-lim_mm, lim_mm)
    ax.set_ylim(-lim_mm, lim_mm)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Atoms in the plane (red: laser beams)")
    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        start = -0.97 * lim_mm * np.array(d)
        ax.annotate("", xy=start + 0.18 * lim_mm * np.array(d), xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5))


def _setup_3d_axes(ax, lim_mm):
    for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_lim(-lim_mm, lim_mm)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("Atoms in 3D (red: the six beams)")
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color(SURFACE)
        pane._axinfo["grid"]["color"] = GRID
    length = 0.25 * lim_mm
    for axis in range(3):
        for sign in (1, -1):
            start = np.zeros(3)
            start[axis] = -sign * lim_mm
            direction = np.zeros(3)
            direction[axis] = sign * length
            ax.quiver(*start, *direction, color=RED, lw=1.5, arrow_length_ratio=0.35)
