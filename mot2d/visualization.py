"""Figures and animation."""

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
from .parameters import MOTParameters
from .physics import mean_force, scattering_rates
from .simulation import DetuningScan, SimulationResult

# --- Palette -----------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8b8a85"
GRID = "#e6e5e1"
BLUE = "#2a78d6"     # x axis / series 1
ORANGE = "#eb6834"   # y axis / series 2
RED = "#e34948"
NEUTRAL = "#f0efec"

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


def _theory_line(ax, value, label, side="right"):
    """Thin horizontal reference line with a direct label above one end."""
    ax.axhline(value, color=INK, lw=1, zorder=1)
    x, dx = (1, -4) if side == "right" else (0, 4)
    ax.annotate(label, (x, value), xycoords=("axes fraction", "data"), xytext=(dx, 4),
                textcoords="offset points", ha=side, va="bottom", color=INK, fontsize=9)


def _force_map(p: MOTParameters, x_range, v_range, n=161):
    """Mean force F_x/F_max on a grid of (x, v_x), atoms on the y = 0 axis."""
    x = np.linspace(*x_range, n)
    v = np.linspace(*v_range, n)
    X, V = np.meshgrid(x, v)
    pos = np.column_stack((X.ravel(), np.zeros(X.size)))
    vel = np.column_stack((V.ravel(), np.zeros(V.size)))
    F = mean_force(pos, vel, p)[:, 0].reshape(X.shape)
    return x, v, F


def _draw_phase_space(ax, p: MOTParameters, x_mm, v_ms, alpha=1.0, streamlines=True):
    x, v, F = _force_map(p, (-x_mm * 1e-3, x_mm * 1e-3), (-v_ms, v_ms))
    limit = np.abs(F).max() / F_MAX
    mesh = ax.pcolormesh(x * 1e3, v, F / F_MAX, cmap=FORCE_CMAP, alpha=alpha, shading="auto",
                         norm=TwoSlopeNorm(0, -limit, limit), rasterized=True)
    if streamlines:
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


# --- Figure 1: the force ---------------------------------------------------------

def plot_force_model(p: MOTParameters, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    beta, kappa = theory.damping_coefficient(p), theory.spring_constant(p)

    # (a) Doppler friction: force against velocity at the centre.
    ax = axes[0]
    v = np.linspace(-12, 12, 600)
    vel = np.column_stack((v, np.zeros_like(v)))
    pos = np.zeros_like(vel)
    total = mean_force(pos, vel, p)[:, 0]
    push = c.HBAR * c.WAVENUMBER * scattering_rates(pos, vel, p)
    ax.plot(v, push[:, 0] / F_MAX, color=BLUE, lw=1.5, label="beam along +x")
    ax.plot(v, -push[:, 1] / F_MAX, color=ORANGE, lw=1.5, label="beam along −x")
    ax.plot(v, total / F_MAX, color=INK, label="total")
    ax.plot(v, -beta * v / F_MAX, color=INK_MUTED, lw=1, label=r"linear: $-\beta v$")
    ax.set_ylim(1.4 * total.min() / F_MAX, 1.4 * total.max() / F_MAX)
    ax.set_title("a  Doppler cooling: friction from the Doppler shift")
    ax.set_xlabel("velocity $v_x$ (m/s)")
    ax.set_ylabel(r"force $F_x$ / ($\hbar k\Gamma/2$)")
    ax.legend(loc="upper right", fontsize=9)

    # (b) MOT trapping: force against position at rest.
    ax = axes[1]
    x = np.linspace(-15e-3, 15e-3, 600)
    pos = np.column_stack((x, np.zeros_like(x)))
    vel = np.zeros_like(pos)
    total = mean_force(pos, vel, p)[:, 0]
    push = c.HBAR * c.WAVENUMBER * scattering_rates(pos, vel, p)
    ax.plot(x * 1e3, push[:, 0] / F_MAX, color=BLUE, lw=1.5, label="beam along +x")
    ax.plot(x * 1e3, -push[:, 1] / F_MAX, color=ORANGE, lw=1.5, label="beam along −x")
    ax.plot(x * 1e3, total / F_MAX, color=INK, label="total")
    ax.plot(x * 1e3, -kappa * x / F_MAX, color=INK_MUTED, lw=1, label=r"linear: $-\kappa x$")
    ax.set_ylim(1.4 * total.min() / F_MAX, 1.4 * total.max() / F_MAX)
    ax.set_title("b  MOT trap: restoring force from the Zeeman shift")
    ax.set_xlabel("position x (mm)")
    ax.set_ylabel(r"force $F_x$ / ($\hbar k\Gamma/2$)")
    ax.legend(loc="upper right", fontsize=9)

    # (c) Both together: phase-space flow.
    ax = axes[2]
    mesh = _draw_phase_space(ax, p, x_mm=12, v_ms=12)
    fig.colorbar(mesh, ax=ax, label=r"$F_x$ / ($\hbar k\Gamma/2$)", shrink=0.9)
    ax.set_title("c  Phase space: every path spirals into the centre")

    fig.suptitle(_parameter_caption(p), x=0.01, ha="left", color=INK_SECONDARY, fontsize=9.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Figure 2: what the simulation does ----------------------------------------

def plot_cooling_dynamics(result: SimulationResult, path: Path):
    p = result.params
    T_theory = theory.doppler_temperature(p)
    sigma_theory = theory.cloud_size(p)
    t_ms = result.times * 1e3

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    T = result.temperature() * 1e6
    ax.plot(t_ms, T[:, 0], color=BLUE, label="$T_x$")
    ax.plot(t_ms, T[:, 1], color=ORANGE, label="$T_y$")
    ax.set_yscale("log")
    _theory_line(ax, T_theory * 1e6, f"Doppler theory  {T_theory * 1e6:.0f} µK")
    ax.set_title("a  Temperature falls to the Doppler temperature")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("temperature (µK)")
    ax.legend(loc="upper center")

    ax = axes[0, 1]
    size = result.rms_size() * 1e6
    ax.plot(t_ms, size[:, 0], color=BLUE, label=r"$\sigma_x$")
    ax.plot(t_ms, size[:, 1], color=ORANGE, label=r"$\sigma_y$")
    ax.set_yscale("log")
    _theory_line(ax, sigma_theory * 1e6, rf"theory $\sqrt{{k_BT/\kappa}}$  {sigma_theory * 1e6:.0f} µm")
    ax.set_title("b  The cloud is compressed by the trap")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("RMS cloud size (µm)")
    ax.legend(loc="upper center")

    steady = result.steady_state()
    ax = axes[1, 0]
    xy = steady.positions.reshape(-1, 2) * 1e6
    half = 4 * sigma_theory * 1e6
    ax.hist2d(xy[:, 0], xy[:, 1], bins=70, range=[[-half, half], [-half, half]],
              cmap=DENSITY_CMAP, rasterized=True)
    angle = np.linspace(0, 2 * np.pi, 200)
    for n in (1, 2):
        r = n * sigma_theory * 1e6
        ax.plot(r * np.cos(angle), r * np.sin(angle), color=INK, lw=1)
        ax.annotate(f"{n}σ", (r / np.sqrt(2), r / np.sqrt(2)), xytext=(3, 3),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title("c  Steady-state density, with theoretical σ")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")

    ax = axes[1, 1]
    v0 = result.velocities[0].ravel()
    v1 = steady.velocities.reshape(-1)
    bins = np.linspace(-3, 3, 121)
    ax.hist(v0, bins=bins, density=True, histtype="step", color=ORANGE, lw=1.5,
            label=f"start ({result.settings.initial_temperature * 1e3:.0f} mK)")
    ax.hist(v1, bins=bins, density=True, color=BLUE, alpha=0.85, label="steady state")
    sigma_v = np.sqrt(c.K_B * T_theory / c.MASS)
    v = np.linspace(-3, 3, 500)
    ax.plot(v, np.exp(-v**2 / (2 * sigma_v**2)) / np.sqrt(2 * np.pi * sigma_v**2), color=INK, lw=1,
            label=f"Maxwell-Boltzmann, {T_theory * 1e6:.0f} µK")
    ax.set_title("d  Velocity distribution ($v_x$ and $v_y$)")
    ax.set_xlabel("velocity (m/s)")
    ax.set_ylabel("probability density (s/m)")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(_parameter_caption(p) + f"  ·  {result.settings.n_atoms} atoms",
                 x=0.01, ha="left", color=INK_SECONDARY, fontsize=9.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Figure 3: detuning scan --------------------------------------------------

def plot_detuning_scan(scan: DetuningScan, base: MOTParameters, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    curve = np.linspace(scan.detuning.min() * 1.05, scan.detuning.max() * 0.95, 300)
    curve_params = [replace(base, detuning=d) for d in curve]

    ax = axes[0]
    ax.plot(curve, [theory.doppler_temperature(q) * 1e6 for q in curve_params],
            color=INK, lw=1.2, label="Doppler theory")
    ax.errorbar(scan.detuning, scan.temperature * 1e6, yerr=scan.temperature_error * 1e6,
                fmt="o", ms=7, color=BLUE, mec=SURFACE, mew=1.5, elinewidth=1.2, label="simulation")
    d_opt = theory.optimal_detuning(base.saturation)
    T_opt = theory.doppler_temperature(replace(base, detuning=d_opt))
    ax.annotate(f"minimum {T_opt * 1e6:.0f} µK\nat δ = {d_opt:.2f} Γ", (d_opt, T_opt * 1e6),
                xytext=(0, -22), textcoords="offset points", ha="center", va="top", fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8))
    T_ideal = theory.ideal_doppler_limit() * 1e6
    _theory_line(ax, T_ideal, f"ideal 1D limit ħΓ/2k$_B$ = {T_ideal:.0f} µK", side="left")
    ax.set_ylim(0, None)
    ax.set_title("a  Temperature against detuning")
    ax.set_xlabel("detuning δ (Γ)")
    ax.set_ylabel("steady-state temperature (µK)")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(curve, [theory.cloud_size(q) * 1e6 for q in curve_params],
            color=INK, lw=1.2, label="theory")
    ax.plot(scan.detuning, scan.cloud_size * 1e6, "o", ms=7, color=BLUE, mec=SURFACE, mew=1.5,
            label="simulation")
    ax.set_ylim(0, None)
    ax.set_title("b  Cloud size against detuning")
    ax.set_xlabel("detuning δ (Γ)")
    ax.set_ylabel("RMS cloud size (µm)")
    ax.legend(loc="lower left")

    caption = (f"s0 = {base.saturation:g} per beam  ·  B' = {base.field_gradient:g} G/cm  ·  "
               "each point: 500 atoms, averaged over the steady-state half of the run")
    fig.suptitle(caption, x=0.01, ha="left", color=INK_SECONDARY, fontsize=9.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Animation ----------------------------------------------------------------

def make_animation(result: SimulationResult, path: Path, max_time=12e-3, n_shown=500, fps=10, stride=2):
    p = result.params
    frames = np.flatnonzero(result.times <= max_time)[::stride]
    shown = slice(0, min(n_shown, result.settings.n_atoms))
    T = result.temperature().mean(axis=1) * 1e6
    T_theory = theory.doppler_temperature(p) * 1e6
    t_ms = result.times * 1e3
    lim_mm = 1.3 * result.settings.initial_radius * 1e3

    fig, (ax_xy, ax_ps, ax_T) = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)

    ax_xy.set_xlim(-lim_mm, lim_mm)
    ax_xy.set_ylim(-lim_mm, lim_mm)
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x (mm)")
    ax_xy.set_ylabel("y (mm)")
    ax_xy.set_title("Atoms in the plane")
    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        start = -0.97 * lim_mm * np.array(d)
        ax_xy.annotate("", xy=start + 0.18 * lim_mm * np.array(d), xytext=start,
                       arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5))
    ax_xy.text(0.02, 0.02, "arrows: the four laser beams", transform=ax_xy.transAxes,
               fontsize=8.5, color=INK_SECONDARY)
    dots_xy = ax_xy.scatter([], [], s=4, color=BLUE, alpha=0.6, linewidths=0)

    # Phase space zooms with the cloud (±4σ), so the spiral stays visible as it shrinks.
    x_span = 4 * result.positions[:, :, 0].std(axis=1) * 1e3
    v_span = 4 * result.velocities[:, :, 0].std(axis=1)
    force_image = ax_ps.imshow(np.zeros((2, 2)), origin="lower", aspect="auto", cmap=FORCE_CMAP,
                               alpha=0.55, interpolation="bilinear")
    ax_ps.grid(False)
    ax_ps.set_xlabel("position x (mm)")
    ax_ps.set_ylabel("velocity $v_x$ (m/s)")
    ax_ps.set_title("Phase space, zooming with the cloud (colour = mean force)")
    dots_ps = ax_ps.scatter([], [], s=4, color=INK, alpha=0.6, linewidths=0)

    ax_T.set_xlim(0, t_ms[frames[-1]])
    ax_T.set_ylim(0.5 * T_theory, 1.5 * T.max())
    ax_T.set_yscale("log")
    ax_T.set_xlabel("time (ms)")
    ax_T.set_ylabel("temperature (µK)")
    ax_T.set_title("Temperature")
    _theory_line(ax_T, T_theory, f"Doppler theory {T_theory:.0f} µK")
    (line_T,) = ax_T.plot([], [], color=BLUE)
    clock = fig.suptitle("", x=0.01, ha="left", fontsize=11)

    def update(i):
        frame = frames[i]
        pos = result.positions[frame, shown] * 1e3
        dots_xy.set_offsets(pos)
        dots_ps.set_offsets(np.column_stack((pos[:, 0], result.velocities[frame, shown, 0])))
        xs, vs = x_span[frame], v_span[frame]
        _, _, F = _force_map(p, (-xs * 1e-3, xs * 1e-3), (-vs, vs), n=81)
        limit = np.abs(F).max()
        force_image.set_data(F)
        force_image.set_norm(TwoSlopeNorm(0, -limit, limit))
        force_image.set_extent((-xs, xs, -vs, vs))
        ax_ps.set_xlim(-xs, xs)
        ax_ps.set_ylim(-vs, vs)
        line_T.set_data(t_ms[: frame + 1], T[: frame + 1])
        clock.set_text(f"t = {t_ms[frame]:5.1f} ms     T = {T[frame]:7.0f} µK")
        return dots_xy, force_image, dots_ps, line_T, clock

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


def _parameter_caption(p: MOTParameters) -> str:
    return (f"Rb-87 D2 line  ·  δ = {p.detuning:g} Γ  ·  s0 = {p.saturation:g} per beam  ·  "
            f"w = {p.beam_waist * 1e3:g} mm  ·  B' = {p.field_gradient:g} G/cm")
