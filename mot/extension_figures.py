"""Figures for the extensions: capture, repumper, density limit, 2D-MOT source, molasses."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import capture, density, molasses, repumper, theory
from . import constants as c
from .parameters import MOTParameters
from .physics import mean_force
from .visualization import (AQUA, AXIS_COLORS, BLUE, DENSITY_CMAP, F_MAX, INK, INK_MUTED, INK_SECONDARY,
                            ORANGE, RED, SURFACE, _draw_phase_space, _parameter_caption, _theory_line)

POINT = dict(fmt="o", ms=7, mec=SURFACE, mew=1.5, elinewidth=1.2)


def _caption(fig, text):
    fig.suptitle(text, x=0.01, ha="left", color=INK_SECONDARY, fontsize=9.5)


def _save(fig, path: Path):
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Capture and loading --------------------------------------------------------------

def _head_on_track(p: MOTParameters, speed: float, s: capture.CaptureSettings, duration=6e-3, dt=2e-6):
    """Mean-force trajectory (x, v_x) of an atom launched at the centre along +x."""
    x = np.array([[-s.start_radius, 0.0, 0.0]])
    v = np.array([[speed, 0.0, 0.0]])
    xs, vs = [], []
    for _ in range(round(duration / dt)):
        v = v + mean_force(x, v, p) / c.MASS * dt
        x = x + v * dt
        xs.append(x[0, 0])
        vs.append(v[0, 0])
        if abs(x[0, 0]) > s.escape_radius:
            break
    return np.array(xs), np.array(vs)


def plot_capture(p: MOTParameters, data: dict, path: Path):
    s = capture.CaptureSettings()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), constrained_layout=True)

    ax = axes[0, 0]
    _draw_phase_space(ax, p, x_mm=s.start_radius * 1e3, v_ms=30)
    v_c = data["head_on_capture_velocity"]
    for speed in (8, 14, v_c - 0.5, v_c + 1.5, 26):
        xs, vs = _head_on_track(p, speed, s)
        captured = speed < v_c
        ax.plot(xs * 1e3, vs, color=INK if captured else RED, lw=1.6)
    ax.set_title("a  Atoms launched at the centre: captured (black) or not (red)")

    ax = axes[0, 1]
    speeds, prob = data["head_on_speeds"], data["head_on_probability"]
    ax.plot(speeds, prob, "o-", color=BLUE, ms=6, mec=SURFACE, mew=1.2, lw=1.2,
            label="head-on along x, with photon noise")
    loading = data["loading"][data["reference_index"]]
    bins = np.linspace(0, loading.v_max, 16)
    which = np.digitize(loading.speeds, bins) - 1
    centres = 0.5 * (bins[1:] + bins[:-1])
    vapour = np.array([loading.trapped[which == i].mean() if np.any(which == i) else np.nan
                       for i in range(len(centres))])
    ax.plot(centres, vapour, "s-", color=ORANGE, ms=6, mec=SURFACE, mew=1.2, lw=1.2,
            label="from the vapour (all directions)")
    ax.axvline(v_c, color=INK, lw=1)
    ax.annotate(f"mean force alone:\n$v_c$ = {v_c:.1f} m/s", (v_c, 0.15), xytext=(6, 0),
                textcoords="offset points", fontsize=9, va="center")
    ax.set_title("b  Capture probability against speed")
    ax.set_xlabel("speed on entering the beams (m/s)")
    ax.set_ylabel("fraction captured")
    ax.legend(loc="center right", fontsize=9)

    ax = axes[1, 0]
    ax.plot(data["detuning_curve"], data["capture_velocity_curve"], color=INK, lw=1.2,
            label="mean force alone")
    ax.plot(data["detunings"], data["half_capture_speeds"], "o", color=BLUE, ms=7, mec=SURFACE, mew=1.5,
            label="simulation: 50 % captured")
    ax.axvline(p.detuning, color=INK_MUTED, lw=0.8, ls=":")
    ax.set_title("c  Capture velocity (head-on) against detuning")
    ax.set_xlabel("detuning δ (Γ)")
    ax.set_ylabel("capture velocity (m/s)")
    ax.legend(loc="lower right")

    ax = axes[1, 1]
    rates = np.array([l.rate for l in data["loading"]])
    errors = np.array([l.rate_error for l in data["loading"]])
    ax.errorbar(data["detunings"], rates, yerr=errors, color=BLUE, label="simulation", **POINT)
    ax.plot(data["detunings"], [l.rate_mean_force for l in data["loading"]], "D", ms=5, color=INK,
            mfc="none", label="same atoms, mean force alone")
    ax.plot(data["detunings"], [l.textbook_rate() for l in data["loading"]], "--", color=INK_MUTED, lw=1.2,
            label=r"textbook: $\frac{1}{2} n V^{2/3} v_c^4 (m/2k_BT)^{3/2}$, $V=(2w)^3$")
    ax.set_yscale("log")
    ax.set_title("d  Loading rate: off-axis atoms leave the beams at large |δ|")
    ax.set_xlabel("detuning δ (Γ)")
    ax.set_ylabel("loading rate $R_L$ (atoms/s)")
    ax.legend(loc="lower right", fontsize=9)
    tau = loading.steady_atom_number() / loading.rate
    right = ax.secondary_yaxis("right", functions=(lambda r: r * tau, lambda n: n / tau))
    right.set_ylabel(f"steady atom number $R_L\\tau$, τ = {tau:.2f} s")

    _caption(fig, _parameter_caption(p) + f"  ·  vapour: {capture.VAPOUR_DENSITY * 1e-6:.0e} cm⁻³ Rb-87 at "
             f"{capture.ROOM_TEMPERATURE:.0f} K  ·  atoms enter at r = {s.start_radius * 1e3:g} mm, lost beyond {s.escape_radius * 1e3:g} mm")
    _save(fig, path)


# --- Repumper -------------------------------------------------------------------------

def plot_repumper(p: MOTParameters, off: repumper.RepumpResult, on: repumper.RepumpResult,
                  scan: list, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    ax = axes[0]
    reference = on.fluorescence[0]
    t_ms = off.times * 1e3
    ax.plot(on.times * 1e3, on.fluorescence / reference, color=AQUA, lw=1.6,
            label=f"repumper on (s = {on.repump_saturation:g})")
    ax.plot(t_ms, off.fluorescence / reference, color=RED, lw=1.6, label="repumper off")
    ax.plot(t_ms, np.exp(-repumper.leak_rate(p) * off.times), color=INK, lw=1, ls="--",
            label=rf"theory $e^{{-\gamma_{{leak}} t}}$, 1/γ = {1e3 / repumper.leak_rate(p):.2f} ms")
    ax.set_yscale("log")
    ax.set_ylim(3e-3, 2)
    ax.set_title("a  Without a repumper the MOT goes dark")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("fluorescence (relative)")
    ax.legend(loc="lower left", fontsize=9)

    saturations = np.array([r.repump_saturation for r in scan])
    grid = np.geomspace(saturations.min() / 2, saturations.max() * 2, 200)
    ax = axes[1]
    ax.plot(grid, [repumper.bright_fraction(p, s) for s in grid], color=INK, lw=1.2, label="theory")
    ax.plot(saturations, [r.bright_fraction()[len(r.times) // 2:].mean() for r in scan], "o", color=BLUE,
            ms=7, mec=SURFACE, mew=1.5, label="simulation")
    ax.set_xscale("log")
    ax.set_title("b  Fraction of atoms in the cycling state F=2")
    ax.set_xlabel("repumper saturation $s_{rep}$")
    ax.set_ylabel("bright fraction")
    ax.legend(loc="lower right")
    ax.annotate(f"$p_{{leak}}$ = {repumper.leak_probability(p):.1e} per photon", (0.03, 0.92),
                xycoords="axes fraction", fontsize=9)

    ax = axes[2]
    sizes = theory.cloud_sizes(p)
    ax.plot(grid, [1 / np.sqrt(repumper.bright_fraction(p, s)) for s in grid], color=INK, lw=1.2, ls="--",
            label=r"size theory $1/\sqrt{f}$")
    for axis, name in ((0, "radial"), (2, "axial")):
        ax.plot(saturations, [r.rms_size()[len(r.times) // 2:, axis].mean() / sizes[axis] for r in scan],
                "o-", color=AXIS_COLORS[axis], ms=7, mec=SURFACE, mew=1.5, lw=1, label=f"size, {name}")
    ax.plot(saturations, [r.temperature()[len(r.times) // 2:].mean() / theory.doppler_temperature(p)
                          for r in scan], "s-", color=INK_SECONDARY, ms=6, mec=SURFACE, mew=1.2, lw=1,
            label="temperature")
    ax.axhline(1, color=INK, lw=1)
    omega = 2 * np.pi * theory.trap_frequencies(p)[0]
    s_slow = omega / (0.25 * c.GAMMA - omega)
    ax.axvline(s_slow, color=INK_MUTED, lw=0.8, ls=":")
    ax.annotate("repumping slower\nthan the trap", (s_slow, 0.98), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points", ha="left", va="top", fontsize=9, color=INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_title("c  Dark atoms feel no trap: the cloud grows, T does not")
    ax.set_xlabel("repumper saturation $s_{rep}$")
    ax.set_ylabel("relative to Doppler theory")
    ax.legend(loc="upper right", fontsize=9)

    _caption(fig, _parameter_caption(p) + "  ·  leaks via F'=2 and F'=1 to F=1, repumper on F=1 → F'=2")
    _save(fig, path)


# --- Density limit --------------------------------------------------------------------

def plot_density(p: MOTParameters, data: dict, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), constrained_layout=True)
    numbers = np.array(data["atom_numbers"])
    steady = data["steady"]
    curve = np.geomspace(numbers.min() / 2, numbers.max() * 2, 200)

    ax = axes[0, 0]
    limited = density.density_limited_sizes(p, curve)
    for axis, name in ((0, "radial"), (2, "axial")):
        color = AXIS_COLORS[axis]
        ax.plot(numbers, [s.rms_size().mean(axis=0)[axis] * 1e6 for s in steady], "o", color=color, ms=7,
                mec=SURFACE, mew=1.5, label=f"simulation, {name}")
        ax.plot(curve, np.full_like(curve, theory.cloud_sizes(p)[axis] * 1e6), color=color, lw=1, alpha=0.7)
        ax.plot(curve, limited[:, axis] * 1e6, color=color, lw=1.2, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("a  Large clouds grow as $N^{1/3}$")
    ax.set_xlabel("atom number N")
    ax.set_ylabel("RMS cloud size (µm)")
    ax.legend(loc="upper left", fontsize=9,
              title="solid: Doppler size,  dashed: uniform cloud at $n_{max}$", title_fontsize=8.5)

    ax = axes[0, 1]
    n_max = density.density_limit(p)
    ax.plot(numbers, [density.central_density(s, n) * 1e-6 for s, n in zip(steady, numbers)], "o",
            color=BLUE, ms=7, mec=SURFACE, mew=1.5, label="simulation (centre)")
    ax.plot(curve, density.temperature_limited_density(p, curve) * 1e-6, color=INK, lw=1,
            label="no density effects: N / (2π)$^{3/2}$σ$_x$σ$_y$σ$_z$")
    _theory_line(ax, n_max * 1e-6, f"$n_{{max}}$ = {n_max * 1e-6:.1e} cm⁻³", side="left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(None, n_max * 1e-6 * 5)
    ax.set_title("b  The density saturates")
    ax.set_xlabel("atom number N")
    ax.set_ylabel("peak density (cm⁻³)")
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1, 0]
    T = np.array([s.temperature().mean(axis=0) for s in steady]) * 1e6
    ax.plot(numbers, T[:, 0], "o-", color=BLUE, ms=7, mec=SURFACE, mew=1.5, lw=1, label="radial")
    ax.plot(numbers, T[:, 2], "o-", color=AQUA, ms=7, mec=SURFACE, mew=1.5, lw=1, label="axial")
    ax.plot(numbers, [s.temperature().mean() * 1e6 for s in data["no_shadow"]], "s", color=INK, ms=6,
            mfc="none", label="rescattering only (conservative)")
    _theory_line(ax, theory.doppler_temperature(p) * 1e6, "Doppler theory", side="left")
    ax.set_xscale("log")
    ax.set_title("c  Shadows drive flows that warm large clouds")
    ax.set_xlabel("atom number N")
    ax.set_ylabel("kinetic temperature (µK)")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1, 1]
    densities = data["vapour_densities"]
    ax.plot(densities * 1e-6, data["steady_no_collisions"], color=INK_MUTED, lw=1.2,
            label="background collisions only")
    ax.plot(densities * 1e-6, data["steady_with_collisions"], color=RED, lw=1.8,
            label=f"+ light-assisted collisions, β = {density.TWO_BODY_LOSS * 1e6:.0e} cm³/s")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("d  Steady atom number against vapour density")
    ax.set_xlabel("Rb-87 vapour density (cm⁻³)")
    ax.set_ylabel("steady atom number")
    ax.legend(loc="lower right", fontsize=9)

    _caption(fig, _parameter_caption(p) + f"  ·  σ_R/σ_L = {density.RESCATTER_RATIO}  ·  "
             f"{data['n_sim']} simulated atoms each")
    _save(fig, path)


# --- 2D-MOT source --------------------------------------------------------------------

def plot_source(data: dict, path: Path):
    from . import source
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1, 1.35]))
    results = data["scan"]
    push = np.array([r.params.push_saturation for r in results])

    ax = axes[0]
    ax.errorbar(push, [r.flux() for r in results], yerr=[r.flux_error() for r in results], color=BLUE,
                label="simulation", **POINT)
    ax.set_ylim(0, None)
    ax.set_title("a  Useful flux through the hole")
    ax.set_xlabel("push-beam saturation")
    ax.set_ylabel(f"atoms/s slower than {source.CAPTURABLE_SPEED:g} m/s")

    ax = axes[1]
    bins = np.linspace(0, 60, 41)
    colors = [INK_MUTED, BLUE, ORANGE, AQUA, RED]
    shown = [0, len(results) // 2, len(results) - 1]
    for color, i in zip(colors, shown):
        r = results[i]
        ax.hist(r.out_velocity[:, 2], bins=bins, histtype="step", lw=1.8, color=color,
                weights=np.full(len(r.out_velocity), r.weight / r.n_sampled / (bins[1] - bins[0])),
                label=f"push s = {r.params.push_saturation:g}")
        if r.params.push_saturation > 0:
            ax.axvline(source.pushed_velocity(r.params), color=color, lw=1, ls="--")
    ax.axvspan(source.CAPTURABLE_SPEED, bins[-1], color=INK_MUTED, alpha=0.08, lw=0)
    ax.set_title("b  Speed of the atoms through the hole")
    ax.set_xlabel("longitudinal velocity $v_z$ (m/s)")
    ax.set_ylabel("flux per speed (atoms s⁻¹ / (m/s))")
    ax.legend(loc="upper right", fontsize=9, title="dashed: atom pushed from rest", title_fontsize=8.5)

    ax = axes[2]
    sp = data["tracks_params"]
    z = np.linspace(-sp.box_half_length, sp.hole_distance, 400)
    xg = np.linspace(-sp.box_half_width, sp.box_half_width, 200)
    Z, X = np.meshgrid(z, xg)
    pts = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], axis=1)
    s_map = sp.saturations(pts).sum(axis=1).reshape(X.shape) * (np.abs(Z) <= sp.box_half_length)
    ax.pcolormesh(Z * 1e3, X * 1e3, s_map, cmap=DENSITY_CMAP, shading="auto", rasterized=True)
    for track, velocity in data["tracks"]:
        ax.plot(track[:, 2] * 1e3, track[:, 0] * 1e3, color=INK, lw=0.8)
        end = track[-1]
        x_hole = end[0] + velocity[0] * (sp.hole_distance - end[2]) / velocity[2]
        ax.plot([end[2] * 1e3, sp.hole_distance * 1e3], [end[0] * 1e3, x_hole * 1e3], color=INK, lw=0.6, ls=":")
    ax.plot([sp.hole_distance * 1e3] * 2, [-sp.box_half_width * 1e3, -sp.hole_radius * 1e3], color=INK, lw=3)
    ax.plot([sp.hole_distance * 1e3] * 2, [sp.hole_radius * 1e3, sp.box_half_width * 1e3], color=INK, lw=3)
    ax.annotate("hole", (sp.hole_distance * 1e3, sp.hole_radius * 1e3), xytext=(-4, 10),
                textcoords="offset points", ha="right", fontsize=9)
    ax.set_xlim(-sp.box_half_length * 1e3, sp.hole_distance * 1e3 + 1)
    ax.set_ylim(-sp.box_half_width * 1e3, sp.box_half_width * 1e3)
    ax.grid(False)
    ax.set_title(f"c  Paths of atoms that reach the hole (push s = {sp.push_saturation:g})")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")

    sp0 = results[0].params
    _caption(fig, f"2D MOT  ·  δ = {sp0.detuning:g} Γ  ·  s0 = {sp0.saturation:g} per beam  ·  "
             f"w = {sp0.beam_waist * 1e3:g} mm × {sp0.beam_length * 1e3:g} mm  ·  B' = {sp0.field_gradient:g} G/cm"
             f"  ·  push: δ = {sp0.push_detuning:g} Γ, w = {sp0.push_waist * 1e3:g} mm  ·  hole ⌀ "
             f"{2 * sp0.hole_radius * 1e3:g} mm at z = {sp0.hole_distance * 1e3:g} mm  ·  "
             f"vapour {source.SOURCE_VAPOUR_DENSITY * 1e-6:.0e} cm⁻³")
    _save(fig, path)


# --- Molasses -------------------------------------------------------------------------

def core_temperature(v: np.ndarray) -> float:
    """Temperature of the cold core of a velocity sample, from its median absolute deviation [K]."""
    sigma = 1.4826 * np.median(np.abs(v - np.median(v)))
    return c.MASS * sigma**2 / c.K_B


def plot_molasses(p: MOTParameters, data: dict, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), constrained_layout=True)
    stages = data["stages"]
    table = stages[0][0]
    unit = c.HBAR * c.WAVENUMBER * c.GAMMA

    ax = axes[0, 0]
    v = np.linspace(-3, 3, 1201)
    force, _ = table(v)
    ax.plot(v, force / unit, color=BLUE, label="F=2 → F'=3, σ+σ− (Bloch equations)")
    ax.plot(v, molasses.doppler_force(v, table.detuning, table.saturation) / unit, color=INK_MUTED, lw=1.4,
            label="two-level atom (Doppler)")
    ax.set_title(f"a  Force in a molasses (δ = {table.detuning:g} Γ, s0 = {table.saturation:g})")
    ax.set_xlabel("velocity (m/s)")
    ax.set_ylabel(r"force / ($\hbar k\Gamma$)")
    ax.legend(loc="lower left", fontsize=9)
    inset = ax.inset_axes([0.58, 0.55, 0.38, 0.4])
    vz = np.linspace(-0.4, 0.4, 801)
    inset.plot(vz, table(vz)[0] / unit, color=BLUE)
    inset.plot(vz, molasses.doppler_force(vz, table.detuning, table.saturation) / unit, color=INK_MUTED, lw=1.2)
    inset.set_title("zoom: sub-Doppler peak", fontsize=8.5)
    inset.tick_params(labelsize=7.5)

    ax = axes[0, 1]
    result = data["result"]
    t_ms = result.times * 1e3
    ax.plot(t_ms, result.temperature().mean(axis=1) * 1e6, color=INK_SECONDARY, lw=1.6,
            label="all atoms, m⟨v²⟩/k$_B$")
    ax.plot(t_ms, [core_temperature(vv.ravel()) * 1e6 for vv in result.velocities], color=BLUE, lw=1.6,
            label="cold core")
    _theory_line(ax, theory.doppler_temperature(p) * 1e6, "MOT (Doppler theory)")
    t_switch = 0.0
    for table_k, duration in stages:
        ax.axvline(t_switch * 1e3, color=INK_MUTED, lw=0.8, ls=":")
        ax.annotate(f"δ = {table_k.detuning:g} Γ\ns0 = {table_k.saturation:g}", (t_switch * 1e3, 0.03),
                    xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points",
                    va="bottom", fontsize=8.5, color=INK_SECONDARY)
        ax.plot([t_switch * 1e3, (t_switch + duration) * 1e3], [table_k.core_temperature() * 1e6] * 2,
                color=INK, lw=1, ls="--")
        t_switch += duration
    ax.set_yscale("log")
    ax.set_ylim(0.5, None)
    ax.set_title("b  Switching from MOT to molasses (dashed: D(0)/β)")
    ax.set_xlabel("time after the field is switched off (ms)")
    ax.set_ylabel("temperature (µK)")
    ax.legend(loc="center right", fontsize=9)

    ax = axes[1, 0]
    for color, (detuning, sats, temps) in zip((BLUE, ORANGE, AQUA), data["theory_scan"]):
        ax.plot(sats / abs(detuning), np.array(temps) * 1e6, color=color, lw=1.4, label=f"δ = {detuning:g} Γ")
    for color, (detuning, sats, temps) in zip((BLUE, ORANGE, AQUA), data["mc_scan"]):
        ax.plot(np.array(sats) / abs(detuning), np.array(temps) * 1e6, "o", color=color, ms=7, mec=SURFACE,
                mew=1.5)
    ax.axhspan(1e-3, 10 * molasses.RECOIL_TEMPERATURE * 1e6, color=INK_MUTED, alpha=0.1, lw=0)
    ax.annotate("below 10 recoil temperatures:\nsemiclassical model unreliable", (0.98, 0.03),
                xycoords="axes fraction", ha="right", va="bottom", fontsize=8.5, color=INK_SECONDARY)
    _theory_line(ax, molasses.RECOIL_TEMPERATURE * 1e6, "recoil temperature (ħk)²/mk$_B$", side="right")
    _theory_line(ax, theory.ideal_doppler_limit() * 1e6, "Doppler limit ħΓ/2k$_B$", side="left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"c  Sub-Doppler temperature $\propto$ intensity / detuning")
    ax.set_xlabel("$s_0$ / |δ/Γ|")
    ax.set_ylabel("temperature of the cold core (µK)")
    ax.set_ylim(0.1 * molasses.RECOIL_TEMPERATURE * 1e6, None)
    ax.legend(loc="center left", fontsize=9, title="lines: D(0)/β, dots: simulation", title_fontsize=8.5)

    ax = axes[1, 1]
    final = result.velocities[-1].ravel()
    bins = np.linspace(-0.3, 0.3, 121)
    counts, _, _ = ax.hist(final, bins=bins, density=True, color=BLUE, alpha=0.85, label="after molasses")
    grid = np.linspace(-0.3, 0.3, 600)
    for T, color, label in ((theory.doppler_temperature(p), ORANGE, "MOT, Doppler"),
                            (stages[-1][0].core_temperature(), INK, "D(0)/β of the last stage")):
        sigma = np.sqrt(c.K_B * T / c.MASS)
        ax.plot(grid, np.exp(-grid**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi) / sigma, color=color, lw=1.2,
                label=f"{label}: {T * 1e6:.1f} µK")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 3 * counts.max())
    ax.set_title("d  A cold core and Doppler wings")
    ax.set_xlabel("velocity (m/s)")
    ax.set_ylabel("probability density (s/m)")
    ax.legend(loc="upper right", fontsize=9)

    _caption(fig, "1D σ+σ− molasses on each axis, from the steady state of the Bloch equations (12 Zeeman "
             f"levels)  ·  {result.positions.shape[1]} atoms from the Doppler-cooled 3D MOT  ·  "
             "dipole-force fluctuations left out")
    _save(fig, path)
