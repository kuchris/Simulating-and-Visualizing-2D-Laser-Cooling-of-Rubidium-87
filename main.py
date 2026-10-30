"""Simulate a magneto-optical trap for Rb-87 (2D or 3D) and make all figures.

    python main.py                       # 3D MOT, every figure (about 10 min)
    python main.py --dimensions 2        # 2D MOT
    python main.py --detuning -1.5       # another detuning (in units of Γ)
    python main.py --quick               # main simulation only: no scans, no animation
"""

import argparse
from pathlib import Path

import numpy as np

from mot import constants as c
from mot import studies, theory
from mot.measurement import absorption_image, time_of_flight
from mot.parameters import MOTParameters
from mot.simulation import SimulationSettings, run
from mot.visualization import (make_animation, plot_cooling_dynamics, plot_detuning_scan,
                               plot_force_model, plot_imperfections, plot_time_of_flight)


def parse_args():
    defaults_p, defaults_s = MOTParameters(), SimulationSettings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dimensions", type=int, choices=(2, 3), default=defaults_p.dimensions)
    parser.add_argument("--detuning", type=float, default=defaults_p.detuning, help="laser detuning in units of Γ")
    parser.add_argument("--saturation", type=float, default=defaults_p.saturation, help="peak s0 = I/I_sat per beam")
    parser.add_argument("--waist", type=float, default=defaults_p.beam_waist * 1e3, help="beam waist in mm")
    parser.add_argument("--gradient", type=float, default=defaults_p.field_gradient,
                        help="field gradient in G/cm (axial gradient in 3D)")
    parser.add_argument("--no-gravity", action="store_true")
    parser.add_argument("--atoms", type=int, default=defaults_s.n_atoms)
    parser.add_argument("--duration", type=float, default=defaults_s.duration * 1e3, help="in ms")
    parser.add_argument("--seed", type=int, default=defaults_s.seed)
    parser.add_argument("--out", type=Path, help="output folder (default: figures/2d or figures/3d)")
    parser.add_argument("--quick", action="store_true", help="skip the scans, TOF and animation")
    parser.add_argument("--no-scan", action="store_true", help="skip the detuning scan")
    parser.add_argument("--no-imperfections", action="store_true", help="skip the imperfection study (3D)")
    parser.add_argument("--no-animation", action="store_true", help="skip the GIF")
    return parser.parse_args()


def print_summary(p: MOTParameters, steady):
    def per_axis(values, fmt):
        return "  ".join(f"{name}: {fmt.format(value)}" for name, value in zip("xyz", values))

    print("\nAnalytic properties of the trap")
    print(f"  scattering rate per beam    {theory.scattering_rate_per_beam(p) / 1e3:.0f} kHz")
    print(f"  velocity damping time m/β   {theory.velocity_damping_time(p) * 1e3:.2f} ms")
    print(f"  trap frequency ω/2π         {per_axis(theory.trap_frequencies(p), '{:.0f} Hz')}")
    print(f"  damping ratio ζ             {per_axis(theory.damping_ratios(p), '{:.2f}')}")
    print(f"  recoil velocity ħk/m        {c.RECOIL_VELOCITY * 1e3:.2f} mm/s")

    print("\n                       theory                     simulation")
    rows = [("temperature (µK)", [theory.doppler_temperature(p)] * p.dimensions,
             steady.temperature().mean(axis=0), 1e6),
            ("cloud size (µm)", theory.cloud_sizes(p), steady.rms_size().mean(axis=0), 1e6),
            ("centre (µm)", theory.equilibrium_position(p), steady.center().mean(axis=0), 1e6)]
    for name, predicted, simulated, scale in rows:
        print(f"  {name:18s} {' '.join(f'{x * scale:7.1f}' for x in predicted):26s} "
              f"{' '.join(f'{x * scale:7.1f}' for x in simulated)}")


def measure_time_of_flight(p: MOTParameters, steady, out: Path):
    times = np.array([0, 1, 2, 3, 4]) * 1e-3
    tof = time_of_flight(steady, times)
    rng = np.random.default_rng(4)
    h, v = (0, 2) if p.dimensions == 3 else (0, 1)
    x0 = steady.positions.reshape(-1, p.dimensions)
    v0 = steady.velocities.reshape(-1, p.dimensions)
    center = theory.equilibrium_position(p)[[h, v]]
    images = [absorption_image(x0 + v0 * t + 0.5 * p.gravity_vector * t**2, n_atoms=1e6, axes=(h, v),
                               center=(center[0], center[1] - 0.6e-3), half_width=2.6e-3,
                               probe_counts=5000, rng=rng) for t in times]
    plot_time_of_flight(tof, images, p, out / "time_of_flight.png")
    print("\nTime of flight: fitted T = "
          + ", ".join(f"{name} {T * 1e6:.0f} µK" for name, T in zip("xyz", tof.temperature)))


def study_imperfections(p: MOTParameters, out: Path):
    print("\nStudying imperfections ...")
    imbalance = studies.scan_power_imbalance(p, np.linspace(0, 0.1, 5))
    stray = studies.scan_stray_field(p, np.linspace(0, 0.2, 5))
    gradient = studies.scan_gradient(p, [3.0, 5.0, 8.0, 12.0, 20.0])
    plot_imperfections(imbalance, stray, gradient, studies.anti_trap(p, duration=4e-3), p,
                       out / "imperfections.png")


def main():
    args = parse_args()
    out = args.out or Path("figures") / f"{args.dimensions}d"
    out.mkdir(parents=True, exist_ok=True)
    p = MOTParameters(dimensions=args.dimensions, detuning=args.detuning, saturation=args.saturation,
                      beam_waist=args.waist * 1e-3, field_gradient=args.gradient, gravity=not args.no_gravity)
    settings = SimulationSettings(n_atoms=args.atoms, duration=args.duration * 1e-3, seed=args.seed)

    plot_force_model(p, out / "force_model.png")

    print(f"Simulating {settings.n_atoms} atoms in a {p.dimensions}D MOT for {settings.duration * 1e3:g} ms ...")
    result = run(p, settings)
    steady = result.steady_state()
    print_summary(p, steady)
    plot_cooling_dynamics(result, out / "cooling_dynamics.png")

    if args.quick:
        print(f"\nFigures written to {out}/")
        return

    if p.dimensions == 3:
        measure_time_of_flight(p, steady, out)
        if not args.no_imperfections:
            study_imperfections(p, out)

    if not args.no_scan:
        print("\nScanning the detuning ...")
        scan = studies.scan_detuning(p, np.linspace(-0.5, -3.0, 11))
        plot_detuning_scan(scan, p, out / "detuning_scan.png")

    if not args.no_animation:
        print("Rendering animation ...")
        make_animation(result, out / "mot_animation.gif")

    print(f"\nFigures written to {out}/")


if __name__ == "__main__":
    main()
