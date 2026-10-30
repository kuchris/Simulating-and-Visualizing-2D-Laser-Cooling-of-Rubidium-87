"""Simulate a 2D magneto-optical trap for Rb-87 and make all figures.

    python main.py                     # everything, default parameters
    python main.py --detuning -1.5     # another detuning (in units of Γ)
    python main.py --no-scan --no-animation
"""

import argparse
from pathlib import Path

import numpy as np

from mot2d import constants as c
from mot2d import theory
from mot2d.parameters import MOTParameters
from mot2d.simulation import SimulationSettings, run, scan_detuning
from mot2d.visualization import (make_animation, plot_cooling_dynamics, plot_detuning_scan,
                                 plot_force_model)


def parse_args():
    defaults_p, defaults_s = MOTParameters(), SimulationSettings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detuning", type=float, default=defaults_p.detuning, help="laser detuning in units of Γ")
    parser.add_argument("--saturation", type=float, default=defaults_p.saturation, help="peak s0 = I/I_sat per beam")
    parser.add_argument("--waist", type=float, default=defaults_p.beam_waist * 1e3, help="beam waist in mm")
    parser.add_argument("--gradient", type=float, default=defaults_p.field_gradient, help="field gradient in G/cm")
    parser.add_argument("--atoms", type=int, default=defaults_s.n_atoms)
    parser.add_argument("--duration", type=float, default=defaults_s.duration * 1e3, help="in ms")
    parser.add_argument("--seed", type=int, default=defaults_s.seed)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--no-scan", action="store_true", help="skip the detuning scan")
    parser.add_argument("--no-animation", action="store_true", help="skip the GIF")
    return parser.parse_args()


def print_summary(p: MOTParameters, T_sim: float, size_sim: float):
    rows = [
        ("scattering rate per beam", f"{theory.scattering_rate_per_beam(p) / 1e3:.0f} kHz"),
        ("velocity damping time m/β", f"{theory.velocity_damping_time(p) * 1e3:.2f} ms"),
        ("trap frequency ω/2π", f"{theory.trap_frequency(p):.0f} Hz"),
        ("damping ratio ζ", f"{theory.damping_ratio(p):.2f}"),
        ("recoil velocity ħk/m", f"{c.RECOIL_VELOCITY * 1e3:.2f} mm/s"),
    ]
    print("\nAnalytic properties of the trap")
    for name, value in rows:
        print(f"  {name:28s} {value}")
    print("\n                          theory    simulation")
    print(f"  temperature (µK)     {theory.doppler_temperature(p) * 1e6:9.0f} {T_sim * 1e6:12.0f}")
    print(f"  cloud size (µm)      {theory.cloud_size(p) * 1e6:9.0f} {size_sim * 1e6:12.0f}")


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    p = MOTParameters(detuning=args.detuning, saturation=args.saturation,
                      beam_waist=args.waist * 1e-3, field_gradient=args.gradient)
    settings = SimulationSettings(n_atoms=args.atoms, duration=args.duration * 1e-3, seed=args.seed)

    plot_force_model(p, args.out / "force_model.png")

    print(f"Simulating {settings.n_atoms} atoms for {settings.duration * 1e3:g} ms ...")
    result = run(p, settings)
    steady = result.steady_state()
    print_summary(p, steady.temperature().mean(), steady.rms_size().mean())
    plot_cooling_dynamics(result, args.out / "cooling_dynamics.png")

    if not args.no_scan:
        detunings = np.linspace(-0.5, -3.0, 11)
        print("\nScanning the detuning ...")
        scan = scan_detuning(p, detunings)
        plot_detuning_scan(scan, p, args.out / "detuning_scan.png")

    if not args.no_animation:
        print("Rendering animation ...")
        make_animation(result, args.out / "mot_animation.gif")

    print(f"\nFigures written to {args.out}/")


if __name__ == "__main__":
    main()
