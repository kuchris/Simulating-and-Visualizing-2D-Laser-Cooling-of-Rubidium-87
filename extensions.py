"""Physics beyond the basic MOT: run the extension studies and make their figures.

    python extensions.py                   # every study (about 6 min)
    python extensions.py --only molasses   # one study: capture, repumper, density, source, molasses
    python extensions.py --quick           # fewer atoms and points (about 1.5 min)

Figures are written to figures/extensions/.
"""

import argparse
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from mot import capture, density, molasses, repumper, source
from mot.parallel import parallel_map
from mot.extension_figures import (core_temperature, plot_capture, plot_density, plot_molasses, plot_repumper,
                                   plot_source)
from mot.parameters import MOTParameters
from mot.simulation import doppler_steady_state

STUDIES = ("capture", "repumper", "density", "source", "molasses")
COARSE_SPEEDS = np.linspace(0.25, 80, 160)


# Each job below runs in its own worker process (see mot/parallel.py).

def _capture_velocity(detuning, p):
    return capture.deterministic_capture_velocity(replace(p, detuning=detuning), speeds=COARSE_SPEEDS)


def _capture_at(detuning, p, n_each, n_atoms):
    q = replace(p, detuning=detuning)
    speeds = np.linspace(0.4, 1.6, 13) * capture.deterministic_capture_velocity(q, speeds=COARSE_SPEEDS)
    half = capture.half_capture_speed(speeds, capture.capture_probability(q, speeds, n_each))
    return half, capture.loading_rate(q, n_atoms=n_atoms)


def _repump(job, p):
    saturation, n_atoms, duration = job
    return repumper.run(p, saturation, n_atoms=n_atoms, duration=duration)


def _density_steady(job, p):
    atom_number, settings = job
    return density.run(p, atom_number, settings).steady_state()


def _source(push, n_atoms):
    return source.run(replace(source.SourceParameters(), push_saturation=push), n_atoms=n_atoms)


def _molasses_core_temperature(job):
    detuning, saturation, x, v = job
    r = molasses.run([(molasses.ForceTable.build(detuning, saturation), 2e-3)], x, v, snapshot_interval=1e-3)
    return core_temperature(r.velocities[-1].ravel())


def _table_core_temperature(job):
    detuning, saturation = job
    return molasses.ForceTable.build(detuning, saturation, n=161).core_temperature()


def study_capture(p: MOTParameters, quick: bool, out: Path):
    detunings = np.array([-1.0, -2.0, -3.0, -4.0]) if quick else np.array([-1.0, -1.5, -2.0, -2.5, -3.0, -4.0])
    speeds = np.arange(10.0, 32.0, 2.0) if quick else np.arange(8.0, 34.0, 1.5)
    n_each = 60 if quick else 150
    probability = capture.capture_probability(p, speeds, n_each)
    v_c = capture.deterministic_capture_velocity(p)

    curve = np.linspace(-0.75, -4.5, 8 if quick else 16)
    curve_vc = parallel_map(partial(_capture_velocity, p=p), curve)
    rows = parallel_map(partial(_capture_at, p=p, n_each=n_each // 2, n_atoms=6000 if quick else 20000), detunings)
    half, loading = map(list, zip(*rows))
    reference = int(np.argmin(np.abs(detunings - p.detuning)))

    plot_capture(p, dict(head_on_speeds=speeds, head_on_probability=probability, head_on_capture_velocity=v_c,
                         detuning_curve=curve, capture_velocity_curve=curve_vc, detunings=detunings,
                         half_capture_speeds=half, loading=loading, reference_index=reference),
                 out / "capture.png")
    ref = loading[reference]
    print(f"  head-on capture velocity {v_c:.1f} m/s;  loading rate {ref.rate:.2e} ± {ref.rate_error:.1e} atoms/s"
          f" (mean force alone {ref.rate_mean_force:.2e});  steady N ≈ {ref.steady_atom_number():.1e}")
    return ref


def study_repumper(p: MOTParameters, quick: bool, out: Path):
    n = 300 if quick else 600
    saturations = [1e-4, 1e-3, 1e-2, 1e-1] if quick else [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1]
    duration = 4e-3 if quick else 8e-3
    jobs = [(0.0, n, 3e-3), (0.01, n, 3e-3)] + [(s, n, duration) for s in saturations]
    off, on, *scan = parallel_map(partial(_repump, p=p), jobs)
    plot_repumper(p, off, on, scan, out / "repumper.png")
    print(f"  leak probability {repumper.leak_probability(p):.2e} per photon; "
          f"dark after {1e3 / repumper.leak_rate(p):.2f} ms without a repumper")


def study_density(p: MOTParameters, quick: bool, out: Path, loading: capture.Loading | None):
    numbers = [1e6, 1e7, 1e8, 3e8] if quick else [1e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8]
    s = density.DensitySettings(n_sim=800 if quick else 1500, duration=10e-3 if quick else 14e-3)
    # The same clouds with the shadows removed and σ_R lowered to σ_R − σ_L:
    # the same total repulsion, but from a conservative force alone.
    conservative = replace(s, shadows=False, rescatter_ratio=density.RESCATTER_RATIO - 1)
    jobs = [(n, s) for n in numbers] + [(n, conservative) for n in numbers]
    results = parallel_map(partial(_density_steady, p=p), jobs)
    steady, no_shadow = results[:len(numbers)], results[len(numbers):]

    loading = loading or capture.loading_rate(p, n_atoms=6000)
    vapour = np.geomspace(1e12, 1e16, 60)
    rate = loading.rate * vapour / capture.VAPOUR_DENSITY
    background = vapour * capture.LOSS_CROSS_SECTION * 2 * capture.most_probable_speed() / np.sqrt(np.pi)
    plot_density(p, dict(atom_numbers=numbers, steady=steady, no_shadow=no_shadow, n_sim=s.n_sim,
                         vapour_densities=vapour, steady_no_collisions=rate / background,
                         steady_with_collisions=[density.steady_atom_number(p, r, b) for r, b in zip(rate, background)]),
                 out / "density.png")
    print(f"  density limit {density.density_limit(p) * 1e-6:.1e} cm⁻³; "
          f"central density at N = {numbers[-1]:.0e}: {density.central_density(steady[-1], numbers[-1]) * 1e-6:.1e} cm⁻³")


def study_source(quick: bool, out: Path):
    pushes = [0.0, 0.5, 1.5] if quick else [0.0, 0.1, 0.25, 0.5, 1.0, 1.5]
    scan = parallel_map(partial(_source, n_atoms=50000 if quick else 150000), pushes)
    reference = scan[pushes.index(0.5)]
    plot_source(dict(scan=scan, tracks=reference.tracks, tracks_params=reference.params), out / "source.png")
    print("  useful flux: " + ", ".join(f"push {r.params.push_saturation:g} → {r.flux():.1e}/s" for r in scan))


def study_molasses(p: MOTParameters, quick: bool, out: Path):
    stages = [(molasses.ForceTable.build(-6.0, 3.0), 1.5e-3), (molasses.ForceTable.build(-8.0, 2.0), 1.5e-3)]
    rng = np.random.default_rng(10)
    x, v = doppler_steady_state(p, 500 if quick else 1500, rng)
    result = molasses.run(stages, x, v)

    # Precooled atoms (end of the first stage) for the scan: at low intensity the
    # sub-Doppler peak is too narrow to catch atoms straight from the MOT.
    first = molasses.run(stages[:1], x[:400], v[:400], snapshot_interval=stages[0][1])
    detunings = (-3.0, -6.0, -10.0)
    sats = np.geomspace(0.1, 6, 6 if quick else 12)
    points = [0.5, 3.0] if quick else [0.3, 1.0, 3.0]
    curve = parallel_map(_table_core_temperature, [(d, s) for d in detunings for s in sats])
    dots = parallel_map(_molasses_core_temperature,
                        [(d, s, first.positions[-1], first.velocities[-1]) for d in detunings for s in points])
    theory_scan = [(d, sats, curve[i * len(sats):(i + 1) * len(sats)]) for i, d in enumerate(detunings)]
    mc_scan = [(d, points, dots[i * len(points):(i + 1) * len(points)]) for i, d in enumerate(detunings)]
    plot_molasses(p, dict(stages=stages, result=result, theory_scan=theory_scan, mc_scan=mc_scan),
                  out / "molasses.png")
    print(f"  MOT → molasses: core temperature {core_temperature(result.velocities[-1].ravel()) * 1e6:.1f} µK "
          f"(D(0)/β = {stages[-1][0].core_temperature() * 1e6:.1f} µK)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=STUDIES, action="append", help="run only this study (repeatable)")
    parser.add_argument("--quick", action="store_true", help="fewer atoms and scan points")
    parser.add_argument("--out", type=Path, default=Path("figures") / "extensions")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    chosen = args.only or STUDIES
    p = MOTParameters(dimensions=3)

    loading = None
    for name in STUDIES:
        if name not in chosen:
            continue
        start = time.time()
        print(f"{name} ...")
        if name == "capture":
            loading = study_capture(p, args.quick, args.out)
        elif name == "repumper":
            study_repumper(p, args.quick, args.out)
        elif name == "density":
            study_density(p, args.quick, args.out, loading)
        elif name == "source":
            study_source(args.quick, args.out)
        else:
            study_molasses(p, args.quick, args.out)
        print(f"  ({time.time() - start:.0f} s)")
    print(f"\nFigures written to {args.out}/")


if __name__ == "__main__":
    main()
