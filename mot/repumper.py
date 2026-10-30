"""Leakage out of the cycling transition and the repumper.

The cooling light is tuned near F=2 → F'=3, but it also excites F'=2 and F'=1
off resonance (267 MHz and 424 MHz further down). These levels can decay to
F=1, which is 6.8 GHz away and dark to the cooling light. After roughly a
thousand photons an atom leaves the cycle; without a repumper (F=1 → F'=2)
the MOT empties within a millisecond.

Per cycling photon, the chance to end up in F=1 is

    p_leak = Σ_{F'=2,1} (S_2F' / S_23) · L(δ_F') / L(δ_3) · b(F' → F=1)

with relative hyperfine strengths S_2F' for atoms spread over the m_F
sublevels, L(δ) = 1/(1 + s_tot + (2δ/Γ)²), and decay branching ratios b.
A resonant repumper of saturation s_rep excites F=1 → F'=2 at
(Γ/2) s_rep/(1 + s_rep); half of the excited atoms return to F=2, so

    γ_rep = (Γ/4) s_rep / (1 + s_rep),   bright fraction  f = γ_rep / (γ_rep + γ_leak).

When atoms switch often (γ_rep, γ_leak ≫ trap frequency), the friction, the
heating and the spring are all scaled by f: the temperature stays at the
Doppler value, but the weaker trap lets the cloud grow as 1/√f.
"""

from dataclasses import dataclass

import numpy as np

from . import constants as c
from . import theory
from .fast import advance
from .parameters import MOTParameters
from .physics import recoil_kicks, scattering_rates
from .simulation import doppler_steady_state

# Steck, "Rubidium 87 D Line Data": excited-state splittings and hyperfine strengths from F=2.
SPLITTING_3_2 = 2 * np.pi * 266.650e6   # F'=3 − F'=2 [rad/s]
SPLITTING_3_1 = 2 * np.pi * 423.597e6   # F'=3 − F'=1 [rad/s]
STRENGTH = {3: 7 / 10, 2: 1 / 4, 1: 1 / 20}   # S_2F'
TO_F1 = {2: 1 / 2, 1: 5 / 6}                  # branching ratio F' → F=1


def leak_probability(p: MOTParameters) -> float:
    """Probability that one scattered photon leaves the atom in F=1."""
    s_tot = 2 * p.dimensions * p.saturation

    def lorentzian(delta):
        return 1 / (1 + s_tot + (2 * delta / c.GAMMA) ** 2)

    cycling = lorentzian(p.delta)
    return sum(STRENGTH[f] / STRENGTH[3] * lorentzian(p.delta + split) / cycling * TO_F1[f]
               for f, split in ((2, SPLITTING_3_2), (1, SPLITTING_3_1)))


def leak_rate(p: MOTParameters) -> float:
    """Rate at which an atom at rest in the centre is pumped into F=1 [1/s]."""
    return leak_probability(p) * 2 * p.dimensions * theory.scattering_rate_per_beam(p)


def repump_rate(repump_saturation: float) -> float:
    """Rate at which a resonant repumper returns an atom from F=1 to F=2 [1/s]."""
    return 0.25 * c.GAMMA * repump_saturation / (1 + repump_saturation)


def bright_fraction(p: MOTParameters, repump_saturation: float) -> float:
    """Steady-state fraction of atoms in F=2, which scatter the cooling light."""
    up = repump_rate(repump_saturation)
    return up / (up + leak_rate(p)) if up > 0 else 0.0


@dataclass
class RepumpResult:
    params: MOTParameters
    repump_saturation: float
    times: np.ndarray        # (n_snapshots,) [s]
    positions: np.ndarray    # (n_snapshots, n_atoms, D) [m]
    velocities: np.ndarray   # (n_snapshots, n_atoms, D) [m/s]
    bright: np.ndarray       # (n_snapshots, n_atoms) atom in F=2
    fluorescence: np.ndarray  # (n_snapshots,) photons per second per atom

    def bright_fraction(self) -> np.ndarray:
        return self.bright.mean(axis=1)

    def temperature(self) -> np.ndarray:
        return c.MASS * self.velocities.var(axis=1) / c.K_B

    def rms_size(self) -> np.ndarray:
        return self.positions.std(axis=1)


def run(p: MOTParameters, repump_saturation: float, n_atoms: int = 600, duration: float = 6e-3,
        dt: float = 1e-6, snapshot_interval: float = 0.05e-3, seed: int = 6) -> RepumpResult:
    """A Doppler-cooled cloud, all in F=2 at t = 0, with a repumper of saturation s_rep.

    Atoms in F=2 scatter cooling light as in `simulation`; each photon sends
    them to F=1 with probability p_leak. Atoms in F=1 feel no cooling light and
    drift freely. Each repump photon kicks the atom (absorbed from one of the
    MOT beam directions, emitted isotropically) and returns it to F=2 with
    probability 1/2.
    """
    rng = np.random.default_rng(seed)
    x, v = doppler_steady_state(p, n_atoms, rng)
    bright = np.ones(n_atoms, dtype=bool)
    p_leak = leak_probability(p)
    excite = 0.5 * c.GAMMA * repump_saturation / (1 + repump_saturation) * dt
    directions = p.beam_directions

    steps = max(1, round(snapshot_interval / dt))
    n_snapshots = max(1, round(duration / (steps * dt)))
    record = {name: [] for name in ("x", "v", "bright", "fluorescence")}
    for i in range(n_snapshots + 1):
        rates = scattering_rates(x, v, p) * bright[:, None]
        record["x"].append(x.copy())
        record["v"].append(v.copy())
        record["bright"].append(bright.copy())
        record["fluorescence"].append(rates.sum() / n_atoms)
        if i == n_snapshots:
            break
        for _ in range(steps):
            n_cooling = advance(x, v, p, 1, dt, rng, brightness=bright.astype(float))
            leaks = rng.random(n_atoms) > (1 - p_leak) ** n_cooling

            n_repump = rng.poisson(excite * ~bright)
            if n_repump.any():
                beams = np.zeros((n_atoms, len(directions)), dtype=int)
                for atom in np.flatnonzero(n_repump):
                    beams[atom] = rng.multinomial(n_repump[atom], np.full(len(directions), 1 / len(directions)))
                v += recoil_kicks(beams, directions, rng)
            returns = rng.random(n_atoms) > 0.5 ** n_repump

            bright = (bright & ~leaks) | (~bright & returns)

    times = np.arange(n_snapshots + 1) * steps * dt
    return RepumpResult(p, repump_saturation, times, np.array(record["x"]), np.array(record["v"]),
                        np.array(record["bright"]), np.array(record["fluorescence"]))
