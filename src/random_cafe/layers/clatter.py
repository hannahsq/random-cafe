"""Clatter: cups, cutlery and glass as modal grains (a click exciting a few decaying resonances), Poisson in time,
uniform over the floor. The grain model follows umbrella-rain's impact ticks."""
import numpy as np

from .. import spatial
from ..scene import Scene

KINDS = {  # (mode frequency range Hz, decay range s, number of modes)
    "cup": ((1200.0, 4500.0), (0.02, 0.12), (2, 4)),
    "cutlery": ((2500.0, 7000.0), (0.005, 0.04), (3, 6)),
    "glass": ((1800.0, 6500.0), (0.08, 0.35), (1, 3)),
}


def grain(rng: np.random.Generator, fs: int) -> np.ndarray:
    (flo, fhi), (dlo, dhi), (mlo, mhi) = KINDS[str(rng.choice(list(KINDS)))]
    n = int(0.5 * fs)
    t = np.arange(n) / fs
    y = np.zeros(n)
    for _ in range(int(rng.integers(mlo, mhi + 1))):
        f = min(rng.uniform(flo, fhi), 0.45 * fs)
        y += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t / rng.uniform(dlo, dhi))
    click = rng.standard_normal(int(0.002 * fs)) * np.exp(-np.arange(int(0.002 * fs)) / (0.0004 * fs))
    y[:click.size] += 0.5 * click
    return y / (np.max(np.abs(y)) + 1e-12)


def render(rng: np.random.Generator, scene: Scene, room: spatial.SharedRoom) -> np.ndarray:
    cfg, fs = scene.clatter, scene.fs
    n = int(scene.duration_s * fs)
    out = np.zeros(n)
    for start in np.cumsum(rng.exponential(1 / cfg.rate_hz, int(cfg.rate_hz * scene.duration_s * 2) + 5)):
        if start >= scene.duration_s:
            break
        src = spatial.floor_position(rng, cfg.radius_m, 0.75, scene.mic[:2])  # table height
        direct = spatial.place_static(grain(rng, fs) * rng.uniform(0.3, 1.0), src, scene.mic, fs, n, start)
        out += room.apply(direct, spatial.distance(src, scene.mic)) if scene.room.indoor else direct
    return out
