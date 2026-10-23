"""Clatter: plates, cups, cutlery and glass as impact grains (a click and a noise burst exciting many short, inharmonic
resonances), Poisson in time, uniform over the floor. The grain model follows umbrella-rain's impact ticks."""
import numpy as np
from scipy import signal

from .. import spatial
from ..scene import Scene

KINDS = {  # (mode frequency range Hz, decay range s, number of modes, weight of the noise body, how often)
    # real crockery and cutlery are far less tonal than a few long-ringing modes (which sound like chimes): short
    # decays, many inharmonic partials, and a broadband impact that dominates the attack (heuristic values)
    "plate": ((600.0, 4000.0), (0.004, 0.025), (5, 10), 1.0, 0.35),
    "cup": ((1000.0, 5000.0), (0.004, 0.03), (4, 9), 0.8, 0.3),
    "cutlery": ((2000.0, 7500.0), (0.002, 0.015), (6, 12), 1.2, 0.3),
    "glass": ((1800.0, 6500.0), (0.02, 0.08), (3, 6), 0.4, 0.05),  # the one kind that rings a little
}


def grain(rng: np.random.Generator, fs: int) -> np.ndarray:
    names = list(KINDS)
    w = np.array([KINDS[k][4] for k in names])
    (flo, fhi), (dlo, dhi), (mlo, mhi), body, _ = KINDS[names[int(rng.choice(len(names), p=w / w.sum()))]]
    n = int(0.3 * fs)
    t = np.arange(n) / fs
    y = np.zeros(n)
    for _ in range(int(rng.integers(mlo, mhi + 1))):  # inharmonic partials, each with its own short decay
        f = min(np.exp(rng.uniform(np.log(flo), np.log(fhi))), 0.45 * fs)
        y += rng.uniform(0.2, 1.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t / rng.uniform(dlo, dhi))
    y /= np.max(np.abs(y)) + 1e-12
    # the impact: a band-limited noise burst decaying with the object (blurs any pitch) plus a sharp click
    tau = rng.uniform(dlo, dhi)
    burst = signal.sosfilt(signal.butter(2, [flo * 0.5, min(fhi * 1.5, 0.45 * fs)], "bandpass", fs=fs, output="sos"),
                           rng.standard_normal(n)) * np.exp(-t / tau)
    y += body * burst / (np.max(np.abs(burst)) + 1e-12)
    m = int(0.002 * fs)
    y[:m] += 0.8 * rng.standard_normal(m) * np.exp(-np.arange(m) / (0.0003 * fs))
    return y / (np.max(np.abs(y)) + 1e-12)


def render(rng: np.random.Generator, scene: Scene, room: spatial.SharedRoom, rx: spatial.Receiver) -> np.ndarray:
    cfg, fs = scene.clatter, scene.fs
    n = int(scene.duration_s * fs)
    out = rx.zeros()
    for start in np.cumsum(rng.exponential(1 / cfg.rate_hz, int(cfg.rate_hz * scene.duration_s * 2) + 5)):
        if start >= scene.duration_s:
            break
        src = spatial.floor_position(rng, cfg.radius_m, 0.75, scene.mic[:2])  # table height
        direct = rx.place_static(grain(rng, fs) * rng.uniform(0.3, 1.0), src, start)
        out += room.apply(direct, spatial.distance(src, scene.mic), rng) if scene.room.indoor else direct
    return out
