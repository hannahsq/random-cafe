"""Rain: Poisson drop impacts, each a short tick with sometimes a Minnaert bubble (an entrained air bubble ringing at
f = 3.26 / R Hz, R its radius in metres), on the ground around the microphone outdoors. Indoors: bright ticks on a
window pane a few metres away plus the outdoor rain through the wall. Physics simplified from umbrella-rain
(github.com/hannahsq/umbrella-rain)."""
import numpy as np
from scipy import signal

from .. import spatial
from ..scene import Scene


def drop(rng: np.random.Generator, fs: int, glass: bool) -> np.ndarray:
    n = int(0.06 * fs)
    t = np.arange(n) / fs
    tick = rng.standard_normal(n) * np.exp(-t / rng.uniform(0.3e-3, 1.5e-3))
    if glass:
        tick = signal.sosfilt(signal.butter(2, 2000, "highpass", fs=fs, output="sos"), tick)
    elif rng.uniform() < 0.3:  # bubble
        r_m = rng.uniform(0.5e-3, 2.5e-3)
        f = min(3.26 / r_m, 0.45 * fs)
        tick += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * f * t) * np.exp(-t / rng.uniform(0.004, 0.015))
    return tick


def field(rng: np.random.Generator, scene: Scene, rate_hz: float, glass: bool) -> np.ndarray:
    fs = scene.fs
    n = int(scene.duration_s * fs)
    out = np.zeros(n)
    k = rng.poisson(rate_hz * scene.duration_s)
    starts = np.sort(rng.uniform(0, scene.duration_s, k))
    for s in starts:
        if glass:  # a window pane 2-4 m away, 1.5 x 1.5 m
            src = np.array([rng.uniform(2.0, 4.0), rng.uniform(-0.75, 0.75), rng.uniform(1.0, 2.5)])
        else:
            src = spatial.floor_position(rng, (0.3, 15.0), 0.0, scene.mic[:2])
        out += spatial.place_static(drop(rng, fs, glass) * rng.lognormal(0, 0.6), src, scene.mic, fs, n, s)
    return out


def render(rng: np.random.Generator, scene: Scene, room: spatial.SharedRoom) -> np.ndarray:
    rate = 150.0 * scene.rain.rain_mm_h  # impacts per second heard (heuristic)
    if not scene.room.indoor:
        return field(rng, scene, rate, glass=False)
    outdoor = spatial.transmission(field(rng, scene, rate, glass=False), scene.fs, rng.uniform(*scene.room.transmission_db))
    pane = room.apply(field(rng, scene, rate * 0.02, glass=True), 3.0)
    return outdoor + pane
