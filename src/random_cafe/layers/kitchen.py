"""Kitchen: an extractor fan (broadband rumble plus motor hum), dish and pan clatter (denser, more metallic grains than
the café's), and espresso-machine steam bursts, at one place in the building. Open: it shares the café's room.
Closed: heard through a wall or door (mass-law low-pass plus loss)."""
import numpy as np
from scipy import signal

from .. import spatial
from ..scene import Scene
from .clatter import grain
from .noise import coloured_noise


def render(rng: np.random.Generator, scene: Scene, room: spatial.SharedRoom, rx: spatial.Receiver) -> np.ndarray:
    cfg, fs = scene.kitchen, scene.fs
    n = int(scene.duration_s * fs)
    t = np.arange(n) / fs
    src = spatial.floor_position(rng, cfg.distance_m, 1.2, scene.mic[:2])
    # extractor: brown-ish rumble with the motor's hum and its harmonics
    hum_f = rng.uniform(45.0, 120.0)
    fan = coloured_noise(rng, n, 4.0, fs) + 0.5 * sum(np.sin(2 * np.pi * k * hum_f * t + rng.uniform(0, 6.3)) / k for k in range(1, 6))
    x = 0.3 * fan / np.std(fan)
    # dishes and pans: metallic grains, Poisson
    for start in np.cumsum(rng.exponential(1 / cfg.clatter_rate_hz, int(cfg.clatter_rate_hz * scene.duration_s * 2) + 5)):
        if start >= scene.duration_s:
            break
        g = grain(rng, fs) * rng.uniform(0.3, 1.5)
        spatial.add_fractional(x, g, start * fs)
    # espresso steam: band-passed hiss bursts of a few seconds
    for start in np.cumsum(rng.exponential(1 / cfg.steam_rate_hz, 8)):
        if start >= scene.duration_s:
            break
        dur = rng.uniform(1.5, 5.0)
        m = int(dur * fs)
        hiss = signal.sosfilt(signal.butter(2, [2500, min(7500, 0.45 * fs)], "bandpass", fs=fs, output="sos"), rng.standard_normal(m))
        env = np.minimum(1, np.minimum(np.arange(m), m - np.arange(m)) / (0.15 * fs))
        spatial.add_fractional(x, 0.6 * hiss * env, start * fs)
    direct = rx.place_static(x, src)
    if not cfg.open:  # through a door (loss climbing only above the mid range), then into the café's room
        direct = spatial.transmission(direct, fs, rng.uniform(*cfg.wall_db), corner_hz=800.0)
    return room.apply(direct, spatial.distance(src, scene.mic), rng) if scene.room.indoor else direct
