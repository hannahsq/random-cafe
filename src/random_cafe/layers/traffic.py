"""Traffic: passing vehicles on a straight lane (tyre noise brightening and growing with speed, plus engine
harmonics), moving past the microphone so distance and Doppler follow from geometry; and a distant road bed. Indoors
the whole stem goes through the wall / window (mass-law low-pass plus loss)."""
import numpy as np
from scipy import signal

from .. import spatial
from ..scene import Scene


def coloured_noise(rng: np.random.Generator, n: int, slope_db_oct: float, fs: int) -> np.ndarray:
    """noise whose power falls by slope_db_oct per octave (3: pink, 6: brown)."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec *= np.maximum(f, 20.0) ** (-slope_db_oct / 6.02)  # amplitude falls half as many dB per octave as power
    x = np.fft.irfft(spec, n)
    return x / (np.std(x) + 1e-12)


def vehicle(rng: np.random.Generator, n: int, fs: int, speed_ms: float) -> np.ndarray:
    tyre = coloured_noise(rng, n, 3.0, fs)
    corner = 400.0 + 25.0 * speed_ms  # faster: brighter
    tyre = signal.sosfilt(signal.butter(2, min(corner, 0.45 * fs), "lowpass", fs=fs, output="sos"), tyre)
    t = np.arange(n) / fs
    f0 = rng.uniform(25.0, 60.0) * (1 + 0.05 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t))
    phase = 2 * np.pi * np.cumsum(f0) / fs
    engine = sum(np.sin(k * phase) / k for k in range(1, 9))
    level = (speed_ms / 15.0) ** 1.5  # tyre noise grows steeply with speed
    return level * (tyre + rng.uniform(0.2, 0.8) * engine / np.std(engine))


def render(rng: np.random.Generator, scene: Scene) -> np.ndarray:
    cfg, fs = scene.traffic, scene.fs
    n = int(scene.duration_s * fs)
    out = np.zeros(n)
    window = 20.0  # vehicles passing up to 20 s before or after the clip still contribute
    count = rng.poisson(cfg.vehicles_per_min / 60 * (scene.duration_s + 2 * window))
    for _ in range(count):
        t_pass = rng.uniform(-window, scene.duration_s + window)
        v = rng.uniform(*cfg.speed_kmh) / 3.6 * rng.choice([-1.0, 1.0])
        d = rng.uniform(*cfg.lane_distance_m)

        def path(t, t_pass=t_pass, v=v, d=d):
            return np.stack([v * (t - t_pass), np.full_like(t, d), np.full_like(t, 0.5)], axis=1)

        out += spatial.place_moving(vehicle(rng, n, fs, abs(v)), path, scene.mic, fs)
    bed = coloured_noise(rng, n, 6.0, fs)
    env = np.interp(np.arange(n), np.linspace(0, n, 8), 1 + 0.3 * rng.standard_normal(8))
    out += bed * np.clip(env, 0.3, None) * np.std(out + 1e-9) * 10 ** (cfg.bed_db / 20)
    if scene.room.indoor:
        out = spatial.transmission(out, fs, rng.uniform(*scene.room.transmission_db))
    return out
