"""Traffic: passing vehicles on a straight lane (tyre noise brightening and growing with speed, plus engine
harmonics; some are prime movers: trucks and buses with deep diesel harmonics and air brakes), moving past the microphone so distance and Doppler follow from geometry; and a distant road bed. Indoors
the whole stem comes in through the facade: windows, and the front door when it is open."""
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


def vehicle(rng: np.random.Generator, n: int, fs: int, speed_ms: float, heavy: bool = False,
            t_pass_s: float = 0.0) -> np.ndarray:
    """A car, or with heavy=True a prime mover (truck or bus): a big diesel's deep, strong engine harmonics, more
    level, and sometimes an air-brake hiss around when it passes (heuristic values)."""
    tyre = coloured_noise(rng, n, 3.0, fs)
    corner = 400.0 + 25.0 * speed_ms  # faster: brighter
    tyre = signal.sosfilt(signal.butter(2, min(corner, 0.45 * fs), "lowpass", fs=fs, output="sos"), tyre)
    t = np.arange(n) / fs
    f0 = (rng.uniform(15.0, 30.0) if heavy else rng.uniform(25.0, 60.0)) * (1 + 0.05 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t))
    phase = 2 * np.pi * np.cumsum(f0) / fs
    engine = sum(np.sin(k * phase) / k ** (0.6 if heavy else 1.0) for k in range(1, 17 if heavy else 9))
    level = (speed_ms / 15.0) ** 1.5  # tyre noise grows steeply with speed
    x = level * (tyre + (rng.uniform(1.5, 3.0) if heavy else rng.uniform(0.2, 0.8)) * engine / np.std(engine))
    if heavy:
        x *= rng.uniform(1.8, 3.0)  # about +5 to +10 dB over a car
        if rng.uniform() < 0.3:  # air brake: a hiss of a second or so near the pass
            m = int(rng.uniform(0.6, 1.5) * fs)
            i = int((t_pass_s + rng.uniform(-3.0, 0.5)) * fs)
            if 0 <= i < n - m:
                hiss = signal.sosfilt(signal.butter(2, [1500, min(6000, 0.45 * fs)], "bandpass", fs=fs, output="sos"), rng.standard_normal(m))
                x[i:i + m] += np.std(x) * 1.5 * hiss / np.std(hiss) * np.exp(-np.arange(m) / (0.4 * fs))
    return x


def render(rng: np.random.Generator, scene: Scene, facade: spatial.Facade) -> np.ndarray:
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

        heavy = rng.uniform() < cfg.p_heavy
        if heavy:
            v *= 0.8  # prime movers run a little slower
        out += spatial.place_moving(vehicle(rng, n, fs, abs(v), heavy, t_pass), path, scene.mic, fs)
    bed = coloured_noise(rng, n, 6.0, fs)
    env = np.interp(np.arange(n), np.linspace(0, n, 8), 1 + 0.3 * rng.standard_normal(8))
    out += bed * np.clip(env, 0.3, None) * np.std(out + 1e-9) * 10 ** (cfg.bed_db / 20)
    if scene.room.indoor:
        out = facade(out, rng)
    return out
