"""Noise shared by the layers: power-law coloured noise and slow random fluctuations."""
import numpy as np
from scipy import signal


def coloured_noise(rng: np.random.Generator, n: int, slope_db_oct: float, fs: int) -> np.ndarray:
    """noise whose power falls by slope_db_oct per octave (3: pink, 6: brown)."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec *= np.maximum(f, 20.0) ** (-slope_db_oct / 6.02)  # amplitude falls half as many dB per octave as power
    x = np.fft.irfft(spec, n)
    return x / (np.std(x) + 1e-12)


def ou(rng: np.random.Generator, n: int, fs: float, time_s: float) -> np.ndarray:
    """Unit-variance Ornstein-Uhlenbeck noise with correlation time time_s, on a 10 ms grid, per sample."""
    k = int(n / fs / 0.01) + 2
    a = np.exp(-0.01 / time_s)
    x = rng.standard_normal(k)
    x[0] /= np.sqrt(1 - a * a)  # start in the stationary state
    x = signal.lfilter([np.sqrt(1 - a * a)], [1, -a], x)
    return np.interp(np.arange(n) / fs, np.arange(k) * 0.01, x)
