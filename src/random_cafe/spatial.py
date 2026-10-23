"""Geometry for one microphone: propagation delay and 1/r gain (static or moving sources, the latter with Doppler),
talker directivity, positions drawn uniformly over floor area, a simple shared room, and wall transmission.
The fractional delay and the Brown-Duda head shadow follow umbrella-rain (github.com/hannahsq/umbrella-rain)."""
import numpy as np
from scipy import signal

C = 343.0  # speed of sound, m/s
HEAD_RADIUS = 0.0875


def add_fractional(buf: np.ndarray, y: np.ndarray, pos: float) -> None:
    """Add y into buf at fractional sample position pos (linear-interpolated delay)."""
    if pos < 0:
        k = int(np.ceil(-pos))
        y, pos = y[k:], pos + k
    i = int(pos)
    f = pos - i
    n = min(y.size, buf.size - i - 1)
    if n > 0:
        buf[i:i + n] += (1 - f) * y[:n]
        buf[i + 1:i + 1 + n] += f * y[:n]


def distance(src, mic) -> float:
    return float(np.linalg.norm(np.asarray(src, float) - np.asarray(mic, float)))


def place_static(x: np.ndarray, src, mic, fs: int, n_out: int, start: float = 0.0, r_min: float = 0.3) -> np.ndarray:
    """x emitted from src starting at `start` seconds, as heard at mic: delayed by r / C, scaled by 1 / r (unity
    at 1 m)."""
    r = max(distance(src, mic), r_min)
    out = np.zeros(n_out)
    add_fractional(out, x / r, (start + r / C) * fs)
    return out


def place_moving(x: np.ndarray, path, mic, fs: int, r_min: float = 0.5) -> np.ndarray:
    """x emitted by a source at path(t) (t: (n,) seconds -> (n, 3) metres), as heard at mic. The received sample at
    time t left the source at t - r / C (solved by fixed-point iteration), so the varying delay gives Doppler."""
    n = x.size
    t = np.arange(n) / fs
    te = t.copy()
    for _ in range(4):  # emission times
        r = np.maximum(np.linalg.norm(path(te) - np.asarray(mic, float), axis=1), r_min)
        te = t - r / C
    return np.interp(te * fs, np.arange(n), x, left=0.0, right=0.0) / r


def head_shadow(x: np.ndarray, theta: float, fs: int, a: float = HEAD_RADIUS) -> np.ndarray:
    """Brown & Duda (1998) one-pole / one-zero head shadow at incidence angle theta (radians from the facing
    direction). Used in reverse for talker directivity: a voice heard from behind its head loses its highs."""
    amin, tmin = 0.1, np.radians(150)
    alpha = 1 + amin / 2 + (1 - amin / 2) * np.cos(min(theta, tmin) / tmin * np.pi)
    beta, k = 2 * C / a, 2 * fs
    b = np.array([alpha * k + beta, beta - alpha * k]) / (k + beta)
    return signal.lfilter(b, [1.0, (beta - k) / (k + beta)], x)


def facing_angle(src, facing, mic) -> float:
    """Angle between a talker's facing direction (unit vector in the floor plane) and the direction to the mic."""
    v = np.asarray(mic, float) - np.asarray(src, float)
    v[2] = 0.0
    nv = np.linalg.norm(v)
    return float(np.arccos(np.clip(v @ np.asarray(facing, float) / max(nv, 1e-9), -1, 1)))


def floor_position(rng: np.random.Generator, r_range, height: float, centre=(0.0, 0.0)) -> np.ndarray:
    """A point drawn uniformly over the floor annulus r_range around centre: density uniform per area, so with 1 / r
    gain every log-spaced ring of distance carries about the same power (umbrella-rain's bed-layer argument), giving
    many quiet far sources and a few loud near ones."""
    lo, hi = r_range
    r = np.sqrt(rng.uniform(lo * lo, hi * hi))
    az = rng.uniform(0, 2 * np.pi)
    return np.array([centre[0] + r * np.cos(az), centre[1] + r * np.sin(az), height])


class SharedRoom:
    """One room per scene: a T60 drawn once, every source getting its own late tail (an exponentially decaying noise
    realisation of that T60, lower-passed as it decays) at a direct-to-reverberant ratio that falls with distance."""

    def __init__(self, rng: np.random.Generator, t60_range, drr_range, fs: int):
        self.fs, self.rng = fs, rng
        self.t60 = rng.uniform(*t60_range)
        self.drr_1m = rng.uniform(*drr_range)

    def tail(self) -> np.ndarray:
        n = int(self.t60 * self.fs)
        t = np.arange(n) / self.fs
        h = self.rng.standard_normal(n) * 10 ** (-3 * t / self.t60)
        h = signal.sosfilt(signal.butter(2, 0.35 * self.fs, "lowpass", fs=self.fs, output="sos"), h)
        return h / np.sqrt(np.sum(h ** 2))

    def apply(self, direct: np.ndarray, r: float) -> np.ndarray:
        """direct (already delayed and 1 / r scaled) plus a late tail whose level does not fall with distance."""
        drr = self.drr_1m - 20 * np.log10(max(r, 0.3))
        tail = signal.oaconvolve(direct * max(r, 0.3), self.tail())[:direct.size]
        g = np.sqrt(np.mean(direct ** 2) + 1e-20) / np.sqrt(np.mean(tail ** 2) + 1e-20) * 10 ** (-drr / 20)
        return direct + g * tail


def transmission(x: np.ndarray, fs: int, loss_db: float, corner_hz: float = 150.0) -> np.ndarray:
    """Outdoor sound through a wall or window: mass-law-like (about 6 dB more loss per octave above the corner)
    plus a broadband loss."""
    sos = signal.butter(1, corner_hz, "lowpass", fs=fs, output="sos")
    return signal.sosfilt(sos, x) * 10 ** (-loss_db / 20)
