"""Wind: turbulence buffeting the microphone (pseudo-sound: the flow's own pressure fluctuations at the capsule, not
sound arriving from elsewhere) and the broadband rush of wind around buildings and trees, both following gusts.
Buffeting is dynamic pressure, so its level goes as U^2; its spectrum is the inertial-range pressure spectrum (f^-7/3)
above a corner that rises with U, so it is almost all below a few hundred Hz. The rush is flow noise off obstacles,
growing faster (heuristic: amplitude as U^3) and brighter with speed. Indoors the microphone is sheltered: no
buffeting, and the rush comes in through the facade (windows, and the front door when it is open)."""
import numpy as np
from scipy import fft as sp_fft
from scipy import signal

from .. import spatial
from ..scene import Scene
from .noise import coloured_noise, ou

U_REF = 4.0  # m/s, the speed the stem level is referred to
EDDY_M = 0.1  # heuristic: eddy scale at the microphone; the buffeting corner is U / EDDY_M
RUSH = 0.3  # heuristic: rush against buffeting at U_REF


def buffeting(rng: np.random.Generator, n: int, fs: int, u: float) -> np.ndarray:
    """Pressure fluctuations of turbulence convected past the capsule: flat below U / EDDY_M, f^-7/3 above."""
    nfft = sp_fft.next_fast_len(n, real=True)
    f = sp_fft.rfftfreq(nfft, 1 / fs)
    shape = (1 + (f / (u / EDDY_M)) ** 2) ** (-7 / 12) * (f > 15.0)  # amplitude; below 15 Hz is not sound
    x = sp_fft.irfft(sp_fft.rfft(rng.standard_normal(nfft)) * shape, nfft)[:n]
    return x / (np.std(x) + 1e-12)


def render(rng: np.random.Generator, scene: Scene, facade: spatial.Facade, rx: spatial.Receiver) -> np.ndarray:
    cfg, fs = scene.wind, scene.fs
    n = int(scene.duration_s * fs)
    u = max(cfg.speed_ms, 0.1)
    # gusts: slow swells plus faster flutter, as a speed multiplier around 1
    g = np.clip(1 + cfg.gust_depth * (ou(rng, n, fs, cfg.gust_time_s) + 0.5 * ou(rng, n, fs, 0.4)), 0.1, None)
    rush = coloured_noise(rng, n, 3.0, fs)
    rush = signal.sosfilt(signal.butter(2, [80.0, min(200.0 + 60.0 * u * g.mean(), 0.45 * fs)], "bandpass", fs=fs,
                                        output="sos"), rush)
    rush = RUSH * rush / (np.std(rush) + 1e-12) * (u * g / U_REF) ** 3
    if scene.room.indoor:
        return facade(rush, rng)
    buffet = buffeting(rng, n, fs, u) * (u * g / U_REF) ** 2
    if not rx.binaural:
        return buffet + rush
    # each ear buffeted by its own turbulence (the eddies are smaller than a head); the rush is a diffuse field
    other = buffeting(rng.spawn(1)[0], n, fs, u) * (u * g / U_REF) ** 2
    return np.stack([buffet, other]) + rx.diffuse(rush, rng)
