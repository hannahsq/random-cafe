"""Rain on open ground around the microphone, after umbrella-rain's ground layer (github.com/hannahsq/umbrella-rain),
kept cheap enough for training data. Drop sizes follow Marshall-Palmer weighted by fall speed (Atlas et al. 1973),
in a few size bins. Each drop is an impact click plus a splash of band-limited noise whose colour follows its contact
time d / v, and a short burst near 11 kHz at the first instant of contact (fitted; see SPIKE). Only large drops
landing in puddles ring a Minnaert bubble (f = 3.26 / R, damping after van den Doel 2005).
(Small drops' regular bubbles, 13-15 kHz, are left out; the spike, above the Nyquist frequency at 16 kHz, is too.)

The nearest drops of each size are rendered one by one, up to a budget per bin; everything else is the dense rain
further out. Ground area grows as rho^2 while intensity falls as 1 / r^2, so every log-spaced ring carries similar
power, and so many drops overlap there that their sum is Gaussian: it is rendered as noise through the same grains,
at the power the rings give in closed form. The rain rate drifts slowly (a gust envelope).

Indoors: drops on a window pane, plus the outdoor rain through the facade (windows, and the front door when it is
open). Wind blows rain sideways at its own speed U while drops fall at v_t, so a vertical pane facing the wind takes
U / v_t of the flux onto the ground (more of the small, slow drops); the facade's angle to the wind is drawn per
scene. The pane's dense drops are a continuum too, so a windward window is a rain bed of its own."""
import numpy as np
from scipy import fft as sp_fft
from scipy import signal

from .. import spatial
from ..scene import Scene
from .noise import ou

EDGES_MM = np.geomspace(0.3, 6.0, 7)  # drop-size bins
VARIANTS = 3  # grain realisations per bin, so near drops don't all share one waveform
SPLASH_RATIO = 0.4  # heuristic (umbrella-rain): splash level against the click
# fitted to recordings (rain on a street, greenery, an umbrella and polycarbonate; Hannah's, eval only): every
# impact also has a burst of ~0.15 ms near 11 kHz, at about the same frequency whatever the drop's size, giving the
# bump every rain recording shows at 10-13 kHz. Likely the air cushion trapped under a landing drop collapsing within
# microseconds (the contact itself is far too long to make it). Its level against the click fitted so the ground
# rain's spectrum matches the recordings' at 10-16 kHz (within ~3 dB); the same for every size. Regular-entrainment
# bubbles (0.8-1.1 mm drops in water) ring there too, but would need to be ~25 dB over their impact to explain it
SPIKE_BAND = (10000.0, 13500.0)
SPIKE_DECAY = 1.5e-4
SPIKE = 1.0
SPIKE_EXP = 0.0  # amplitude against the click goes as (d / 2 mm)^SPIKE_EXP
JITTER = 0.4  # heuristic: lognormal spread of impact strength within a bin (sizes, angles, surfaces)
BUBBLE_MIN_MM = 2.2  # drops this large entrain bubbles irregularly when they land in water
P_BUBBLE = 0.5  # heuristic: chance such a drop landing in a puddle rings
BUBBLE_LEVEL = 2.0  # heuristic (umbrella-rain): bubble level against the splash
SECTORS = 16  # binaural only: azimuth sectors for the near drops
PANE_M2 = 2.25  # a 1.5 x 1.5 m window
PANE_SHARE_MIN = 0.005  # heuristic: share of the rain eddies bring onto a sheltered (leeward) pane


def terminal_velocity(d_mm):
    """Atlas et al. (1973) fit, m/s."""
    return np.maximum(9.65 - 10.3 * np.exp(-0.6 * np.asarray(d_mm, float)), 0.1)


def drop_flux(rain_mm_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Drops per m^2 per second in each size bin (Marshall-Palmer, N0 = 8000 m^-3 mm^-1, times fall speed), and each
    bin's flux-weighted mean diameter in mm."""
    lam = 4.1 * rain_mm_h ** -0.21
    flux, mean = [], []
    for lo, hi in zip(EDGES_MM[:-1], EDGES_MM[1:]):
        d = np.linspace(lo, hi, 200)
        f = 8000.0 * np.exp(-lam * d) * terminal_velocity(d)
        flux.append(np.trapezoid(f, d))
        mean.append(np.trapezoid(f * d, d) / max(flux[-1], 1e-30))
    return np.array(flux), np.array(mean)


def strength(d_mm: float) -> tuple[float, float]:
    """(click, splash) amplitudes against a 2 mm drop's click. The click is the impact force J / tau^2 with impulse
    J = m v and contact time tau = d / v, so it goes as d v^3; the splash grows with size, speed and Weber number."""
    v, v_ref = float(terminal_velocity(d_mm)), float(terminal_velocity(2.0))
    weber = 1000.0 * d_mm * 1e-3 * v * v / 0.072
    return d_mm * v ** 3 / (2.0 * v_ref ** 3), SPLASH_RATIO * (d_mm / 2) ** 1.5 * (v / v_ref) * weber / (weber + 100)


def grain(rng: np.random.Generator, d_mm: float, fs: int, contact_s: float | None = None,
          splash_share: float = 1.0) -> np.ndarray:
    """One drop on hard ground: a one-cycle click as long as the contact, plus a splash of noise in 0.4 / tau to
    2 / tau (a pulse that short has little energy above 1 / tau) with a 0.3 ms attack and a size-dependent decay.
    On a yielding surface (contact_s longer than the drop's own d / v) the click is as long as that contact and its
    force (impulse / contact time) as much weaker; the splash is the water's own, but a surface that soaks the drop
    up keeps only splash_share of it."""
    tau = d_mm * 1e-3 / float(terminal_velocity(d_mm))
    click, splash = strength(d_mm)
    splash *= splash_share
    t_click = max(contact_s or tau, tau)
    hard = click  # the spike is the drop's (its trapped air), whatever it lands on
    click *= (tau / t_click) ** 2
    decay = 1e-3 * d_mm ** 0.7
    n = int(5 * decay * fs) + 1
    t = np.arange(n) / fs
    hi = min(2 / tau, 0.45 * fs)
    noise = signal.sosfilt(signal.butter(2, [min(0.4 / tau, hi / 2), hi], "bandpass", fs=fs, output="sos"),
                           rng.standard_normal(2 * n))[n:]  # past the filter's start-up
    y = splash * noise / (np.std(noise) + 1e-12) * np.minimum(t / 3e-4, 1) * np.exp(-t / decay)
    tt = (np.arange(int(np.ceil(t_click * fs))) + 0.5) / fs
    y = np.pad(y, (0, max(0, tt.size - y.size)))
    y[:tt.size] += click * np.sin(2 * np.pi * tt / t_click)
    if SPIKE and SPIKE_BAND[1] < 0.45 * fs:  # (above Nyquist at 16 kHz: left out, and no draw)
        m = min(int(8 * SPIKE_DECAY * fs) + 1, n)
        burst = signal.sosfilt(signal.butter(2, SPIKE_BAND, "bandpass", fs=fs, output="sos"),
                               rng.standard_normal(m) * np.exp(-np.arange(m) / fs / SPIKE_DECAY))
        y[:m] += SPIKE * (d_mm / 2) ** SPIKE_EXP * hard * burst / (np.sqrt(np.sum(burst ** 2) / fs / SPIKE_DECAY * 2) + 1e-12)
    return y


def envelope(rng: np.random.Generator, scene: Scene) -> np.ndarray:
    """Rain-rate multiplier per sample: Ornstein-Uhlenbeck around 1 (umbrella-rain's gusts)."""
    cfg = scene.rain
    return np.clip(1 + cfg.gust_depth * ou(rng, int(scene.duration_s * scene.fs), scene.fs, cfg.gust_time_s), 0.2, None)


def bubble(rng: np.random.Generator, fs: int) -> tuple[np.ndarray, float]:
    """A Minnaert bubble (radius around 1.2 mm, so near 2.7 kHz), its pitch rising slightly as it decays; returns the
    waveform and the radius relative to the median."""
    r = np.clip(1.2 * rng.lognormal(0, 0.35), 0.4, 3.0) * 1e-3
    f0 = min(3.26 / r, 0.45 * fs)
    damp = 0.13 / r + 0.0072 * r ** -1.5  # 1/s
    t = np.arange(int(6 / damp * fs)) / fs
    return np.sin(2 * np.pi * f0 * (t + 0.05 * damp * t * t)) * np.exp(-damp * t), r / 1.2e-3


def field(rng: np.random.Generator, scene: Scene, env: np.ndarray, rx: spatial.Receiver | None = None,
          pane_m: float | None = None, windward: float = 0.0) -> np.ndarray:
    """Rain on the ground around the microphone, or with pane_m on a window that far away, facing the wind by
    `windward` (the cosine of its angle to the wind, 0 for a sheltered pane). The near drops are added
    grain by grain (vectorised); the rest, whose sum is Gaussian, is one white noise coloured by every bin's grain
    spectrum at the power the rings give, and modulated by the (slow) gust envelope.

    With a binaural rx (ground only), near drops are gathered into SECTORS azimuth sectors at two elevations (steep,
    within the microphone's height, and shallow beyond), each heard through one HRIR, and the dense rain is diffuse;
    their azimuths come from a child generator, so the drops themselves are the mono ones."""
    cfg, fs = scene.rain, scene.fs
    n = env.size
    nfft = sp_fft.next_fast_len(n, real=True)
    flux, sizes = drop_flux(cfg.rain_mm_h)
    h, rho_max, peak = scene.mic[2], cfg.radius_m, env.max()
    binaural = rx is not None and rx.binaural and pane_m is None
    n_sec = 2 * SECTORS if binaural else 1
    az_rng = rng.spawn(1)[0] if binaural else None
    buf, dense = np.zeros(n_sec * n), np.zeros(nfft // 2 + 1)
    pulses = []
    for f_b, d_b in zip(flux, sizes):
        kernels = [grain(rng, d_b, fs) for _ in range(VARIANTS)]
        if pane_m is None:  # explicit drops within rho_b, as many as the budget allows
            rho_b = min(np.sqrt(cfg.grains_hz / max(f_b, 1e-12) / np.pi), rho_max)
            rate = f_b * np.pi * rho_b ** 2
        else:  # everything lands at one distance: the budget one by one, the rest dense
            on_pane = f_b * PANE_M2 * max(scene.wind.speed_ms / float(terminal_velocity(d_b)) * windward, PANE_SHARE_MIN)
            rate = min(on_pane, cfg.grains_hz)
        k = rng.poisson(rate * peak * scene.duration_s)
        i = rng.integers(0, n, k)
        keep = rng.uniform(size=k) < env[i] / peak  # thinned by the gusts
        r = np.full(k, pane_m) if pane_m is not None else np.hypot(rho_b * np.sqrt(rng.uniform(size=k)), h)
        amp = rng.lognormal(0, JITTER, k) / r
        delay = np.round(r / spatial.C * fs).astype(int)
        v = rng.integers(0, VARIANTS, k)
        wet = (rng.uniform(size=k) < cfg.puddles * P_BUBBLE) & keep & (d_b >= BUBBLE_MIN_MM) & (pane_m is None)
        sec = np.zeros(k, int)
        if binaural:
            sec = (az_rng.uniform(0, SECTORS, k)).astype(int) + SECTORS * (r < h * np.sqrt(2))  # steep: within 45 deg
        for j, kern in enumerate(kernels):
            sel = keep & (v == j)
            idx = (sec[sel] * n + i[sel] + delay[sel])[:, None] + np.arange(kern.size)
            idx = np.minimum(idx, (sec[sel] * n + n)[:, None])  # a grain running past the end goes to a spill sample
            buf += np.bincount(idx.ravel(), (amp[sel, None] * kern).ravel(), n_sec * n + 1)[:n_sec * n]
        if pane_m is None and rho_b < rho_max:  # the dense rain beyond: sum of a^2 per second over the rings
            power = f_b * np.exp(2 * JITTER ** 2) * np.pi * np.log((rho_max ** 2 + h * h) / (rho_b ** 2 + h * h))
            dense += power / fs * np.abs(sp_fft.rfft(kernels[0], nfft)) ** 2
        elif pane_m is not None and on_pane > rate:
            dense += (on_pane - rate) * np.exp(2 * JITTER ** 2) / pane_m ** 2 / fs * np.abs(sp_fft.rfft(kernels[0], nfft)) ** 2
        pulses += [(i[j] + delay[j], amp[j] * strength(d_b)[1], sec[j]) for j in np.flatnonzero(wet)]
    near = buf.reshape(n_sec, n)
    far = np.zeros(n)
    if dense.any():  # stationary noise, so the circular convolution is harmless
        far = sp_fft.irfft(sp_fft.rfft(rng.standard_normal(nfft)) * np.sqrt(dense), nfft)[:n] * np.sqrt(env)
    if not binaural:
        out = near[0]
        out += far
        for pos, a, _ in pulses:  # bubbles pinch off 8-30 ms after the impact
            y, rel = bubble(rng, fs)
            spatial.add_fractional(out, BUBBLE_LEVEL * a * rel * y, pos + rng.uniform(0.008, 0.03) * fs)
        return out
    for pos, a, s in pulses:
        y, rel = bubble(rng, fs)
        spatial.add_fractional(near[s], BUBBLE_LEVEL * a * rel * y, pos + rng.uniform(0.008, 0.03) * fs)
    out = rx.diffuse(far, rng)
    az = 2 * np.pi * (np.arange(2 * SECTORS) % SECTORS + 0.5) / SECTORS
    el = np.radians(np.where(np.arange(2 * SECTORS) < SECTORS, -20.0, -60.0))
    for s in np.flatnonzero(near.any(axis=1)):
        out += rx.toward(near[s], rx.mic + [np.cos(el[s]) * np.cos(az[s]), np.cos(el[s]) * np.sin(az[s]), np.sin(el[s])])
    return out


def render(rng: np.random.Generator, scene: Scene, room: spatial.SharedRoom, facade: spatial.Facade,
           rx: spatial.Receiver) -> np.ndarray:
    env = envelope(rng, scene)
    if not scene.room.indoor:
        return field(rng, scene, env, rx)
    outdoor = facade(field(rng, scene, env), rng)
    r_pane, windward = rng.uniform(2.0, 4.0), max(np.cos(rng.uniform(0, 2 * np.pi)), 0.0)
    pane = field(rng, scene, env, pane_m=r_pane, windward=windward)  # glass: a hard, bright tick
    pane = signal.sosfilt(signal.butter(2, 2000, "highpass", fs=scene.fs, output="sos"), pane) * 10 ** (scene.rain.pane_db / 20)
    return outdoor + room.apply(rx.toward(pane, facade.opening), r_pane, rng)  # the pane is in the facade
