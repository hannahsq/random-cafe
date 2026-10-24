"""Geometry for one microphone: propagation delay and 1/r gain (static or moving sources, the latter with Doppler),
talker directivity, positions drawn uniformly over floor area, a simple shared room, and wall transmission. With an
Hrtf, a Receiver renders a listener's two ears instead of the microphone (for listening previews).
The fractional delay, the Brown-Duda head shadow and the SOFA HRIR handling follow umbrella-rain
(github.com/hannahsq/umbrella-rain)."""
from fractions import Fraction

import numpy as np
from scipy import fft as sp_fft
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


def place_moving(x: np.ndarray, path, mic, fs: int, r_min: float = 0.5, t0: float = 0.0,
                 n: int | None = None, span: tuple[int, int] | None = None) -> np.ndarray:
    """x emitted by a source at path(t) (t: (n,) seconds -> (n, 3) metres), as heard at mic over n samples (default
    x.size); x[k] leaves the source at t0 + k / fs, so a negative t0 covers sound emitted before the clip starts.
    The received sample at time t left the source at t - r / C (solved by fixed-point iteration), so the varying
    delay gives Doppler. The delay is smooth, so it is solved every 16 samples and interpolated; with span, only
    received samples span[0]:span[1] are computed (the rest are silent)."""
    n = x.size if n is None else n
    i0, i1 = span if span is not None else (0, n)
    out = np.zeros(n)
    if i1 <= i0:
        return out
    t = np.arange(i0, i1) / fs
    tc = np.r_[t[::16], t[-1]]
    te = tc.copy()
    for _ in range(4):  # emission times
        r = np.maximum(np.linalg.norm(path(te) - np.asarray(mic, float), axis=1), r_min)
        te = tc - r / C
    te, r = np.interp(t, tc, te), np.interp(t, tc, r)
    out[i0:i1] = np.interp((te - t0) * fs, np.arange(x.size), x, left=0.0, right=0.0) / r
    return out


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
    realisation of that T60, drawn by the calling layer) at a direct-to-reverberant ratio that falls with distance."""

    def __init__(self, rng: np.random.Generator, t60_range, drr_range, fs: int):
        self.fs = fs
        self.t60 = rng.uniform(*t60_range)
        self.drr_1m = rng.uniform(*drr_range)

    def tail(self, rng: np.random.Generator) -> np.ndarray:
        n = int(self.t60 * self.fs)
        t = np.arange(n) / self.fs
        h = rng.standard_normal(n) * 10 ** (-3 * t / self.t60)
        h = signal.sosfilt(signal.butter(2, 0.35 * self.fs, "lowpass", fs=self.fs, output="sos"), h)
        return h / np.sqrt(np.sum(h ** 2))

    def apply(self, direct: np.ndarray, r: float, rng: np.random.Generator) -> np.ndarray:
        """direct (already delayed and 1 / r scaled) plus a late tail whose level does not fall with distance; the
        tail's noise comes from the caller's generator, so layers stay independent of each other. Binaural direct
        sound (2, n) gets a diffuse tail."""
        drr = self.drr_1m - 20 * np.log10(max(r, 0.3))
        mono = direct if direct.ndim == 1 else direct.mean(0)
        tail = signal.oaconvolve(mono * max(r, 0.3), self.tail(rng))[:mono.size]
        g = np.sqrt(np.mean(direct ** 2) + 1e-20) / np.sqrt(np.mean(tail ** 2) + 1e-20) * 10 ** (-drr / 20)
        if direct.ndim == 1:
            return direct + g * tail
        return direct + g * diffuse(tail, rng.spawn(1)[0], self.fs)


class Facade:
    """The café's street side, drawn once per scene so every outdoor source agrees on it: window loss, whether the
    front door is open (and its loss), and the facade's distance from the microphone. Outdoor sound comes in through
    the windows (mass-law loss) plus, with the front door open, through the opening (little loss, broadband), and then
    reverberates in the café's room."""

    def __init__(self, rng: np.random.Generator, room_cfg, room: "SharedRoom", fs: int, rx: "Receiver | None" = None):
        self.fs, self.room, self.corner, self.rx = fs, room, room_cfg.window_corner_hz, rx
        self.window_db = rng.uniform(*room_cfg.window_db)
        self.door_open = bool(rng.uniform() < room_cfg.p_door_open)
        self.door_db = rng.uniform(*room_cfg.door_db)
        self.distance = rng.uniform(*room_cfg.facade_distance_m)
        # the street side is +y, where the traffic lane runs; the window (or door) is somewhere along it
        self.opening = np.array([rng.uniform(-3.0, 3.0), self.distance, 0.3])

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        y = transmission(x, self.fs, self.window_db, corner_hz=self.corner)
        if self.door_open:
            y = y + x * 10 ** (-self.door_db / 20)
        # the losses are facade insulation (outdoor level minus indoor level, D = L_out - L_in, as measured), which
        # already includes spreading and the room: the room shapes the transmitted sound but keeps its level
        heard = self.rx.toward(y, self.opening) if self.rx is not None else y  # binaural: from the opening
        z = self.room.apply(heard, self.distance, rng)
        return z * np.sqrt(np.mean(y ** 2) / (np.mean(z ** 2) + 1e-30))


def transmission(x: np.ndarray, fs: int, loss_db: float, corner_hz: float = 150.0, hf_db: float = 6.0) -> np.ndarray:
    """Outdoor sound through a wall or window: a broadband loss, plus more above the corner (mass law, ~6 dB per
    octave) up to hf_db more: a real facade's insulation levels off (leaks, the coincidence dip of its glazing)
    rather than growing without end (heuristic)."""
    sos = signal.butter(1, corner_hz, "lowpass", fs=fs, output="sos")
    lo = signal.sosfilt(sos, x)
    return (lo + (x - lo) * 10 ** (-hf_db / 20)) * 10 ** (-loss_db / 20)


def diffuse(x: np.ndarray, rng: np.random.Generator, fs: int, crossover_hz: float = 700.0) -> np.ndarray:
    """Two ears' versions (2, n) of a diffuse sound: identical below the crossover, where a diffuse field is coherent
    across a head, and independently decorrelated above it by short decaying-noise filters (heuristic), joined by a
    Linkwitz-Riley crossover so the bands sum flat."""
    lp = signal.butter(2, crossover_hz, "lowpass", fs=fs, output="sos")
    hp = signal.butter(2, crossover_hz, "highpass", fs=fs, output="sos")
    low = signal.sosfilt(lp, signal.sosfilt(lp, x))
    m = int(0.02 * fs)
    ears = []
    for _ in range(2):
        h = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.005 * fs))
        y = signal.oaconvolve(x, h / np.sqrt(np.sum(h ** 2)))[:x.size]
        ears.append(low + signal.sosfilt(hp, signal.sosfilt(hp, y)))
    return np.stack(ears)


def diffuse_field_equalise(ir: np.ndarray) -> np.ndarray:
    """HRIRs (M, 2, N) divided by their diffuse-field response (the RMS over directions and ears at each frequency,
    third-octave smoothed) through a minimum-phase filter, so they stay causal. On average over directions every
    frequency then passes at unit gain, as for the omnidirectional microphone."""
    n = ir.shape[-1]
    nfft = sp_fft.next_fast_len(8 * n, real=True)
    spec = sp_fft.rfft(ir, nfft)
    power = np.mean(np.abs(spec) ** 2, axis=(0, 1))
    k = np.arange(power.size)
    lo, hi = np.floor(k * 2 ** (-1 / 6)).astype(int), np.ceil(k * 2 ** (1 / 6)).astype(int) + 1
    cum = np.concatenate([[0.0], np.cumsum(power)])
    smooth = (cum[np.minimum(hi, power.size)] - cum[lo]) / (np.minimum(hi, power.size) - lo)
    cep = sp_fft.irfft(-0.5 * np.log(np.maximum(smooth, 1e-12 * smooth.max())), nfft)  # log of 1 / magnitude
    cep[1:nfft // 2] *= 2  # fold to minimum phase
    cep[nfft // 2 + 1:] = 0
    eq = np.exp(sp_fft.rfft(cep, nfft))
    return sp_fft.irfft(spec * eq, nfft)[..., :2 * n]


class Hrtf:
    """Head-related impulse responses for binaural previews: the nearest measured direction is used, with its two
    ears' delays (the interaural time difference) baked into the filters relative to the head centre. Diffuse-field
    equalised, so binaural stems keep the mono level at every frequency on average. After umbrella-rain's
    Spatialiser (without its near-field parallax: sources here are a metre or more away)."""

    def __init__(self, ir: np.ndarray, dirs: np.ndarray, delay: np.ndarray, fs_in: float, fs: int):
        """ir (M, 2, N) left / right (left = +y), dirs (M, 3) head frame (x front, y left, z up), delay (M, 2) in
        samples at fs_in."""
        if int(round(fs_in)) != fs:
            ratio = Fraction(fs, int(round(fs_in)))
            ir = signal.resample_poly(ir, ratio.numerator, ratio.denominator, axis=-1)
            delay = delay * fs / fs_in
        ir = diffuse_field_equalise(ir)
        front = int(np.argmax(dirs @ np.array([1.0, 0.0, 0.0]) / np.linalg.norm(dirs, axis=1)))
        delay = delay - delay[front].mean()  # relative to the head centre, roughly
        self.lead = int(np.ceil(max(-delay.min(), 0.0))) + 2  # a fixed latency so that every shift is causal
        n = ir.shape[-1] + self.lead + int(np.ceil(max(delay.max(), 0.0))) + 2
        nfft = sp_fft.next_fast_len(2 * n, real=True)
        shift = np.exp(-2j * np.pi * sp_fft.rfftfreq(nfft) * (self.lead + delay)[..., None])
        ir = sp_fft.irfft(sp_fft.rfft(ir, nfft) * shift, nfft)[..., :n]
        self.ir = ir / np.sqrt(np.mean(np.sum(ir ** 2, axis=-1)))
        self.dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        self.fs = fs
        self._spectra = {}

    @classmethod
    def from_sofa(cls, path: str, fs: int) -> "Hrtf":
        """A SimpleFreeFieldHRIR .sofa file (needs h5py: pip install random-cafe[binaural])."""
        import h5py
        with h5py.File(path, "r") as f:
            ir = np.asarray(f["Data.IR"], float)
            delay = np.broadcast_to(np.asarray(f["Data.Delay"], float), ir.shape[:2]).copy()
            pos = np.asarray(f["SourcePosition"], float)
            kind = f["SourcePosition"].attrs.get("Type", b"spherical")
            fs_in = float(np.ravel(f["Data.SamplingRate"])[0])
            rec = np.asarray(f["ReceiverPosition"], float).reshape(ir.shape[1], 3, -1)[:, :, 0]
        if (kind.decode() if isinstance(kind, bytes) else str(kind)).lower().startswith("spher"):
            az, el = np.radians(pos[:, 0]), np.radians(pos[:, 1])
            pos = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
        if rec[0, 1] < rec[1, 1]:  # channel 0 must be the +y (left) ear
            ir, delay = ir[:, ::-1], delay[:, ::-1]
        return cls(ir, pos, delay, fs_in, fs)

    def index(self, v: np.ndarray) -> np.ndarray:
        """Nearest measured direction for head-frame vectors v (k, 3)."""
        return np.argmax(np.atleast_2d(v) @ self.dirs.T, axis=1)

    def spectra(self, nfft: int) -> np.ndarray:
        if nfft not in self._spectra:
            self._spectra[nfft] = sp_fft.rfft(self.ir, nfft)
        return self._spectra[nfft]


class Receiver:
    """Where a scene is heard: the microphone (mono, (n,)), or with an Hrtf a listener's two ears ((2, n)) at the
    microphone's position, facing yaw_deg (0: +x, 90: +y, the street side)."""
    HOP = 256  # moving sources: the HRIR changes every HOP samples, crossfaded

    def __init__(self, mic, fs: int, n: int, hrtf: Hrtf | None = None, yaw_deg: float = 90.0):
        self.mic, self.fs, self.n, self.hrtf = np.asarray(mic, float), fs, n, hrtf
        c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
        self.to_head = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    @property
    def binaural(self) -> bool:
        return self.hrtf is not None

    def zeros(self) -> np.ndarray:
        return np.zeros((2, self.n) if self.binaural else self.n)

    def direction(self, src) -> np.ndarray:
        """HRIR indices for world positions src (k, 3)."""
        return self.hrtf.index((np.atleast_2d(np.asarray(src, float)) - self.mic) @ self.to_head.T)

    def place_static(self, x: np.ndarray, src, start: float = 0.0, r_min: float = 0.3) -> np.ndarray:
        if not self.binaural:
            return place_static(x, src, self.mic, self.fs, self.n, start, r_min)
        r = max(distance(src, self.mic), r_min)
        h = self.hrtf.ir[self.direction(src)[0]]
        out = np.zeros((2, self.n))
        for e in (0, 1):
            add_fractional(out[e], signal.oaconvolve(x, h[e]) / r, (start + r / C) * self.fs - self.hrtf.lead)
        return out

    def toward(self, x: np.ndarray, src) -> np.ndarray:
        """x, whose level and timing are already set, heard from the direction of src."""
        if not self.binaural:
            return x
        h, lead = self.hrtf.ir[self.direction(src)[0]], self.hrtf.lead
        return np.stack([signal.oaconvolve(x, h[e])[lead:lead + x.size] for e in (0, 1)])

    def place_moving(self, x: np.ndarray, path, r_min: float = 0.5, t0: float = 0.0,
                     span: tuple[int, int] | None = None) -> np.ndarray:
        """Doppler and 1 / r from the head centre; binaurally, the HRIR follows the source frame by frame (Hann
        windows at 50 % overlap, each filtered for its own direction and overlap-added)."""
        y = place_moving(x, path, self.mic, self.fs, r_min, t0, self.n, span)
        if not self.binaural:
            return y
        hop, n = self.HOP, y.size
        m = -(-n // hop) + 1
        pad = np.zeros((m + 1) * hop)
        pad[hop:hop + n] = y
        frames = np.lib.stride_tricks.sliding_window_view(pad, 2 * hop)[::hop][:m] * np.hanning(2 * hop + 1)[:-1]
        idx = self.direction(path(np.arange(m) * hop / self.fs))  # frame k is centred on sample k * hop
        nfft = sp_fft.next_fast_len(2 * hop + self.hrtf.ir.shape[-1], real=True)
        spec = sp_fft.rfft(frames, nfft)
        out = np.zeros((2, n))
        for e in (0, 1):
            seg = sp_fft.irfft(spec * self.hrtf.spectra(nfft)[idx, e], nfft)
            acc = np.zeros((m + 1) * hop + nfft)
            for j in range(-(-nfft // hop)):  # overlap-add: frame k's j-th hop lands at (k + j) * hop
                part = seg[:, j * hop:(j + 1) * hop]
                acc[j * hop:j * hop + m * hop] += np.pad(part, ((0, 0), (0, hop - part.shape[1]))).ravel()
            out[e] = acc[hop + self.hrtf.lead:hop + self.hrtf.lead + n]
        return out

    def diffuse(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """x as a diffuse sound; binaurally, its decorrelation draws from a child of rng, so the caller's own stream
        (and everything else in the layer) stays as it is in mono."""
        return diffuse(x, rng.spawn(1)[0], self.fs) if self.binaural else x
