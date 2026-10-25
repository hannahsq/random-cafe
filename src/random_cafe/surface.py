"""Rain-struck surfaces: a drop rings what it lands on. After umbrella-rain's canopy model: the drop is an impulse
J = m v delivered over a contact time, so the surface feels that pulse's spectrum; the surface answers with its own
modes, individually where they are still apart and as a decaying noise continuum where they overlap, each ringing as
long as the surface's damping lets it; and a listener on the far side (under an umbrella, behind a window) hears the
drop's own splash through the surface, by the mass law.

Two kinds of surface. A membrane (fabric under tension: umbrellas, awnings, tarps) carries waves at sqrt(T / sigma)
whatever the frequency, so its modes spread evenly in frequency; it loses energy at its ribs (umbrella-rain's
rib-terminated damping, copied: rib_split, membrane_q). A plate (glass, sheet metal, polycarbonate) is stiff instead:
its waves get faster with frequency, its modes are spread evenly in frequency squared... i.e. a constant number per Hz,
it loses energy in its mounting (gaskets, frame) more than in itself, and it radiates well only above its coincidence
frequency, where its bending waves outrun sound (heuristic radiation efficiency below).

Prototype: not yet used by the rain layer."""
from dataclasses import dataclass

import numpy as np
from scipy import signal

RHO_AIR, C, RHO_W = 1.2, 343.0, 1000.0
ETA_MAX = 0.5  # umbrella-rain's numerical guard: Q below 2 is no ring at all


@dataclass
class Surface:
    name: str
    kind: str  # "membrane" or "plate"
    lx: float  # one panel (a membrane between ribs, a plate in its frame), m
    ly: float
    sigma: float  # areal density, kg/m^2
    tension: float = 0.0  # membrane, N/m
    bending: float = 0.0  # plate, D = E h^3 / (12 (1 - nu^2)), N m
    loss: float = 0.02  # plate: total loss factor (material + mounting)
    rib_impedance: float = 1.4  # membrane (umbrella-rain): rib termination at the reference tension and frequency
    rib_ref_tension: float = 180.0
    rib_ref_hz: float = 100.0
    rib_dissipation: float = 0.45  # share of what crosses a rib that is lost there
    contact_ms: float = 0.0  # extra contact time a soft surface adds (umbrella-rain's fabric compliance)
    top_hz: float = 8000.0  # top of the continuum
    extra_loss: float = 0.0  # membrane: the fabric's own loss factor, on top of its ribs'
    t60_s: float = 0.0  # plate: if set, one decay time at every frequency (losses into the mounting and air)
    modes_hz: tuple = ()  # measured modes standing out of the continuum (plates that are not flat panels)
    ring_db: float = -3.0  # the ring's energy re the drop's own tick (through the surface; a ground tick for plates)
    flank_share: float = 0.0  # corrugated: the share of drops landing on a flank, and its tilt
    flank_deg: float = 0.0
    p_relanding: float = 0.0  # chance a flank drop's water lands again in the valley, 2-4.5 ms later
    p_splash_relanding: float = 0.0  # chance a flat drop's splash lands again on a flank


def plate(name, lx, ly, thickness_m, youngs_pa, density, poisson=0.25, loss=0.02):
    return Surface(name, "plate", lx, ly, density * thickness_m,
                   bending=youngs_pa * thickness_m ** 3 / (12 * (1 - poisson ** 2)), loss=loss)


# the pedestrian umbrella is umbrella-rain's (an 8-panel canopy of 0.7 m radius, 70 g/m^2 at 180 N/m); the cafe
# umbrella is a canvas one 2.5 m across on heavier ribs (heuristic: 300 g/m^2 at 400 N/m, ribs scaled to keep the
# same mismatch); the window a 1.2 x 1.0 m pane of 5 mm float glass in rubber gaskets (loss factor ~0.02: glass alone
# is ~0.002, the mounting dominates)
PEDESTRIAN_UMBRELLA = Surface("pedestrian umbrella", "membrane", 0.7, np.pi * 0.7 / 8, 0.07, tension=180.0,
                              contact_ms=0.3)
CAFE_UMBRELLA = Surface("cafe umbrella", "membrane", 1.25, np.pi * 1.25 / 8, 0.3, tension=400.0,
                        rib_impedance=1.4 * np.sqrt((400.0 * 0.3) / (180.0 * 0.07)), contact_ms=0.4)
WINDOW = plate("window pane", 1.2, 1.0, 0.005, 70e9, 2500.0, poisson=0.22, loss=0.02)
WINDOW.ring_db = 0.0  # heuristic: a 2 mm drop's ring as loud as its tick on the ground
# fitted to a recording (Hannah's, under an ultra-light folding umbrella, eval only): the fabric pushes out and back
# once in ~1.4 ms (energy peaking at ~470 Hz), falls 20 dB within 3 ms, and leaves no ring above the street; slack,
# flimsy fabric (tension and areal density heuristic), contact time and its own loss fitted
FOLDING_UMBRELLA = Surface("folding umbrella", "membrane", 0.5, np.pi * 0.5 / 8, 0.05, tension=40.0,
                           contact_ms=1.3, extra_loss=0.5, ring_db=-6.0)
# corrugated polycarbonate roofing (0.8 mm, one bay between purlins), fitted to recordings (Hannah's, eval only;
# 5-10 cm, ~1 m and 1.5-2 m off): modes at the frequencies standing out of its ring, which fades at about one rate at
# every frequency (T60 ~0.1 s). Its trapezoid profile /-\_/-\_ with 45 degree flanks (~0.4 of its plan area if all
# faces are equally wide): a drop on a flank glances (only the impulse's normal part, over a longer contact; the
# first of a recorded pair is ~4 dB duller than a lone tick) and its water lands again in the valley 2-4.5 ms later.
# Recorded, 40-70 % of ticks have a broadband follower 2-6 ms later (independent drops would give ~6-13 %), about as
# loud as the first from 1-2 m off (closer, whichever lands nearer the mic dominates); not echoes (the pair's
# waveforms are no more alike than unrelated ticks') nor the ring's modes beating (the follower has the first's highs).
# Flat hits' splashes landing again on a flank (heuristic) fill the rest of that rate.
CORRUGATED_POLYCARBONATE = plate("corrugated polycarbonate", 0.8, 0.76, 0.0008, 2.4e9, 1200.0, poisson=0.37)
CORRUGATED_POLYCARBONATE.t60_s = 0.1
CORRUGATED_POLYCARBONATE.modes_hz = (220.0, 420.0, 670.0, 940.0, 1190.0, 1690.0, 2020.0, 2190.0)
CORRUGATED_POLYCARBONATE.ring_db = -6.0
CORRUGATED_POLYCARBONATE.flank_share, CORRUGATED_POLYCARBONATE.flank_deg = 0.4, 45.0
CORRUGATED_POLYCARBONATE.p_relanding, CORRUGATED_POLYCARBONATE.p_splash_relanding = 1.0, 0.4


def terminal_velocity(d_mm):
    """Atlas et al. (1973) fit, m/s."""
    return np.maximum(9.65 - 10.3 * np.exp(-0.6 * np.asarray(d_mm, float)), 0.1)


def hann_spectrum(x):
    """|FT| of a unit-area Hann pulse at (frequency x duration), 1 at DC (umbrella-rain)."""
    x = np.asarray(x, dtype=float)
    den = 1 - x * x
    ok = np.abs(den) > 1e-6
    return np.abs(np.where(ok, np.sinc(x) / np.where(ok, den, 1), 0.5))


def wave_speed(s: Surface, f):
    """Phase speed of the surface's waves at f: constant for a membrane, sqrt(omega) for a plate."""
    if s.kind == "membrane":
        return np.full(np.shape(f), np.sqrt(s.tension / s.sigma))
    return np.sqrt(2 * np.pi * np.asarray(f, float)) * (s.bending / s.sigma) ** 0.25


def rib_split(s: Surface, wavenumber):
    """umbrella-rain: the energy fraction leaving a membrane panel at a rib, split into what the rib dissipates and
    what it passes on (a beam's impedance rises as sqrt(omega), so higher modes lose more there)."""
    z_panel = np.sqrt(s.tension * s.sigma)
    f = np.sqrt(s.tension / s.sigma) * wavenumber / (2 * np.pi)
    z_rib = s.rib_impedance * (s.tension / s.rib_ref_tension) ** 0.25 * np.sqrt(f / s.rib_ref_hz)
    x = z_rib / z_panel
    tau = 4 * x / (1 + x) ** 2
    return tau * s.rib_dissipation, tau * (1 - s.rib_dissipation)


def quality(s: Surface, f):
    """Q at frequency f: a plate's from its loss factor; a membrane's from how many chords fit across the panel before
    its ribs have taken its energy (umbrella-rain's mode_q, dry)."""
    f = np.asarray(f, float)
    if s.kind == "plate":
        return np.pi * f * s.t60_s / np.log(1000) if s.t60_s else np.full(f.shape, 1 / s.loss)
    k = 2 * np.pi * f / wave_speed(s, f)
    chord = np.pi * s.lx * s.ly / (2 * (s.lx + s.ly))
    tau_d, _ = rib_split(s, k)
    return 1 / np.minimum(tau_d / (chord * k) + s.extra_loss, ETA_MAX)


def modal_density(s: Surface, f):
    """Modes per Hz of one panel: 2 pi f A / c^2 for a membrane, (A / 2) sqrt(sigma / D) for a plate."""
    a = s.lx * s.ly
    if s.kind == "membrane":
        return 2 * np.pi * np.asarray(f, float) * a / (s.tension / s.sigma)
    return np.full(np.shape(f), a / 2 * np.sqrt(s.sigma / s.bending))


def crossover(s: Surface) -> float:
    """Where the modal overlap (modes per Hz x half-power bandwidth) reaches 1: modes above it are a continuum."""
    f = np.geomspace(10.0, 20000.0, 2000)
    overlap = modal_density(s, f) * f / quality(s, f)
    return float(f[np.argmax(overlap >= 1.0)]) if (overlap >= 1.0).any() else 20000.0


def coincidence(s: Surface) -> float:
    """Where a plate's bending waves reach the speed of sound (a membrane's never do)."""
    return C ** 2 / (2 * np.pi) * np.sqrt(s.sigma / s.bending) if s.kind == "plate" else np.inf


def radiation(s: Surface, f):
    """Radiated pressure per unit surface velocity, relative. A membrane is subsonic everywhere and radiates only
    evanescently, the more so the lower (umbrella-rain: as f). A plate radiates fully above coincidence and weakly
    below it, from its edges and corners (heuristic: efficiency (f / f_c)^2 below, 1 above, as amplitude)."""
    f = np.asarray(f, float)
    if s.kind == "membrane":
        return f / 1000.0
    return (1 + (coincidence(s) / np.maximum(f, 1.0)) ** 4) ** -0.25


def contact(s: Surface, d_mm: float, stretch: float = 1.0) -> tuple[float, float]:
    """(impulse J in N s, contact time in s): a rigid surface takes the drop's own d / v (times stretch: a glancing
    hit lasts longer); a membrane yields, which lengthens it (umbrella-rain: its compliance and the membrane's own
    give)."""
    v = float(terminal_velocity(d_mm))
    j = RHO_W * np.pi * (d_mm * 1e-3) ** 3 / 6 * v
    tau = d_mm * 1e-3 / v * stretch
    if s.kind == "membrane":
        tau = float(np.hypot(tau, np.hypot(s.contact_ms * 1e-3, np.sqrt((j / v) / (np.pi * s.tension)))))
    return j, tau


def body(rng: np.random.Generator, s: Surface, d_mm: float, fs: int, max_modes: int = 40,
         scale: float = 1.0, near: bool = False, stretch: float = 1.0) -> np.ndarray:
    """The surface's ring after one drop at a random point, as radiated (relative units, consistent across sizes and
    surfaces of one kind). Each mode is a decaying sine of amplitude drive x mode shape at the hit x radiation over the
    panel's mass, energy A^2 T / 4; the continuum above the crossover is decaying band noise carrying the energy the
    modes in its band would (modes per Hz x bandwidth x <A^2> x T / 4), so the two meet without hand-matching."""
    j, tau = contact(s, d_mm, stretch)
    j *= scale
    mass = s.sigma * s.lx * s.ly
    f_x = min(crossover(s), 0.45 * fs)
    u, v = rng.uniform(0.05, 0.95, 2)
    dur = 4 * float(np.max(quality(s, np.array([20.0, f_x, 1000.0])) / (np.pi * np.array([20.0, f_x, 1000.0])))) + 0.05
    t = np.arange(int(min(dur, 3.0) * fs)) / fs
    y = np.zeros(t.size)
    # close behind the surface (within ~a bending wavelength) its near field is heard whole: no radiation loss
    drive = lambda f: j * hann_spectrum(f * tau) * (1.0 if near else radiation(s, f)) / mass  # noqa: E731
    # the modes below the crossover
    m, n = np.meshgrid(np.arange(1, 60), np.arange(1, 60))
    if s.kind == "membrane":
        f_mn = 0.5 * wave_speed(s, 0.0)[()] * np.hypot(m / s.lx, n / s.ly)
    else:
        f_mn = np.pi / 2 * np.sqrt(s.bending / s.sigma) * ((m / s.lx) ** 2 + (n / s.ly) ** 2)
    keep = np.argsort(f_mn, axis=None)[:max_modes]
    for mi, ni, f in zip(m.flat[keep], n.flat[keep], f_mn.flat[keep]):
        if f >= f_x:
            break
        a = drive(f) * np.sin(mi * np.pi * u) * np.sin(ni * np.pi * v)
        decay = quality(s, f) / (np.pi * f)
        y += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t / decay)
    # measured modes: each carrying the continuum's energy within a third of an octave of it (heuristic: they stand
    # a few dB out of it, as recorded)
    for f in s.modes_hz:
        bw = f * (2 ** (1 / 6) - 2 ** (-1 / 6))
        a = float(drive(f)) * np.sqrt(float(modal_density(s, f)) * bw) / 2 * rng.uniform(0.5, 1.5)
        decay = float(quality(s, f)) / (np.pi * f)
        y += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t / decay)
    # the continuum above it
    top = min(s.top_hz, 0.45 * fs)
    if f_x < top:
        edges = np.geomspace(f_x, top, 9)
        for lo, hi in zip(edges[:-1], edges[1:]):
            fc = np.sqrt(lo * hi)
            decay = float(quality(s, fc)) / (np.pi * fc)
            energy = float(modal_density(s, fc)) * (hi - lo) * float(drive(fc)) ** 2 / 4 * decay / 4
            noise = signal.sosfilt(signal.butter(2, [lo, hi], "bandpass", fs=fs, output="sos"), rng.standard_normal(t.size))
            k = noise * np.exp(-t / decay)
            y += k * np.sqrt(energy / (np.sum(k ** 2) / fs + 1e-30))
    return y


def through(s: Surface, x: np.ndarray, fs: int) -> np.ndarray:
    """A sound heard through the surface (the drop's own splash, from the other side): a limp sheet passes the lows
    and loses the highs above 2 rho c / (2 pi sigma) (umbrella-rain's fabric_transmission; for glass that corner is
    ~10 Hz, so a splash outside barely gets in)."""
    fc = 2 * RHO_AIR * C / (2 * np.pi * s.sigma)
    b, a = signal.butter(1, min(fc / (fs / 2), 0.99), "low")
    return signal.lfilter(b, a, x)


def _energy(x):
    return float(np.sum(x ** 2))


_GAINS = {}


def ring_gain(s: Surface, fs: int, near: bool = False) -> float:
    """The body's gain giving s.ring_db: a 2 mm drop's mean ring energy against its own tick heard through the
    surface (membranes) or on the ground (plates, heard from behind, where the tick barely gets through). Set close
    behind the surface (near), where the recordings were made; heard further off the ring keeps that gain."""
    from .layers.rain import grain
    key = (repr(s), fs)
    if key not in _GAINS:
        rng = np.random.default_rng(0)
        ring = np.mean([_energy(body(rng, s, 2.0, fs, near=True)) for _ in range(20)])
        ref = np.mean([_energy(grain(rng, 2.0, fs)) for _ in range(20)])
        _GAINS[key] = float(np.sqrt(ref / ring) * 10 ** (s.ring_db / 20))
    return _GAINS[key]


def strike(rng: np.random.Generator, s: Surface, d_mm: float, fs: int, near: bool = False) -> np.ndarray:
    """One drop as heard beneath or behind the surface: its ring plus its own tick. Close behind (near: within
    tens of cm, as recorded) the surface's motion under the drop carries the tick whole and its near field is heard
    without radiation loss; further off the tick comes through the surface by the mass law. On a
    corrugated sheet a flank takes the impulse's normal part (cos of its tilt) over a contact as much longer
    (heuristic), and the water lands again 2-4.5 ms later, hitting about as hard as the glance (as recorded from 1-2 m
    off); a flat hit's splash may too."""
    from .layers.rain import grain
    flank = rng.random() < s.flank_share
    c = np.cos(np.radians(s.flank_deg)) if flank else 1.0

    def hit(scale, stretch):
        tick = grain(rng, d_mm, fs)
        if stretch > 1.0:  # the drop's own tick, duller over the longer contact (heuristic corner 3 / contact time)
            corner = 3 / (d_mm * 1e-3 / float(terminal_velocity(d_mm)) * stretch)
            tick = signal.lfilter(*signal.butter(1, min(corner / (fs / 2), 0.99)), tick)
        return (ring_gain(s, fs) * body(rng, s, d_mm, fs, scale=scale, near=near, stretch=stretch),
                scale * (tick if near else through(s, tick, fs)))

    parts = [(0, p) for p in hit(c, 1 / c)]
    if rng.random() < (s.p_relanding if flank else s.p_splash_relanding):
        o = int(rng.uniform(0.002, 0.0045) * fs)
        parts += [(o, p) for p in hit(np.cos(np.radians(45.0)), 1.0)]
    y = np.zeros(max(o + p.size for o, p in parts))
    for o, p in parts:
        y[o:o + p.size] += p
    return y
