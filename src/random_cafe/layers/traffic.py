"""Traffic: a street with one or two lanes (the second running the other way), simulated, then heard.

Flow: each lane is a 600 m stretch centred on the microphone. Vehicles arrive at its start (a Poisson stream whose
rate drifts slowly, so platoons and quiet spells form) and follow one another by the Intelligent Driver Model
(Treiber, Hennecke & Helbing 2000): each driver has a desired speed, a time headway, and an aggressiveness that sets
how hard they accelerate and brake and how late they change up. With lights, a pedestrian crossing's stop line is
a standing obstacle during amber and red, except for drivers too close to stop comfortably, who carry on.

Sound: each vehicle's speed and acceleration drive its engine through a gearbox. Cars are petrol or diesel
automatics (idling at 650-850 rpm, a torque converter slipping as they pull away); prime movers (trucks, buses) are
manual diesels, whose clutch interrupts the pull at every upshift, and which may dump air from their brakes as they
stop. The engine is its cycle's harmonics (the firing order strongest, the other half-orders much weaker, as in a
balanced engine; brighter under load) at a level that grows with rpm and load, plus exhaust flow noise pulsing at the
firing rate, both through a silencer (heuristic: a fourth-order low-pass a little above the firing rate's range, so
the rasp of a straight pipe is gone; the intake's airbox is lumped in), plus the block's mechanical noise (valve
gear, injection, combustion; most of what makes an engine audible) radiated through the bonnet at 300-1500 Hz,
growing as rpm^2 and with load, pulsing at the firing rate (more so for diesels' clatter). Tyre noise is pink, from
300 Hz to a corner that rises with speed, and grows steeply with speed. All of a vehicle's sources move together, with Doppler and
distance from the geometry. Vehicles further than HEARD_M are left to a distant road bed. Indoors the whole stem
comes in through the facade: windows, and the front door when it is open."""
import numpy as np
from scipy import signal

from .. import spatial
from ..scene import Scene
from .noise import coloured_noise, ou

DT = 0.1  # simulation step, s
ROAD_M = 300.0  # each lane runs from 300 m before the microphone to 300 m past it
HEARD_M = 250.0  # vehicles nearer than this are heard one by one (faded in over FADE_M)
FADE_M = 40.0
LANE_W = 3.5
AMBER_S = 3.0
STOP_DECEL = 4.0  # m/s^2: at amber or red, a driver who would have to brake harder than this carries on
TABLE = 2048  # engine wavetable length
ENGINE, EXHAUST, TYRE, HISS = 0.5, 0.3, 1.0, 2.0  # heuristic levels at 1 m against a car's tyres at 15 m/s
G, CRR, RHO = 9.81, 0.012, 1.2  # gravity, rolling resistance, air

KINDS = {
    # gears: engine rpm per m/s of road speed; shift: upshift rpm for the gentlest and the most aggressive driver;
    # rolloff: of the harmonics (smaller is brighter, as diesels are); muffler: the silencer's corner, Hz; mech: the
    # block's noise through the bonnet; clatter: how much of it pulses with the firing (heuristic values)
    "petrol": dict(cyl=4, idle=(650.0, 800.0), gears=(450, 260, 170, 125, 100, 80), torque=220.0, mass=1400.0,
                   cda=0.65, shift=(1900.0, 4000.0), manual=False, length=4.5, level=1.0, rolloff=1.2, muffler=220.0,
                   mech=0.16, clatter=0.3),
    "diesel": dict(cyl=4, idle=(700.0, 850.0), gears=(420, 240, 155, 115, 90, 72), torque=320.0, mass=1600.0,
                   cda=0.7, shift=(1600.0, 2800.0), manual=False, length=4.6, level=1.3, rolloff=0.9, muffler=180.0,
                   mech=0.13, clatter=0.8),
    "heavy": dict(cyl=6, idle=(550.0, 700.0), gears=(760, 560, 410, 300, 220, 165, 125, 95), torque=1800.0,
                  mass=14000.0, cda=6.0, shift=(1500.0, 1900.0), manual=True, length=12.0, level=3.0, rolloff=0.7,
                  muffler=150.0, mech=0.15, clatter=0.7),
}


def driver(rng: np.random.Generator, cfg) -> dict:
    """A vehicle and its driver; aggressiveness (0-1) raises acceleration, braking and shift points and shortens
    the gap kept to the vehicle ahead."""
    u_kind, u_fuel = rng.uniform(size=2)
    kind = "heavy" if u_kind < cfg.p_heavy else "diesel" if u_fuel < cfg.p_diesel else "petrol"
    heavy, aggr = kind == "heavy", rng.uniform()
    return dict(kind=kind, aggr=aggr, length=KINDS[kind]["length"],
                v0=rng.uniform(*cfg.speed_kmh) / 3.6 * (0.9 + 0.2 * aggr) * (0.85 if heavy else 1.0),
                T=max(0.8, rng.normal(*cfg.headway_s)) * (1.15 - 0.3 * aggr) + (0.5 if heavy else 0.0),
                a=(0.5 + 0.4 * aggr) if heavy else (1.0 + 1.5 * aggr),
                b=(1.2 + 0.6 * aggr) if heavy else (1.5 + 1.5 * aggr),
                s0=2.0 + (1 - aggr) + (1.0 if heavy else 0.0))


def idm(v, v0, T, a, b, s0, gap, dv):
    """Intelligent Driver Model acceleration for speed v behind a gap closing at dv."""
    s_star = s0 + np.maximum(v * T + v * dv / (2 * np.sqrt(a * b)), 0.0)
    return a * (1 - (v / v0) ** 4 - (s_star / np.maximum(gap, 0.1)) ** 2)


def simulate(rng: np.random.Generator, scene: Scene):
    """Every vehicle on the street from a warm-up before the clip to its end. Returns the step times and one dict
    per vehicle: its driver, on (on the road), x (world, along the street; off the road, waiting at the lane's
    ends), y (its lane), speed and acceleration."""
    cfg = scene.traffic
    warm = min(2 * ROAD_M / (cfg.speed_kmh[0] / 3.6), 90.0)
    t = np.arange(-warm, scene.duration_s + DT, DT)
    d = rng.uniform(*cfg.lane_distance_m)
    crossing = rng.uniform(*cfg.crossing_m)
    cycle = cfg.green_s + AMBER_S + cfg.red_s
    stopping = cfg.lights & ((t + rng.uniform(0, cycle)) % cycle >= cfg.green_s)  # amber or red
    out = []
    for lane in range(cfg.lanes):
        sign = 1.0 if lane == 0 else -1.0  # the second lane runs the other way
        dens = np.exp(cfg.density_depth * ou(rng, t.size, 1 / DT, cfg.density_time_s) - cfg.density_depth ** 2 / 2)
        rate = cfg.vehicles_per_min / cfg.lanes / 60
        k = rng.poisson(rate * dens.max() * (t[-1] - t[0]))
        arrive = np.sort(rng.uniform(t[0], t[-1], k))
        arrive = arrive[rng.uniform(size=k) < np.interp(arrive, t, dens) / dens.max()]
        cars = [driver(rng, cfg) for _ in arrive]
        p = {key: np.array([c[key] for c in cars], float) for key in ("v0", "T", "a", "b", "s0", "length")}
        m = len(cars)
        s, v = np.zeros(m), np.zeros(m)
        S, V, A = np.full((m, t.size), np.nan), np.zeros((m, t.size)), np.zeros((m, t.size))
        line = sign * crossing + ROAD_M - 3.0  # the stop line, in this lane's own coordinate
        front = nxt = 0  # vehicles front..nxt-1 are on the road, in order
        for j, tj in enumerate(t):
            if nxt < m and tj >= arrive[nxt]:  # enter once there is room behind the last vehicle
                free = nxt == front
                if free or s[nxt - 1] - p["length"][nxt - 1] >= p["s0"][nxt] + v[nxt - 1] * p["T"][nxt]:
                    s[nxt], v[nxt] = 0.0, p["v0"][nxt] if free else min(p["v0"][nxt], v[nxt - 1])
                    nxt += 1
            if front == nxt:
                continue
            i = np.arange(front, nxt)
            gap, dv = np.full(i.size, np.inf), np.zeros(i.size)
            gap[1:] = s[i[:-1]] - s[i[1:]] - p["length"][i[:-1]]
            dv[1:] = v[i[1:]] - v[i[:-1]]
            par = [p[key][i] for key in ("v0", "T", "a", "b", "s0")]
            acc = idm(v[i], *par, gap, dv)
            if stopping[j]:
                dist = line - s[i]
                stop = (dist > 0) & (v[i] ** 2 / (2 * np.maximum(dist, 0.1)) < STOP_DECEL)
                acc = np.where(stop, np.minimum(acc, idm(v[i], *par, dist, v[i])), acc)
            acc = np.maximum(acc, -9.0)
            vn = np.maximum(v[i] + acc * DT, 0.0)
            s[i] += (v[i] + vn) / 2 * DT
            v[i] = vn
            S[i, j], V[i, j], A[i, j] = s[i], vn, acc
            while front < nxt and s[front] > 2 * ROAD_M:
                front += 1
        on = ~np.isnan(S)
        S = np.where(on, S, np.where(np.cumsum(on, axis=1) == 0, 0.0, 2 * ROAD_M))
        for c, o, x, sp, ac in zip(cars, on, sign * (S - ROAD_M), V, A):
            out.append(dict(c, on=o, x=x, y=d + lane * LANE_W, v=sp, acc=ac))
    return t, out


def drive(rng: np.random.Generator, car: dict, v: np.ndarray, acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Engine rpm and load (0-1) per step from road speed and acceleration, through the gearbox. Load is the
    tractive force needed (inertia, rolling, drag) over what the engine gives in that gear."""
    k = KINDS[car["kind"]]
    gears, aggr = np.array(k["gears"], float), car["aggr"]
    idle = rng.uniform(*k["idle"])
    down, launch = idle + 350.0, 600.0 if k["manual"] else 1500.0 + 1000.0 * aggr
    ok = np.flatnonzero(v[0] * gears >= down)
    g = int(ok.max()) if ok.size else 0  # start in the highest gear that keeps the engine above the downshift
    rpm, load = np.empty(v.size), np.empty(v.size)
    clutch = 0
    for j in range(v.size):
        force = k["mass"] * (acc[j] + G * CRR) + 0.5 * RHO * k["cda"] * v[j] ** 2
        ld = float(np.clip(force / (k["torque"] * gears[g] * 2 * np.pi / 60), 0.05, 1.0))
        up = k["shift"][0] + (k["shift"][1] - k["shift"][0]) * aggr * (0.4 + 0.6 * ld)
        if g < gears.size - 1 and v[j] * gears[g] > up:
            g += 1
            clutch = int(0.7 / DT) if k["manual"] else 0  # the pull stops while the clutch is out
        elif g > 0 and v[j] * gears[g] < down:
            g -= 1
        r = max(idle, v[j] * gears[g])
        if g == 0 and acc[j] > 0.05:  # pulling away: torque converter (or clutch) slip
            r = max(r, idle + ld * launch * float(np.clip(1 - v[j] / 5.0, 0.0, 1.0)))
        if clutch:
            ld, clutch = 0.05, clutch - 1
        rpm[j], load[j] = r, ld
    smooth = lambda x, tau: signal.lfilter([1 - np.exp(-DT / tau)], [1, -np.exp(-DT / tau)], x, zi=[x[0] * np.exp(-DT / tau)])[0]  # noqa: E731
    return smooth(rpm, 0.25), smooth(load, 0.15)


def wavetable(rng: np.random.Generator, cyl: int, rolloff: float) -> np.ndarray:
    """One engine cycle (two revolutions) of half-order harmonics: the firing order and its multiples strongest, the
    rest small (cylinder-to-cylinder differences)."""
    k = np.arange(1, 6 * cyl + 1)
    amp = np.where(k % cyl == 0, 1.0, rng.uniform(0.03, 0.1, k.size)) * (k / cyl) ** -rolloff
    th = 2 * np.pi * np.arange(TABLE) / TABLE
    y = (amp[:, None] * np.sin(k[:, None] * th + rng.uniform(0, 2 * np.pi, k.size)[:, None])).sum(0)
    return y / np.std(y)


def sound(rng: np.random.Generator, car: dict, te: np.ndarray, t: np.ndarray, rpm, load, v, fade, banks, fs) -> np.ndarray:
    """A vehicle's sound at the source (1 m), sampled at emission times te. Levels change slowly, so they are worked
    out on the simulation's steps and interpolated; the waveforms are table lookups and noise banks."""
    k = KINDS[car["kind"]]
    lv, r1k = k["level"] * fade, rpm / 1000
    engine = ENGINE * lv * r1k ** 1.5 * (0.25 + 0.75 * load)
    gains = (engine * (1 - load), engine * load, EXHAUST * lv * load * r1k, k["mech"] * lv * r1k ** 2 * (0.5 + 0.5 * load),
             TYRE * (v / 15.0) ** 2 * (1.5 if car["kind"] == "heavy" else 1.0) * fade)
    g_soft, g_hard, g_exh, g_mech, g_tyre = (np.interp(te, t, g) for g in gains)
    cycles = np.cumsum(np.interp(te, t, rpm) / (120.0 * fs))  # engine cycles (one every two revolutions)
    pos = ((cycles % 1.0) * TABLE).astype(np.intp)
    soft, hard = wavetable(rng, k["cyl"], k["rolloff"] + 0.5), wavetable(rng, k["cyl"], k["rolloff"])
    pulse = ((0.5 + 0.5 * np.cos(k["cyl"] * 2 * np.pi * np.arange(TABLE) / TABLE)) ** 4 / 0.273)[pos]  # firing, mean 1
    m = te.size
    seg = lambda bank: bank[(o := int(rng.integers(0, bank.size - m))):o + m]  # noqa: E731
    corner = min(1000.0 + 60.0 * v[fade > 0].max(), 0.45 * fs)  # faster: brighter
    tyre = signal.sosfilt(signal.butter(2, [300.0, corner], "bandpass", fs=fs, output="sos"), seg(banks["tyre"]))
    x = g_soft * soft[pos] + g_hard * hard[pos] + g_exh * seg(banks["exhaust"]) * (0.5 + 0.5 * pulse)
    x = signal.sosfilt(signal.butter(4, k["muffler"], "lowpass", fs=fs, output="sos"), x)  # the silencer
    x += g_mech * seg(banks["mech"]) * (1 - k["clatter"] + k["clatter"] * pulse) + g_tyre * tyre / (np.std(tyre) + 1e-12)
    if car["kind"] == "heavy":  # air dumped from the brakes as it comes to a stop, sometimes
        hiss = seg(banks["hiss"])
        for j in np.flatnonzero((v[:-1] >= 0.3) & (v[1:] < 0.3)):
            if rng.uniform() < 0.6:
                start = int((t[j] + rng.uniform(0.3, 1.0) - te[0]) * fs)
                n_h = int(rng.uniform(0.6, 1.5) * fs)
                if 0 <= start < m - n_h:
                    x[start:start + n_h] += HISS * np.interp(te[start], t, lv) * hiss[start:start + n_h] * np.exp(-np.arange(n_h) / (0.4 * fs))
    return x


def render(rng: np.random.Generator, scene: Scene, facade: spatial.Facade, rx: spatial.Receiver) -> np.ndarray:
    cfg, fs = scene.traffic, scene.fs
    n = int(scene.duration_s * fs)
    outside = not scene.room.indoor  # indoors, everything arrives through the facade's opening
    out = rx.zeros() if outside else np.zeros(n)
    t, cars = simulate(rng, scene)
    t0 = -(HEARD_M + 10.0) / spatial.C  # the earliest emission heard in the clip
    te = t0 + np.arange(n + int(np.ceil(-t0 * fs))) / fs
    w = t >= t0 - 3.0  # the engine model settles over a few seconds before that
    nb = te.size + 4 * fs  # noise banks: each vehicle reads its own stretch
    bp = lambda lo, hi, x: signal.sosfilt(signal.butter(2, [lo, hi], "bandpass", fs=fs, output="sos"), x)  # noqa: E731
    banks = {"tyre": coloured_noise(rng, nb, 3.0, fs), "mech": bp(300.0, 1500.0, rng.standard_normal(nb)),
             "exhaust": signal.sosfilt(signal.butter(2, 400.0, "lowpass", fs=fs, output="sos"), coloured_noise(rng, nb, 3.0, fs)),
             "hiss": bp(1500.0, min(6000.0, 0.45 * fs), rng.standard_normal(nb))}
    for key in ("mech", "exhaust", "hiss"):
        banks[key] /= np.std(banks[key])
    for car in cars:
        r = np.hypot(car["x"], np.hypot(car["y"], scene.mic[2] - 0.5))
        fade = np.clip((HEARD_M - r) / FADE_M, 0.0, 1.0) * car["on"]
        if not fade[w].any():
            continue
        rpm, load = drive(rng, car, car["v"][w], car["acc"][w])
        live = np.flatnonzero(fade[w])  # only while it is within earshot
        a = max(int((t[w][live[0]] - DT - t0) * fs), 0)
        b = min(int(np.ceil((t[w][live[-1]] + DT - t0) * fs)), te.size)
        x = sound(rng, car, te[a:b], t[w], rpm, load, car["v"][w], fade[w], banks, fs)
        span = (max(int(te[a] * fs), 0), min(int(np.ceil((te[b - 1] + (HEARD_M + 10.0) / spatial.C) * fs)) + 1, n))

        def path(tt, tx=t, xx=car["x"], y=car["y"]):
            return np.stack([np.interp(tt, tx, xx), np.full_like(tt, y), np.full_like(tt, 0.5)], axis=1)

        if outside:
            out += rx.place_moving(x, path, t0=te[a], span=span)
        else:
            out += spatial.place_moving(x, path, scene.mic, fs, t0=te[a], n=n, span=span)
    bed = coloured_noise(rng, n, 6.0, fs)
    env = np.interp(np.arange(n), np.linspace(0, n, 8), 1 + 0.3 * rng.standard_normal(8))
    bed = bed * np.clip(env, 0.3, None) * np.std(out + 1e-9) * 10 ** (cfg.bed_db / 20)
    out += rx.diffuse(bed, rng) if outside else bed
    if scene.room.indoor:
        out = facade(out, rng)
    return out
