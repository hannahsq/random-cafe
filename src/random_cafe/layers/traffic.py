"""Traffic: a street with one or two lanes (the second running the other way), simulated, then heard.

Flow: each lane is a 600 m stretch centred on the microphone. Vehicles arrive at its start (a Poisson stream whose
rate drifts slowly, so platoons and quiet spells form) and follow one another by the Intelligent Driver Model
(Treiber, Hennecke & Helbing 2000): each driver has a desired speed, a time headway, and an aggressiveness that sets
how hard they accelerate and brake and how late they change up. With lights, a pedestrian crossing's stop line is
a standing obstacle during amber and red, except for drivers too close to stop comfortably, who carry on.

Sound: each vehicle's speed and acceleration drive its engine through a gearbox. Cars are petrol or diesel
automatics (idling at 650-850 rpm, a torque converter slipping as they pull away); prime movers (trucks, buses) are
manual diesels, double-clutched: the pull stops for about 1.3 s at every shift while the engine falls (or is
blipped) to the next gear's speed, and which may dump air from their brakes as they
stop. Each vehicle draws its engine: a layout (I3, I4, boxer 4, I6, V6, odd-fire V6, flat- or cross-plane V8; small
ones mostly petrol, large ones mostly diesel), a turbocharger or not, and a stock or high-flow exhaust.

The exhaust is a train of blowdown pulses at the layout's firing angles, each bank's through its own pipe, so a
cross-plane V8 or a boxer with unequal headers keeps its uneven rhythm (burble, rumble) while an even-firing engine
gives mostly its firing order. Petrol pulses are wide and soft at light throttle and sharp at full throttle; a
diesel, unthrottled, pulses alike at every load; a turbine takes energy from the pulses and smears them. With exhaust
flow noise pulsing with them, they pass a silencer (heuristic: a fourth-order low-pass a little above the firing
rate's range, so the rasp of a straight pipe is gone; a high-flow one is second-order, higher and louder). The intake
is its own source: inlet pulses from a common plenum (even, whatever the banks), weak at light throttle in a petrol
engine and further smoothed by a compressor, through a plastic airbox (a resonant low-pass: little damping); petrol
engines hiss past a nearly closed throttle. A turbo, sized to the engine's displacement, adds the compressor:
its blade-pass whistle (with H2 and H3) over a broadband pedestal, as measured on recordings, rising with the
shaft's speed, which builds with the exhaust's flow above a boost threshold, after a lag, and falls at each shift. Then the block's mechanical noise (valve gear, injection, combustion)
radiated through the bonnet at 300-1500 Hz, growing as rpm^2 and with load, much of it in clicks: each cylinder's
combustion and its valves seating, the cylinders and the cycles differing (more so for diesels' knock). Tyre noise is pink, from 300 Hz to a corner that rises with
speed, and grows steeply with speed. All of a vehicle's sources move together, with Doppler and distance from the
geometry. Vehicles further than HEARD_M are left to a distant road bed. Indoors the whole stem
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
HARMONICS = 64  # of the engine cycle kept in its wavetables (64 x 50 Hz at 6000 rpm, well below Nyquist)
# a stock petrol exhaust, fitted to a 2023 BMW X1 1.5 L turbo I3 revving on the spot (1 m from the tailpipe at 45
# degrees; listened to and measured through the recorder's automatic gain, emulated): pulses with a steep front
# (PULSE_RISE of their width) and harmonics to 128 cycle orders, rolling off smoothly above TONE_CEILING, mixed with
# white-ish noise at FLOOR x their level, through the silencer's measured response (SHELF_DB, SHELF_PEAKS); below
# RES_CUT its resonances are driven harder as the throttle opens (revving on the spot, the load is the crank's
# acceleration), and the tones depend less on load than elsewhere (SHELF_TONE_LOAD). Diesels, trucks and high-flow
# exhausts keep the simpler silencer until there are references for them
PULSE_RISE = 0.03
# averaged over rpm so the harmonics fill it in: resonances at ~110, ~205, ~307 (the strongest, 17 dB clear), ~377,
# ~490, ~689 Hz, broad ones at ~1.0 and ~1.2-1.4 kHz, a trough at 240-280 Hz, the fall above 390 Hz; above 480 Hz
# raised to make up for the model's pulses being poorer in high harmonics than the real ones
SHELF_DB = ((0, 0.0), (90, -2.0), (110, 1.0), (130, -3.0), (160, -5.0), (185, -3.0), (205, 2.0), (225, -4.0),
            (250, -11.0), (270, -10.0), (290, -2.0), (307, 8.0), (325, 0.0), (345, -1.0), (377, 4.0), (392, 0.0),
            (410, -7.0), (440, -14.0), (470, -12.0), (490, 1.0), (510, -2.0), (600, -3.0), (650, -3.0), (689, 0.0),
            (730, -3.0), (800, -3.0), (900, -2.0), (1000, 6.0), (1200, 5.0), (1400, 6.5), (1500, 4.0), (1600, 4.0),
            (1800, 3.0), (2000, 4.0), (3000, 2.0), (6000, -4.0), (12000, -10.0), (24000, -16.0))
# and its higher resonances (90th-percentile spectrum, alike idling and revving): (Hz, dB, width re frequency), at
# 0.8 x their measured prominence (which the harmonics also lift)
SHELF_PEAKS = ((1700, 4.4, 0.015), (1795, 1.5, 0.015), (1910, 1.6, 0.015), (2100, 2.1, 0.015), (2395, 3.6, 0.015),
               (2737, 2.5, 0.015), (2880, 2.2, 0.015), (3195, 3.3, 0.015), (3390, 1.8, 0.015), (3800, 1.6, 0.015),
               (3944, 2.5, 0.015), (4330, 3.0, 0.015), (4893, 2.8, 0.015), (6410, 2.6, 0.015), (6775, 1.9, 0.015),
               (7620, 1.6, 0.06), (8720, 2.5, 0.015))
SHELF_TONE_LOAD = (0.7, 0.3)  # the tones' level as a + b x load (elsewhere 0.25 + 0.75 load)
RES_CUT, RES_BOOST = 430.0, 3.5  # below RES_CUT the resonances are driven harder with load: x (1 + RES_BOOST load),
# load capped at 0.5 (fitted revving on the spot, where it stays low; heuristic beyond)
X1_GAIN = 0.065  # its overall level (the fit sets the shape): the stock petrol tones as loud as before it
FLOOR = 0.15  # white-ish noise in the source, 300 Hz-8 kHz, at this ratio to the tones (growing with rpm and load)
TONE_CEILING = 1300.0  # the harmonics roll off smoothly above it (third order), mostly gone by 2 kHz
# every engine: each cylinder's exhaust pulse arrives at its own crank angle (unequal runners: SD 5 degrees) and level
# (log-SD 0.25), and no two firings are alike (log-SD 0.2), so lines appear between the firing harmonics (the X1's
# sit ~12 dB below them); and every vehicle is heard from two places, its exhaust at the back and the rest from the
# engine bay, whose sounds reach a listener by different paths (combing as the revs sweep, as recorded)
CYL_LAG, CYL_SPREAD, FIRING_SPREAD = 5.0, 0.25, 0.2
# heuristic levels at 1 m against a car's tyres at 15 m/s
ENGINE, EXHAUST, INTAKE, THROTTLE, TURBO, TYRE, HISS = 5.0, 0.3, 3.0, 0.05, 1.34, 1.0, 2.0
HIGH_FLOW = 2.0  # heuristic: a high-flow exhaust's pulses against a stock one's
# exhaust flow noise, from recordings (a Hayabusa straight-piped and stock, a Suzuki Cappuccino on a dyno): above
# ~1 kHz an exhaust is broadband noise, made by the flow at the tailpipe and in the silencer itself, so after the
# silencer's attenuation: roughly flat per octave to a corner of a few kHz; each octave above 800 Hz is ~27-31 dB
# below everything under 200 Hz at idle and 0-14 dB below at high flow (revving, or pulling hard). That growth is
# what jet noise does: power as the gas speed^8 (Lighthill); gas speed ~ rpm x (0.5 + 0.5 load), heuristic. A modern
# consumer car's stock exhaust makes ~6 dB less of it than a high-flow one: fitted to a 2023 BMW X1 1.5 L turbo I3
# (1 m from the tailpipe, 45 degrees off axis), whose octaves above 800 Hz sit ~22 dB below everything under 200 Hz at
# idle and 10-14 dB when revving
FLOW_STOCK = 10 ** (-6 / 20)
FLOW_SAT = 3.0  # gas speed (rpm/1000 x (0.5 + 0.5 load)) above which the flow noise grows like the tones; with
# EXHAUST, fitted to the X1 revving on the spot (idle, a hold, blips to 6000 rpm: within ~3 dB in both octaves)
FLOW_FULL = 6.0  # gas speed (rpm/1000 x (0.5 + 0.5 load)) at which the flow noise's corner is at its highest
TURBINE = 1.3  # heuristic: a turbine takes energy from the exhaust pulses (level / TURBINE) and smears them (width x)
COMPRESSOR = 0.4  # heuristic: the intake pulses left once they have passed back through a compressor
# turbo, measured on recordings (a Peterbilt 379, ~14 L, ~350 kW; a stock Suzuki Cappuccino, 0.66 L, 47 kW): the
# compressor's blade-pass line is a tone (Q above ~600, narrower than a spectrogram resolves; here narrowband noise,
# so it wavers a little) with H2 at -28 dB and H3 at -35 dB, over a broadband pedestal with twice its power, flat from
# 0.4 to 1 x the line and falling ~17 dB/octave above (a band-pass of Q ~0.9). A car's small compressor, its full
# blades alternating with splitter blades, also has a line at half the blade-pass (the full blades alone), only ~3 dB
# weaker (median of -2 and -3 dB over two passages; the truck's has none). The wheel's tip speed is what limits it,
# so its shaft speed goes as 1 / diameter, and its diameter as the square root of the air it moves, i.e. of the
# engine's power: at full boost the line is near 9 kHz for the truck and ~26 kHz for the Cappuccino (its half-line
# reaching ~13 kHz), against 24.6 kHz from this scaling. Within each gear the truck's line climbs from ~4 to ~8 kHz
# with the rpm and falls at each shift, a little faster than it climbs.
TURBO_HZ, TURBO_KW = 9000.0, 350.0  # full-boost line: TURBO_HZ x (rated power / TURBO_KW) ^ -0.5
TURBO_IDLE = 0.25  # heuristic: shaft speed with no boost, re full boost (the recording's lowest is ~0.45, mid-shift)
TURBO_Q, TURBO_H = 400.0, (1.0, 10 ** (-28 / 20), 10 ** (-35 / 20))
PEDESTAL = np.sqrt(2.0)  # its amplitude re the line's
# the block's clicks: each cylinder's combustion (with injection) and its valves seating, at crank angles after its
# firing top dead centre (exhaust closing just after the overlap, inlet closing ~40 degrees after bottom dead centre);
# cylinders differ (injectors, combustion, valve lash), so the pattern repeats once a cycle: a recorded truck idle
# (~700 rpm) clicks at the cycle rate's multiples (5.8, 11.6, 17.5 Hz), not only at the firing rate
VALVE_DEG, VALVE_LEVEL, VALVE_SPREAD, CLICK_DEG = (370.0, 580.0), 0.4, 0.3, 12.0  # heuristic
CYCLE_SPREAD = 0.3  # heuristic: and no two cycles are alike (log-SD per firing interval)
G, CRR, RHO = 9.81, 0.012, 1.2  # gravity, rolling resistance, air
# a manual heavy diesel's shift, double-clutched or rev-matched: off the throttle for a time measured on a recording
# (1.06-1.59 s over four shifts: log-normal, median 1.29 s, log-SD 0.18) while the engine falls to the next gear's
# speed (heuristic: 500 rpm/s), or, shifting down, is blipped up to it (1500 rpm/s, a little throttle)
SHIFT_LOG_S, REV_FALL, REV_BLIP = (np.log(1.29), 0.18), 500.0, 1500.0

LAYOUTS = {
    # fire: firing angles in the cycle's 720 crank degrees; bank: the exhaust manifold each cylinder (in firing order)
    # feeds; lag: extra crank angle on its way to the collector (unequal-length headers)
    "I3": dict(fire=(0, 240, 480)),
    "I4": dict(fire=(0, 180, 360, 540)),
    "boxer4": dict(fire=(0, 180, 360, 540), bank=(0, 0, 1, 1), lag=(0, 0, 40, 40)),  # 1-3-2-4, unequal headers: rumble
    "I6": dict(fire=(0, 120, 240, 360, 480, 600)),
    "V6": dict(fire=(0, 120, 240, 360, 480, 600), bank=(0, 1, 0, 1, 0, 1)),
    "V6-odd": dict(fire=(0, 90, 240, 330, 480, 570), bank=(0, 1, 0, 1, 0, 1)),  # 90-degree V6 on shared crankpins
    "V8-flat": dict(fire=tuple(range(0, 720, 90)), bank=(0, 1) * 4),  # each bank fires evenly, like a four
    "V8-cross": dict(fire=tuple(range(0, 720, 90)), bank=(0, 1, 1, 0, 1, 0, 0, 1)),  # 1-8-4-3-6-5-7-2: uneven banks
}

KINDS = {
    # gears: engine rpm per m/s of road speed; shift: upshift rpm for the gentlest and the most aggressive driver;
    # curve: full-load torque against rpm, re its peak (curve_turbo: a turbo petrol's; a big diesel's governor cuts
    # its fuel steeply near rated speed);
    # throttled: petrol (pulses grow with the throttle) or not (diesel: a full charge of air at every load);
    # pulse_deg: blowdown pulse width at light and full load (crank degrees, narrower is sharper and brighter);
    # muffler: the silencer's corner, Hz; airbox: the intake's resonance, Hz; mech: the block's noise through the
    # bonnet; clatter: how much of it comes in clicks (combustion, valves; diesel knock); knock_spread: log-SD of
    # the cylinders' combustion clicks; layouts: their shares; p_turbo;
    # p_high_flow: without and with a turbo or a V8; litres: displacement by layout; kw_per_l: a turbo engine's
    # rated power per litre; split: its compressor's half-order line re the blade-pass (0: none); whistle: its level;
    # boost: the torque (re peak) where boost starts and where it is full: the shaft's speed traces the torque
    # curve under full load (see sound; diesels: almost from the first fuel);
    # spool_s: turbo lag (heuristic values throughout)
    "petrol": dict(idle=(650.0, 800.0), gears=(450, 260, 170, 125, 100, 80), torque=220.0, mass=(1100.0, 1800.0),
                   curve=((600, 2500, 4000, 5500, 6500), (0.7, 0.9, 1.0, 0.9, 0.7)),
                   curve_turbo=((600, 1500, 1800, 4500, 6000), (0.55, 0.85, 1.0, 1.0, 0.8)),
                   cda=0.65, shift=(1900.0, 4000.0), manual=False, length=4.5, level=1.0, throttled=True,
                   pulse_deg=(80.0, 20.0), muffler=220.0, airbox=500.0, mech=0.18, clatter=0.3, knock_spread=0.15,
                   layouts={"I3": 0.15, "I4": 0.5, "boxer4": 0.04, "I6": 0.05, "V6": 0.15, "V6-odd": 0.02,
                            "V8-cross": 0.07, "V8-flat": 0.02},
                   litres={"I3": (1.0, 1.5), "I4": (1.3, 2.5), "boxer4": (2.0, 2.5), "I6": (2.5, 3.5), "V6": (3.0, 3.8),
                           "V6-odd": (3.8, 4.3), "V8-cross": (5.0, 6.6), "V8-flat": (3.8, 5.2)},
                   p_turbo=0.45, p_high_flow=(0.05, 0.4), kw_per_l=70.0, split=10 ** (-3 / 20), whistle=0.9, boost=(0.35, 1.0),
                   spool_s=0.4),
    "diesel": dict(idle=(700.0, 850.0), gears=(420, 240, 155, 115, 90, 72), torque=320.0, mass=(1300.0, 2100.0),
                   curve=((700, 1500, 1750, 2750, 4000, 4800), (0.45, 0.85, 1.0, 1.0, 0.75, 0.4)),
                   cda=0.7, shift=(1600.0, 2800.0), manual=False, length=4.6, level=1.3, throttled=False,
                   pulse_deg=(40.0, 28.0), muffler=180.0, airbox=400.0, mech=0.15, clatter=0.7, knock_spread=0.35,
                   layouts={"I3": 0.03, "I4": 0.75, "I6": 0.12, "V6": 0.1},
                   litres={"I3": (1.4, 1.6), "I4": (1.5, 2.2), "I6": (3.0, 3.5), "V6": (2.7, 3.3)},
                   p_turbo=1.0, p_high_flow=(0.02, 0.05), kw_per_l=55.0, split=10 ** (-3 / 20), whistle=0.4,
                   boost=(0.25, 0.95), spool_s=0.4),
    "heavy": dict(idle=(600.0, 750.0), gears=(760, 560, 410, 300, 220, 165, 125, 95), torque=1800.0,
                  mass=(12000.0, 40000.0), cda=6.0,
                  curve=((550, 1000, 1150, 1400, 1700, 1900, 2100), (0.5, 0.9, 1.0, 1.0, 0.8, 0.55, 0.0)), shift=(1500.0, 1900.0), manual=True, length=12.0, level=3.0, throttled=False,
                  pulse_deg=(40.0, 25.0), muffler=150.0, airbox=300.0, mech=0.16, clatter=0.6, knock_spread=0.35,
                  layouts={"I6": 0.85, "V8-cross": 0.15},
                  litres={"I6": (11.0, 16.0), "V8-cross": (13.0, 16.5)},
                  p_turbo=1.0, p_high_flow=(0.0, 0.0), kw_per_l=25.0, split=0.0, whistle=0.4, boost=(0.25, 0.95),
                  spool_s=0.5),
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


def torque_curve(k: dict, e: dict) -> tuple:
    return k["curve_turbo"] if e["turbo"] and "curve_turbo" in k else k["curve"]


def drive(rng: np.random.Generator, car: dict, v: np.ndarray, acc: np.ndarray,
          e: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Engine rpm and load (0-1) per step from road speed and acceleration, through the gearbox. Load is the
    tractive force needed (inertia, rolling, drag) over what the engine gives at that rpm (its torque curve) in that
    gear. e: the engine (engine(); drawn if not given)."""
    k = KINDS[car["kind"]]
    curve = torque_curve(k, e or engine(rng, car))
    gears, aggr = np.array(k["gears"], float), car["aggr"]
    idle, mass = rng.uniform(*k["idle"]), rng.uniform(*k["mass"])  # mass: laden or not, unless the car says
    mass = car.get("mass", mass)
    down, launch = idle + 350.0, 600.0 if k["manual"] else 1500.0 + 1000.0 * aggr
    ok = np.flatnonzero(v[0] * gears >= down)
    g = int(ok.max()) if ok.size else 0  # start in the highest gear that keeps the engine above the downshift
    rpm, load = np.empty(v.size), np.empty(v.size)
    shift = lambda: int(round(np.exp(rng.normal(*SHIFT_LOG_S)) / DT)) if k["manual"] else 0  # noqa: E731
    clutch, r, shifting_down = 0, idle, False
    pull = curve[0][list(curve[1]).index(1.0)]  # the start of peak torque
    for j in range(v.size):
        force = mass * (acc[j] + G * CRR) + 0.5 * RHO * k["cda"] * v[j] ** 2
        avail = k["torque"] * np.interp(max(idle, v[j] * gears[g]), *curve)
        ld = float(np.clip(force / (max(avail, 1.0) * gears[g] * 2 * np.pi / 60), 0.05, 1.0))
        if k["manual"]:  # so little power for the mass that any real acceleration takes full fuel: they gun it
            ld = max(ld, float(np.clip((acc[j] - 0.05) / 0.1, 0.0, 1.0)))
        up = k["shift"][0] + (k["shift"][1] - k["shift"][0]) * aggr * (0.4 + 0.6 * ld)
        if not clutch and g < gears.size - 1 and v[j] * gears[g] > up:
            clutch, shifting_down = shift(), False
            # a manual's driver picks the gear that will pull from peak torque once the clutch is back in, skipping
            # gears if the vehicle has gained enough speed meanwhile
            fit = np.flatnonzero((v[j] + max(acc[j], 0.0) * clutch * DT) * gears >= pull) if k["manual"] else []
            g = min(max(int(fit.max()) if len(fit) else 0, g + 1), gears.size - 1)
        elif not clutch and g > 0 and v[j] * gears[g] < down:
            g, clutch, shifting_down = g - 1, shift(), True
        target = max(idle, v[j] * gears[g])
        if g == 0 and acc[j] > 0.05:  # pulling away: torque converter (or clutch) slip
            target = max(target, idle + ld * launch * float(np.clip(1 - v[j] / 5.0, 0.0, 1.0)))
        if clutch:  # the pull stops while the engine is brought to the next gear's speed
            blip = shifting_down and target > r  # only a downshift is blipped; shifting up, the engine just falls
            r += float(np.clip(target - r, -REV_FALL * DT, REV_BLIP * DT if blip else 0.0))
            ld, clutch = 0.3 if blip else 0.05, clutch - 1
        else:
            r = target
        rpm[j], load[j] = r, ld
    smooth = lambda x, tau: signal.lfilter([1 - np.exp(-DT / tau)], [1, -np.exp(-DT / tau)], x, zi=[x[0] * np.exp(-DT / tau)])[0]  # noqa: E731
    return smooth(rpm, 0.25), smooth(load, 0.15)


def engine(rng: np.random.Generator, car: dict) -> dict:
    """What is under the bonnet, drawn from the kind's shares unless the car names it ("layout", "turbo",
    "high_flow", "litres"): the layout and its displacement, a turbocharger or not, a stock or high-flow exhaust,
    and the exhaust's paths: each cylinder's pulse a little different, each bank's pipe its own length (heuristic:
    0.6-1 in level, 0-30 crank degrees later), so even a merged V engine keeps some of its banks' unevenness."""
    k = KINDS[car["kind"]]
    u = rng.uniform(size=3)
    names, p = list(k["layouts"]), np.array(list(k["layouts"].values()))
    layout = car.get("layout") or names[int(np.searchsorted(np.cumsum(p) / p.sum(), u[0], side="right"))]
    turbo = car["turbo"] if "turbo" in car else bool(u[1] < k["p_turbo"])
    high_flow = car["high_flow"] if "high_flow" in car else bool(u[2] < k["p_high_flow"][turbo or layout.startswith("V8")])
    lay = LAYOUTS[layout]
    fire = np.array(lay["fire"], float)
    bank = np.array(lay.get("bank", (0,) * fire.size))
    path_amp, path_lag = rng.uniform(0.6, 1.0, 2), rng.uniform(0.0, 30.0, 2)
    litres = car.get("litres") or rng.uniform(*k["litres"][layout])
    rc = rng.spawn(1)[0]  # the cylinders' differences
    cyl_lag, cyl_amp, jitter = rc.normal(0.0, CYL_LAG, fire.size), rc.lognormal(0.0, CYL_SPREAD, fire.size), rc.spawn(1)[0]
    return dict(layout=layout, turbo=turbo, high_flow=high_flow, fire=fire, litres=litres, jitter=jitter,
                amp=rng.uniform(0.9, 1.1, fire.size) * path_amp[bank] * cyl_amp,
                arrive=fire + np.array(lay.get("lag", (0,) * fire.size), float) + path_lag[bank] + cyl_lag,
                # the turbo matched to the engine (smaller ones spin faster), give or take
                turbo_hz=TURBO_HZ * (k["kw_per_l"] * litres / TURBO_KW) ** -0.5 * rng.uniform(0.9, 1.1), split=k["split"],
                # how hard each cylinder's combustion and each valve click (cylinders differ)
                knock=rng.lognormal(0.0, k["knock_spread"], fire.size),
                valves=VALVE_LEVEL * rng.lognormal(0.0, VALVE_SPREAD, (len(VALVE_DEG), fire.size)))


def pulses(at: np.ndarray, amp: np.ndarray, width: float, envelope: bool = False, harmonics: int = HARMONICS,
           rise: float = 0.0) -> np.ndarray:
    """One engine cycle of pressure pulses arriving at crank angles `at`: each a fast rise and an exponential fall
    (an alpha function peaking `width` degrees after it starts), at unit mean: the same gas each cycle however it is
    released, so a sharp release is a strong, bright pulse and a slow one little more than steady flow. A waveform
    has its mean removed and is band-limited to HARMONICS; an envelope is left as it is."""
    th = np.arange(TABLE) * 720.0 / TABLE
    u = ((th[None, :] - at[:, None]) % 720.0) / width
    # rise: a steep front (rising over that fraction of the width, then the same fall) instead of the alpha's gradual one
    y = (amp[:, None] * ((1 - np.exp(-u / rise)) * np.exp(-u) if rise else u * np.exp(1 - u))).sum(0)
    y /= y.mean()
    if envelope:
        return y
    Y = np.fft.rfft(y)
    Y[0], Y[harmonics + 1:] = 0.0, 0.0
    return np.fft.irfft(Y, TABLE)


def resonant_lowpass(f0: float, q: float, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Second-order low-pass with a resonance at f0 (an airbox and its snorkel as a Helmholtz resonator)."""
    w = 2 * np.pi * min(f0, 0.45 * fs) / fs
    al, c = np.sin(w) / (2 * q), np.cos(w)
    return np.array([(1 - c) / 2, 1 - c, (1 - c) / 2]), np.array([1 + al, -2 * c, 1 - al])


def pedestal(r: np.ndarray) -> np.ndarray:
    """The compressor's broadband noise at r = f / line, in amplitude: second-order high-pass at 0.35, third-order
    low-pass at 1.05 (fitted to the recording)."""
    return 1 / np.sqrt((1 + (0.35 / r) ** 4) * (1 + (r / 1.05) ** 6))


def turbo(rng: np.random.Generator, f: np.ndarray, log_f: np.ndarray, bands, e: dict, fs: int, seg) -> np.ndarray:
    """The compressor at unit line level: the blade-pass line, its harmonics and (with splitter blades) its half-order
    line as narrowband noise around the blade-pass frequency f (per sample), each faded out before Nyquist, and the
    pedestal from the half-octave noise
    bands, weighted around f (its power is counted over bands up to 64 x f, so what lies above Nyquist is lost)."""
    phase = 2 * np.pi * np.cumsum(f) / fs
    y = np.zeros(f.size)
    for h, a in [(0.5, e.get("split", 0.0))] * bool(e.get("split")) + list(zip((1, 2, 3), TURBO_H)):
        nyq = np.clip((0.45 * fs - h * f) / (0.05 * fs), 0.0, 1.0)
        if not nyq.any():
            break
        sos = signal.butter(2, h * e["turbo_hz"] / TURBO_Q / 2, "lowpass", fs=fs, output="sos")
        iq = signal.sosfilt(sos, rng.standard_normal((2, f.size)), axis=1)
        line = iq[0] * np.cos(h * phase) - iq[1] * np.sin(h * phase)
        y += a * nyq * line / (np.std(line) + 1e-12)
    # the pedestal's weights change slowly: worked out every 64 samples and interpolated
    coarse, f_c = np.arange(0, f.size, 64), np.exp(log_f[::64])
    norm = np.sqrt((pedestal(700.0 * 2 ** (np.arange(30) / 2)[:, None] / f_c[None, :]) ** 2).sum(0))
    return y + PEDESTAL * sum(np.interp(np.arange(f.size), coarse, pedestal(c / f_c) / norm) * seg(b) for c, b in bands)


PARTS = ("exhaust", "exhaust-flow", "intake", "throttle", "turbo", "block", "tyres", "air-brakes")


def sound(rng: np.random.Generator, car: dict, e: dict, te: np.ndarray, t: np.ndarray, rpm, load, v, fade, banks,
          fs, parts=PARTS) -> np.ndarray:
    """A vehicle's sound at the source (1 m), sampled at emission times te. Levels change slowly, so they are worked
    out on the simulation's steps and interpolated; the waveforms are table lookups and noise banks. parts: which of
    PARTS to sound (exhaust: the pulses; exhaust-flow: the flow noise in them; block: the mechanical noise); the
    others are silenced but drawn all the same, so the parts add up to the whole vehicle."""
    k = KINDS[car["kind"]]
    lv, r1k = k["level"] * fade, rpm / 1000
    exh = (HIGH_FLOW if e["high_flow"] else 1.0) / (TURBINE if e["turbo"] else 1.0)
    x1 = k["throttled"] and not e["high_flow"]  # a stock petrol exhaust: fitted to the X1
    ta, tb = SHELF_TONE_LOAD if x1 else (0.25, 0.75)
    tone = ENGINE * lv * r1k ** 1.5 * (ta + tb * load) * exh
    # petrol breathes through its throttle: weak intake pulses and a hiss past the nearly closed plate at light load
    breathe = 0.1 + 0.9 * load if k["throttled"] else 0.5 + 0.5 * load
    hiss = THROTTLE * lv * r1k * (1 - load) ** 2 if k["throttled"] else np.zeros_like(lv)
    spool = np.zeros_like(lv)  # shaft speed, 0-1: it builds with the exhaust's flow, after a lag, and falls faster
    if e["turbo"]:
        # the turbine runs on the exhaust's enthalpy flow (mass flow x temperature drop) and drives the compressor,
        # whose work also goes with the mass flow, so the flow mostly cancels: boost, and the shaft's speed, follow the
        # exhaust's temperature, i.e. the fuel per cycle, i.e. the torque; what is left of the flow is a weak rise
        # with rpm (heuristic: ^0.25)
        curve = torque_curve(k, e)
        peak = curve[0][list(curve[1]).index(1.0)]
        torque = load * np.interp(rpm, *curve) * (rpm / peak) ** 0.25
        target = np.clip((torque - k["boost"][0]) / (k["boost"][1] - k["boost"][0]), 0.0, 1.0)
        up, down = 1 - np.exp(-DT / k["spool_s"]), 1 - np.exp(-1.5 * DT / k["spool_s"])
        for j in range(1, spool.size):
            d = target[j] - spool[j - 1]
            spool[j] = spool[j - 1] + d * (up if d > 0 else down)
    f_line = e["turbo_hz"] * (TURBO_IDLE + (1 - TURBO_IDLE) * spool)
    gas = r1k * (0.5 + 0.5 * load)  # the exhaust gas's speed, roughly
    # power as speed^8 (Lighthill) at first, then growing only as fast as the tones (speed^3 in power): against the
    # tones it rises by ~11-21 dB from idle, then levels off, alike pulling hard at 3000-4500 rpm and blipping to 6000
    flow = EXHAUST * lv * gas ** 4 / (1 + (gas / FLOW_SAT) ** 2.5) * (1.0 if e["high_flow"] else FLOW_STOCK)
    gains = (tone * (1 - load), tone * load, flow, k["mech"] * lv * r1k ** 2 * (0.5 + 0.5 * load),
             TYRE * (v / 15.0) ** 2 * (1.5 if car["kind"] == "heavy" else 1.0) * fade,
             INTAKE * lv * r1k ** 1.5 * breathe * (COMPRESSOR if e["turbo"] else 1.0), hiss,
             # the compressor's pressure rise goes as its tip speed^2 (Euler) and the noise's amplitude as boost^2:
             # shaft speed^4
             TURBO * k["whistle"] * lv * (f_line / e["turbo_hz"]) ** 4 * (spool > 0))
    on = [float(name in parts) for name in ("exhaust", "exhaust", "exhaust-flow", "block", "tyres", "intake", "throttle", "turbo")]
    g_soft, g_hard, g_exh, g_mech, g_tyre, g_in, g_hiss, g_turbo = (np.interp(te, t, g) * o for g, o in zip(gains, on))
    cycles = np.cumsum(np.interp(te, t, rpm) / (120.0 * fs))  # engine cycles (one every two revolutions)
    pos = ((cycles % 1.0) * TABLE).astype(np.intp)
    w_soft, w_hard = np.array(k["pulse_deg"]) * (TURBINE if e["turbo"] else 1.0)
    hm, rise = (128, PULSE_RISE) if x1 else (HARMONICS, 0.0)
    soft, hard = (pulses(e["arrive"], e["amp"], w, harmonics=hm, rise=rise) for w in (w_soft, w_hard))
    gate = pulses(e["arrive"], e["amp"], w_hard, envelope=True)[pos]  # the flow noise comes in the pulses
    clicks = np.concatenate([e["fire"]] + [e["fire"] + d for d in VALVE_DEG])
    knock = pulses(clicks, np.concatenate([e["knock"], *e["valves"]]), CLICK_DEG, envelope=True)[pos]
    event = (cycles * e["fire"].size).astype(np.intp)  # firing intervals so far
    knock *= rng.lognormal(-CYCLE_SPREAD ** 2 / 2, CYCLE_SPREAD, event[-1] + 1)[event]
    # intake: a common plenum, so its pulses (rarefactions as each inlet valve closes) are even, whatever the banks
    inlet = -pulses(e["fire"] + 220.0, rng.uniform(0.9, 1.1, e["fire"].size), 40.0)
    m = te.size
    seg = lambda bank: bank[(o := int(rng.integers(0, bank.size - m))):o + m]  # noqa: E731
    corner = min(1000.0 + 60.0 * v[fade > 0].max(), 0.45 * fs)  # faster: brighter
    tyre = signal.sosfilt(signal.butter(2, [300.0, corner], "bandpass", fs=fs, output="sos"), seg(banks["tyre"]))
    x = g_soft * soft[pos] + g_hard * hard[pos]
    if e.get("jitter") is not None:  # no two firings alike (smoothly from one to the next)
        jit = e["jitter"].lognormal(-FIRING_SPREAD ** 2 / 2, FIRING_SPREAD, event[-1] + 2)
        x *= np.interp(cycles * e["fire"].size, np.arange(jit.size), jit)
    # the silencer; a high-flow one is freer: gentler, and higher
    if x1:
        x = signal.sosfilt(signal.butter(3, TONE_CEILING, "lowpass", fs=fs, output="sos"), x)  # the harmonics' ceiling
        o = int(banks["floor_rng"].integers(0, banks["floor"].size - m))  # white-ish noise in the source
        x += FLOOR * (g_soft + g_hard) * banks["floor"][o:o + m]
        x = signal.oaconvolve(x, shelf(fs), mode="same")
        res = signal.sosfiltfilt(signal.butter(2, RES_CUT, "lowpass", fs=fs, output="sos"), x)
        x += res * RES_BOOST * np.minimum(np.interp(te, t, load), 0.5)  # the resonances, driven harder as the throttle opens
        x *= X1_GAIN
    else:
        order, f_m = (2, 2.5 * k["muffler"]) if e["high_flow"] else (4, k["muffler"])
        x = signal.sosfilt(signal.butter(order, min(f_m, 0.45 * fs), "lowpass", fs=fs, output="sos"), x)
    # then the flow noise made at the tailpipe, pulsing with the exhaust, brighter as the gas speeds up
    bright = np.interp(te, t, np.clip(gas / FLOW_FULL, 0.0, 1.0))
    x += g_exh * (np.sqrt(1 - bright) * seg(banks["flow"]) + np.sqrt(bright) * seg(banks["flow_bright"])) * (0.5 + 0.5 * gate)
    rear, x = x, np.zeros_like(x)  # the exhaust so far, heard from the back; the rest from the engine bay
    if x1:  # a petrol airbox as on the X1: everything below ~430 Hz grows with the throttle
        y_in = signal.lfilter(*resonant_lowpass(430.0, 2.0, fs), g_in * inlet[pos])
        x += signal.sosfilt(signal.butter(2, 450.0, "lowpass", fs=fs, output="sos"), y_in)
    else:
        x += signal.lfilter(*resonant_lowpass(k["airbox"], 2.0, fs), g_in * inlet[pos])  # a plastic airbox: little damping
    x += g_mech * seg(banks["mech"]) * (1 - k["clatter"] + k["clatter"] * knock) + g_tyre * tyre / (np.std(tyre) + 1e-12)
    x += g_hiss * seg(banks["hiss"])
    if e["turbo"]:
        x += g_turbo * turbo(rng, np.interp(te, t, f_line), np.interp(te, t, np.log(f_line)), banks["turbo"], e, fs, seg)
    if car["kind"] == "heavy":  # air dumped from the brakes as it comes to a stop, sometimes
        hiss = seg(banks["hiss"])
        for j in np.flatnonzero((v[:-1] >= 0.3) & (v[1:] < 0.3)):
            if rng.uniform() < 0.6:
                start = int((t[j] + rng.uniform(0.3, 1.0) - te[0]) * fs)
                n_h = int(rng.uniform(0.6, 1.5) * fs)
                if 0 <= start < m - n_h:
                    x[start:start + n_h] += float("air-brakes" in parts) * HISS * np.interp(te[start], t, lv) * hiss[start:start + n_h] * np.exp(-np.arange(n_h) / (0.4 * fs))
    return rear, x


_SHELF = {}


def shelf(fs: int) -> np.ndarray:
    """The stepped stock silencer (SHELF_DB with SHELF_PEAKS) as a linear-phase FIR, fine enough for the narrow peaks."""
    if fs not in _SHELF:
        f = np.linspace(0.0, fs / 2, 4097)
        db = np.interp(f, *(np.array(v, float) for v in zip(*SHELF_DB)))
        for fc, gain, width in SHELF_PEAKS:
            db += gain * np.exp(-0.5 * (np.log(np.maximum(f, 1.0) / fc) / width) ** 2)
        _SHELF[fs] = signal.firwin2(int(fs // 8) | 1, f, 10 ** (db / 20), fs=fs)  # ~8 Hz resolution
    return _SHELF[fs]


def render(rng: np.random.Generator, scene: Scene, facade: spatial.Facade, rx: spatial.Receiver,
           sim: tuple | None = None, parts=PARTS) -> np.ndarray:
    """sim: (t, cars) as simulate returns them, to hear given vehicles (e.g. one at a time) instead of the street's;
    parts: of each vehicle's sound (PARTS), e.g. to hear one at a time (the distant road bed only comes with all
    of them)."""
    cfg, fs = scene.traffic, scene.fs
    n = int(scene.duration_s * fs)
    outside = not scene.room.indoor  # indoors, everything arrives through the facade's opening
    out = rx.zeros() if outside else np.zeros(n)
    t, cars = simulate(rng, scene) if sim is None else sim
    t0 = -(HEARD_M + 10.0) / spatial.C  # the earliest emission heard in the clip
    te = t0 + np.arange(n + int(np.ceil(-t0 * fs))) / fs
    w = t >= t0 - 3.0  # the engine model settles over a few seconds before that
    nb = te.size + 4 * fs  # noise banks: each vehicle reads its own stretch
    bp = lambda lo, hi, x: signal.sosfilt(signal.butter(2, [lo, hi], "bandpass", fs=fs, output="sos"), x)  # noqa: E731
    banks = {"tyre": coloured_noise(rng, nb, 3.0, fs), "mech": bp(300.0, 1500.0, rng.standard_normal(nb)),
             "hiss": bp(1500.0, min(6000.0, 0.45 * fs), rng.standard_normal(nb))}
    pink = coloured_noise(rng, nb, 3.0, fs)  # exhaust flow noise: flat per octave from 150 Hz to a corner
    banks["flow"], banks["flow_bright"] = (bp(150.0, min(c, 0.45 * fs), pink) for c in (2000.0, 6000.0))
    for key in ("mech", "hiss", "flow", "flow_bright"):
        banks[key] /= np.std(banks[key])
    # the stock petrol exhaust's source noise, from a spawned generator (the layer's own stream does not move)
    rf = rng.spawn(1)[0]
    fl = bp(300.0, min(8000.0, 0.45 * fs), coloured_noise(rf, nb, 1.5, fs))
    banks["floor"], banks["floor_rng"] = fl / np.std(fl), rf
    # turbo: overlapping half-octave bands from 700 Hz, which each turbo weights around its centre
    centres = [c for c in 700.0 * 2 ** (np.arange(12) / 2) if c * 2 ** 0.35 < 0.45 * fs]
    banks["turbo"] = [(c, (x := bp(c / 2 ** 0.35, c * 2 ** 0.35, rng.standard_normal(nb))) / np.std(x)) for c in centres]
    for car in cars:
        r = np.hypot(car["x"], np.hypot(car["y"], scene.mic[2] - 0.5))
        fade = np.clip((HEARD_M - r) / FADE_M, 0.0, 1.0) * car["on"]
        if not fade[w].any():
            continue
        e = engine(rng, car)
        if "rpm" in car:  # given (e.g. revving on the spot), not through the gearbox
            rpm, load = car["rpm"][w], car["load"][w]
        else:
            rpm, load = drive(rng, car, car["v"][w], car["acc"][w], e)
        live = np.flatnonzero(fade[w])  # only while it is within earshot
        a = max(int((t[w][live[0]] - DT - t0) * fs), 0)
        b = min(int(np.ceil((t[w][live[-1]] + DT - t0) * fs)), te.size)
        x = sound(rng, car, e, te[a:b], t[w], rpm, load, car["v"][w], fade[w], banks, fs, parts)
        span = (max(int(te[a] * fs), 0), min(int(np.ceil((te[b - 1] + (HEARD_M + 10.0) / spatial.C) * fs)) + 1, n))

        heading = car.get("heading") or (1.0 if car["x"][-1] >= car["x"][0] else -1.0)
        length = car.get("length", KINDS[car["kind"]]["length"])
        for xs, off in zip(x, (-0.5 * length, 0.3 * length)):  # the tailpipe, the engine bay
            def path(tt, tx=t, xx=car["x"] + heading * off, y=car["y"]):
                return np.stack([np.interp(tt, tx, xx), np.full_like(tt, y), np.full_like(tt, 0.5)], axis=1)

            if outside:
                out += rx.place_moving(xs, path, t0=te[a], span=span)
            else:
                out += spatial.place_moving(xs, path, scene.mic, fs, t0=te[a], n=n, span=span)
    bed = coloured_noise(rng, n, 6.0, fs)
    env = np.interp(np.arange(n), np.linspace(0, n, 8), 1 + 0.3 * rng.standard_normal(8))
    bed = bed * np.clip(env, 0.3, None) * np.std(out + 1e-9) * 10 ** (cfg.bed_db / 20)
    if set(parts) >= set(PARTS):  # the distant road is no one vehicle's part
        out += rx.diffuse(bed, rng) if outside else bed
    if scene.room.indoor:
        out = facade(out, rng)
    return out
