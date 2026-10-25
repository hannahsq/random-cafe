import numpy as np
import pytest
from scipy import signal

from random_cafe import Scene, mix, render, spatial

FS = 16000


def fake_talkers(n_speakers=12, per=4):
    """speaker -> fake paths; the loader turns a path into a short harmonic 'utterance' with its own f0."""
    return {f"s{i}": [f"s{i}/u{j}" for j in range(per)] for i in range(n_speakers)}


def fake_load(path, fs):
    i, j = (int(p[1:]) for p in path.split("/"))
    t = np.arange(int((1.0 + 0.3 * j) * fs)) / fs
    f0 = 100 + 15 * i
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 15))
    return x * np.hanning(t.size)


def scene(**kw):
    s = Scene(duration_s=4.0, fs=FS, seed=kw.pop("seed", 1))
    for k, v in kw.items():
        obj, attr = k.split("__")
        setattr(getattr(s, obj), attr, v)
    return s


def test_seeded_and_deterministic():
    a = render(scene(), fake_talkers(), fake_load)
    b = render(scene(), fake_talkers(), fake_load)
    c = render(scene(seed=2), fake_talkers(), fake_load)
    assert set(a) == {"babble", "clatter"}
    for k in a:
        assert np.array_equal(a[k], b[k])
        assert not np.array_equal(a[k], c[k])


def test_layers_independent():
    """switching any layer on or off must not change the others (room tails and the facade included)"""
    names = ("babble", "clatter", "kitchen", "traffic", "rain", "wind")
    full = render(scene(kitchen__enabled=True, traffic__enabled=True, rain__enabled=True, wind__enabled=True),
                  fake_talkers(), fake_load)
    for off in names:
        kw = {f"{n}__enabled": n != off for n in names}
        part = render(scene(**kw), fake_talkers(), fake_load)
        assert set(part) == set(full) - {off}
        for k in part:
            assert np.array_equal(part[k], full[k]), (off, k)


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


def test_stems_and_mix():
    st = render(scene(traffic__enabled=True, rain__enabled=True, kitchen__enabled=True), fake_talkers(), fake_load)
    assert set(st) == {"babble", "clatter", "kitchen", "traffic", "rain"}
    for x in st.values():
        assert x.shape == (4 * FS,) and np.all(np.isfinite(x))
    assert np.allclose(mix(st), sum(st.values()))
    assert rms_db(mix(st, target_dbfs=-20.0)) == pytest.approx(-20.0, abs=1e-6)


def mean_db(layer, seeds=range(4), **kw):
    """stem level averaged over seeds (single renders vary)"""
    return np.mean([rms_db(render(scene(seed=s, **kw), fake_talkers(30), fake_load)[layer]) for s in seeds])


def test_full_band_rates():
    """at 44.1 and 48 kHz every layer renders, reaches above 8 kHz, and keeps its 16 kHz level (one calibration)"""
    from random_cafe.scene import SAMPLE_RATES
    kw = dict(kitchen__enabled=True, traffic__enabled=True, rain__enabled=True, wind__enabled=True, room__indoor=False)
    base = {k: [] for k in ("babble", "clatter", "kitchen", "traffic", "rain", "wind")}
    levels = {fs: {k: [] for k in base} for fs in SAMPLE_RATES}
    for fs in SAMPLE_RATES:
        for seed in range(6):
            s = scene(seed=seed, **kw)
            s.fs, s.duration_s = fs, 3.0
            for k, x in render(s, fake_talkers(30), fake_load).items():
                assert np.all(np.isfinite(x))
                levels[fs][k].append(rms_db(x))
                if fs > 16000 and k in ("rain", "clatter"):
                    f, p = signal.welch(x, fs, nperseg=1024)
                    assert p[f > 9000].sum() > 1e-3 * p.sum()
    for fs in SAMPLE_RATES[1:]:
        # median over seeds (one near vehicle can set a short clip's level); clatter is too sparse for short clips
        # even so (one near impact sets the level), and agrees within a dB over 8 x 20 s
        for k in set(base) - {"clatter"}:
            assert abs(np.median(levels[fs][k]) - np.median(levels[16000][k])) < 3.0, (fs, k)


def test_levels_follow_the_scene():
    """intensities are physical: heavier rain, denser traffic and more tables are louder; walls make things quieter"""
    assert mean_db("rain", rain__enabled=True, rain__rain_mm_h=30.0) > mean_db("rain", rain__enabled=True, rain__rain_mm_h=1.5) + 10
    assert mean_db("traffic", traffic__enabled=True, traffic__vehicles_per_min=40.0, room__indoor=False) > \
        mean_db("traffic", traffic__enabled=True, traffic__vehicles_per_min=3.0, room__indoor=False) + 5
    assert mean_db("babble", babble__tables=(8, 10)) > mean_db("babble", babble__tables=(1, 1)) + 3
    outside = mean_db("traffic", traffic__enabled=True, room__indoor=False)
    windows = mean_db("traffic", traffic__enabled=True, room__indoor=True, room__p_door_open=0.0)
    door = mean_db("traffic", traffic__enabled=True, room__indoor=True, room__p_door_open=1.0)
    assert outside - 30 < windows < outside - 10  # audible through the windows, well below outside
    assert windows + 3 < door < outside  # an open front door lets in much more
    assert mean_db("kitchen", kitchen__enabled=True, kitchen__open=False) < mean_db("kitchen", kitchen__enabled=True, kitchen__open=True) - 8


def test_traffic_flow_and_engines():
    """vehicles never run into each other, the lanes run opposite ways, lights make traffic stop (and only lights
    do), and engines idle at 550-850 rpm and rev up pulling away"""
    from random_cafe.layers import traffic

    def sim(**kw):
        s = scene(traffic__enabled=True, traffic__vehicles_per_min=30.0, traffic__crossing_m=(-20.0, 20.0), **kw)
        s.duration_s = 30.0
        return traffic.simulate(np.random.default_rng(0), s)

    t, cars = sim()
    for lane_y in {c["y"] for c in cars}:
        lane = [c for c in cars if c["y"] == lane_y]
        for ahead, behind in zip(lane, lane[1:]):
            both = ahead["on"] & behind["on"]
            assert np.all(np.abs(ahead["x"][both] - behind["x"][both]) > ahead["length"])
        moving = [np.sign(np.diff(c["x"][c["on"]])).sum() for c in lane if c["on"].sum() > 10]
        assert len(set(np.sign(moving))) == 1  # one direction per lane
    assert len({c["y"] for c in cars}) == 2
    stopped = lambda cs: np.mean(np.concatenate([c["v"][c["on"] & (t > 0)] for c in cs]) < 0.3)  # noqa: E731
    assert stopped(cars) < 0.002
    t, lit = sim(traffic__lights=True)
    assert stopped(lit) > 0.01
    car = next(c for c in lit if c["kind"] != "heavy" and (c["v"][c["on"]] < 0.3).any() and c["v"].max() > 8)
    rpm, load = traffic.drive(np.random.default_rng(1), car, car["v"], car["acc"])
    still = car["on"] & (car["v"] < 0.1) & (np.abs(car["acc"]) < 0.05)
    assert 550 < np.median(rpm[still]) < 850
    assert rpm[car["on"] & (car["acc"] > 0.5)].max() > 1500


def test_heavy_shifts():
    """A prime mover's shifts take a second or so off the throttle (double-clutched), with the engine falling to
    the next gear's speed rather than jumping to it."""
    from random_cafe.layers import traffic
    t = np.arange(0, 40, traffic.DT)
    v = np.minimum(0.5 * t, 20.0)
    car = dict(kind="heavy", aggr=0.5, layout="I6", turbo=True, high_flow=False, mass=30000.0)
    rpm, load = traffic.drive(np.random.default_rng(0), car, v, np.gradient(v, t))
    off = np.flatnonzero(np.diff((load < 0.15).astype(int)))  # load is smoothed: a lower threshold sees less
    gaps = (off[1::2] - off[::2])[: (off.size // 2)] * traffic.DT
    assert gaps.size >= 4 and np.all((gaps > 0.7) & (gaps < 2.5))
    assert np.abs(np.diff(rpm)).max() < 1.5 * traffic.REV_BLIP * traffic.DT


def test_vehicle_parts_add_up():
    """each part of a vehicle's sound can be heard alone, and the parts add up to the whole"""
    from random_cafe import spatial
    from random_cafe.layers import traffic
    s = scene(traffic__enabled=True, traffic__vehicles_per_min=30.0, traffic__bed_db=-np.inf, room__indoor=False)
    rx = spatial.Receiver(s.mic, s.fs, int(s.duration_s * s.fs))
    room = spatial.SharedRoom(np.random.default_rng(0), s.room.t60_s, s.room.drr_db, s.fs)
    facade = spatial.Facade(np.random.default_rng(1), s.room, room, s.fs, rx)
    run = lambda parts: traffic.render(np.random.default_rng(2), s, facade, rx, parts=parts)  # noqa: E731
    whole = run(traffic.PARTS)
    parts = [run((p,)) for p in traffic.PARTS]
    assert np.abs(sum(parts) - whole).max() < 1e-9 * np.abs(whole).max() + 1e-12
    assert sum(np.mean(x ** 2) > 0 for x in parts) >= 5


def test_stock_silencer():
    """the stock petrol silencer as measured on a BMW X1: its strongest resonance (~307 Hz) well clear of the trough
    below it, the fall above ~390 Hz, and the higher resonances (e.g. ~1.7 kHz) standing out"""
    from random_cafe.layers import traffic
    w, H = signal.freqz(traffic.shelf(FS), worN=2 ** 15, fs=FS)
    db = lambda f: 20 * np.log10(np.abs(H[np.argmin(np.abs(w - f))]))  # noqa: E731
    assert db(307) - db(260) > 10 and db(377) - db(440) > 10
    assert db(1700) - (db(1640) + db(1760)) / 2 > 1.5


def test_engine_layouts():
    """A flat-plane V8's banks each fire evenly (every 180 degrees: only every fourth cycle harmonic); a cross-plane
    V8's do not, which is its burble. The same gas released sharply makes a stronger, brighter firing order."""
    from random_cafe.layers import traffic
    for name, uneven in (("V8-flat", False), ("V8-cross", True)):
        lay = traffic.LAYOUTS[name]
        at = np.array(lay["fire"], float)[np.array(lay["bank"]) == 0]
        spec = np.abs(np.fft.rfft(traffic.pulses(at, np.ones(at.size), 30.0)))[1:traffic.HARMONICS + 1]
        off = np.arange(1, traffic.HARMONICS + 1) % 4 != 0
        assert (spec[off].max() > 0.1 * spec.max()) == uneven
    at = np.array(traffic.LAYOUTS["I4"]["fire"], float)
    soft, sharp = (np.abs(np.fft.rfft(traffic.pulses(at, np.ones(4), w))) for w in (80.0, 20.0))
    assert sharp[4] > soft[4] and sharp[20] / sharp[4] > 2 * soft[20] / soft[4]  # 4: the firing order


def test_turbo_line():
    """The compressor's whistle is a narrow line at the blade-pass frequency, H2 about 28 dB down (and, for a car's
    compressor, a half-order line about 3 dB down), and gone once it would pass Nyquist."""
    from random_cafe.layers import traffic
    fs, n = 44100, 44100
    e = dict(turbo_hz=8000.0)
    y = traffic.turbo(np.random.default_rng(0), np.full(n, 5000.0), np.full(n, np.log(5000.0)), [], e, fs, None)
    f, p = signal.welch(y, fs, nperseg=8192)
    band = lambda lo, hi: p[(f >= lo) & (f < hi)].sum()  # noqa: E731
    assert band(4950, 5050) > 0.9 * band(3000, 7000)
    assert -33 < 10 * np.log10(band(9900, 10100) / band(4950, 5050)) < -23
    # a car's compressor, with splitter blades: a line at half the blade-pass, ~3 dB down
    y = traffic.turbo(np.random.default_rng(0), np.full(n, 5000.0), np.full(n, np.log(5000.0)), [],
                      dict(turbo_hz=8000.0, split=10 ** (-3 / 20)), fs, None)
    f, p = signal.welch(y, fs, nperseg=8192)
    assert -4 < 10 * np.log10(band(2475, 2525) / band(4950, 5050)) < -2
    y = traffic.turbo(np.random.default_rng(0), np.full(n, 5000.0), np.full(n, np.log(5000.0)), [], e, 16000, None)
    f, p = signal.welch(y, 16000, nperseg=4096)
    assert np.all(p[f > 7300] < 1e-6 * p.max())


def test_wind():
    """buffeting is dynamic pressure (12 dB per doubling of speed) and mostly low frequency; wind drives rain
    onto the windows"""
    calm = mean_db("wind", wind__enabled=True, wind__speed_ms=3.0, room__indoor=False)
    gale = mean_db("wind", wind__enabled=True, wind__speed_ms=12.0, room__indoor=False)
    assert gale > calm + 20
    x = render(scene(wind__enabled=True, babble__enabled=False, clatter__enabled=False, room__indoor=False))["wind"]
    f, p = signal.welch(x, FS, nperseg=4096)
    assert p[f < 300].sum() > 0.8 * p.sum()
    still = mean_db("rain", rain__enabled=True, rain__rain_mm_h=8.0, wind__speed_ms=0.0, seeds=range(8))
    windy = mean_db("rain", rain__enabled=True, rain__rain_mm_h=8.0, wind__speed_ms=8.0, seeds=range(8))
    assert windy > still + 6


def test_rain_continuum_carries_the_far_drops():
    """the dense rain is rendered in closed form, so how many drops are rendered one by one doesn't change the level"""
    few = mean_db("rain", rain__enabled=True, rain__rain_mm_h=8.0, rain__grains_hz=50.0, room__indoor=False)
    many = mean_db("rain", rain__enabled=True, rain__rain_mm_h=8.0, rain__grains_hz=5000.0, room__indoor=False)
    assert few == pytest.approx(many, abs=1.0)


def hf_share(x):
    f, p = signal.welch(x, FS, nperseg=2048)
    return p[f > 1000].sum() / p.sum()


def test_indoor_traffic_is_darker():
    """the windows' insulation climbs with frequency (mass law) and then levels off (leaks, the glazing's coincidence
    dip): the facade takes more off the highs, but only so much more"""
    out = render(scene(traffic__enabled=True, babble__enabled=False, clatter__enabled=False, room__indoor=False))["traffic"]
    ins = render(scene(traffic__enabled=True, babble__enabled=False, clatter__enabled=False, room__indoor=True,
                       room__p_door_open=0.0))["traffic"]

    def band_db(x, lo, hi):
        f, p = signal.welch(x, FS, nperseg=2048)
        return 10 * np.log10(p[(f >= lo) & (f < hi)].sum())

    loss_low = band_db(out, 50, 300) - band_db(ins, 50, 300)
    loss_high = band_db(out, 1000, 4000) - band_db(ins, 1000, 4000)
    assert loss_low + 3 < loss_high < loss_low + 10


def test_closed_kitchen_is_darker():
    o = render(scene(kitchen__enabled=True, kitchen__open=True, babble__enabled=False, clatter__enabled=False))["kitchen"]
    c = render(scene(kitchen__enabled=True, kitchen__open=False, babble__enabled=False, clatter__enabled=False))["kitchen"]
    assert hf_share(c) < 0.5 * hf_share(o)


def test_presets_and_random_scenes():
    from random_cafe.presets import PRESETS, preset, random_scene
    for name in PRESETS:
        st = render(preset(name, duration_s=2.0, fs=FS, seed=3), fake_talkers(20), fake_load)
        assert st and all(np.all(np.isfinite(x)) for x in st.values()), name
    assert set(render(preset("rain-heavy", duration_s=2.0, fs=FS))) == {"rain"}
    assert set(render(preset("traffic-light", duration_s=2.0, fs=FS))) == {"traffic"}
    a, b = random_scene(5, duration_s=2.0, fs=FS), random_scene(5, duration_s=2.0, fs=FS)
    assert a == b
    layer_sets = {tuple(n for n in ("babble", "clatter", "kitchen", "traffic", "rain", "wind") if getattr(random_scene(s), n).enabled)
                  for s in range(40)}
    assert len(layer_sets) > 10 and () not in layer_sets


def fake_hrtf():
    """six directions; the ear facing the source gets it louder and earlier, the far ear quieter and later"""
    dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    ir = np.zeros((6, 2, 32))
    ir[:, :, 2] = 1.0
    ir[2, 1, 2], ir[3, 0, 2] = 0.2, 0.2  # +y is the left ear's side
    delay = np.zeros((6, 2))
    delay[2, 1] = delay[3, 0] = 10.0
    return spatial.Hrtf(ir, dirs, delay, FS, FS)


def test_binaural():
    """with an HRTF every stem is (2, n), deterministic, near the mono level, and a source on the listener's left is
    louder and earlier in the left ear; layers stay independent"""
    h = fake_hrtf()
    kw = dict(kitchen__enabled=True, traffic__enabled=True, rain__enabled=True, wind__enabled=True)
    a = render(scene(**kw), fake_talkers(), fake_load, hrtf=h)
    b = render(scene(**kw), fake_talkers(), fake_load, hrtf=h)
    mono = render(scene(**kw), fake_talkers(), fake_load)
    for k in a:
        assert a[k].shape == (2, 4 * FS) and np.array_equal(a[k], b[k])
        assert abs(rms_db(a[k]) - rms_db(mono[k])) < 4, k
    part = render(scene(**{**kw, "rain__enabled": False}), fake_talkers(), fake_load, hrtf=h)
    for k in part:
        assert np.array_equal(part[k], a[k]), k
    rx = spatial.Receiver((0, 0, 1.2), FS, FS, h, yaw_deg=90.0)  # facing +y, so the left is -x
    y = rx.place_static(np.r_[1.0, np.zeros(99)], (-5.0, 0.0, 1.2))
    assert np.sum(y[0] ** 2) > 10 * np.sum(y[1] ** 2) and np.argmax(np.abs(y[0])) < np.argmax(np.abs(y[1]))


def test_doppler():
    """a tone source approaching then receding is heard sharp, then flat"""
    f, v = 1000.0, 20.0
    n = 4 * FS
    x = np.sin(2 * np.pi * f * np.arange(n) / FS)
    y = spatial.place_moving(x, lambda t: np.stack([v * (t - 2.0), np.full_like(t, 5.0), np.zeros_like(t)], 1), (0, 0, 0), FS)

    def freq(seg):
        z = np.flatnonzero(np.diff(np.signbit(seg)))
        return (len(z) - 1) / 2 / ((z[-1] - z[0]) / FS)

    approach, recede = freq(y[int(0.3 * FS):int(1.3 * FS)]), freq(y[int(2.7 * FS):int(3.7 * FS)])
    assert approach > f * 1.04 and recede < f * 0.96


def test_directivity_darkens_from_behind():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(FS)
    assert hf_share(spatial.head_shadow(x, np.pi, FS)) < hf_share(spatial.head_shadow(x, 0.0, FS))


def test_floor_positions_uniform_by_area():
    rng = np.random.default_rng(0)
    r = np.array([np.linalg.norm(spatial.floor_position(rng, (1.0, 10.0), 0.0)[:2]) for _ in range(20000)])
    # uniform over area: the share within radius R is (R^2 - 1) / (100 - 1)
    assert np.mean(r < 5.0) == pytest.approx(24 / 99, abs=0.02)


def test_rain_overhead():
    """an umbrella overhead is heard over the ground rain; a roof adds to the rain indoors; a ceiling takes its flat
    loss and more above its knee"""
    from random_cafe.layers import rain
    kw = dict(rain__enabled=True, babble__enabled=False, clatter__enabled=False, room__indoor=False)
    assert mean_db("rain", rain__overhead="cafe-umbrella", **kw) > mean_db("rain", **kw) + 2
    kw = dict(rain__enabled=True, babble__enabled=False, clatter__enabled=False)
    assert mean_db("rain", rain__overhead="polycarbonate", rain__overhead_height_m=2.5, **kw) > mean_db("rain", **kw) + 1
    x = np.random.default_rng(0).standard_normal(4 * FS)
    y = rain.ceiling(x, FS, 10.0, 300.0, 6.0)
    assert rms_db(y) < rms_db(x) - 10 and hf_share(y) < 0.5 * hf_share(x)
