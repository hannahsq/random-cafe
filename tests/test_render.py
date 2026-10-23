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
    """the windows' insulation climbs with frequency (mass law), so the facade takes far more off the highs"""
    out = render(scene(traffic__enabled=True, babble__enabled=False, clatter__enabled=False, room__indoor=False))["traffic"]
    ins = render(scene(traffic__enabled=True, babble__enabled=False, clatter__enabled=False, room__indoor=True,
                       room__p_door_open=0.0))["traffic"]

    def band_db(x, lo, hi):
        f, p = signal.welch(x, FS, nperseg=2048)
        return 10 * np.log10(p[(f >= lo) & (f < hi)].sum())

    loss_low = band_db(out, 50, 300) - band_db(ins, 50, 300)
    loss_high = band_db(out, 1000, 4000) - band_db(ins, 1000, 4000)
    assert loss_high > loss_low + 8


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
