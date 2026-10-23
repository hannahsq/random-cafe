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
    """switching traffic on must not change the other stems"""
    a = render(scene(), fake_talkers(), fake_load)
    b = render(scene(traffic__enabled=True), fake_talkers(), fake_load)
    for k in a:
        assert np.array_equal(a[k], b[k])
    assert "traffic" in b


def test_stem_levels_and_mix():
    st = render(scene(traffic__enabled=True, rain__enabled=True), fake_talkers(), fake_load)
    s = scene()
    for k, x in st.items():
        assert x.shape == (4 * FS,) and np.all(np.isfinite(x))
    ref = 10 ** (s.reference_dbfs / 20)
    assert np.sqrt(np.mean(st["babble"] ** 2)) == pytest.approx(ref * 10 ** (s.babble.level_db / 20), rel=1e-6)
    assert np.allclose(mix(st), sum(st.values()))


def hf_share(x):
    f, p = signal.welch(x, FS, nperseg=2048)
    return p[f > 1000].sum() / p.sum()


def test_indoor_traffic_is_darker():
    out = render(scene(traffic__enabled=True, babble__enabled=False, clatter__enabled=False, room__indoor=False))["traffic"]
    ins = render(scene(traffic__enabled=True, babble__enabled=False, clatter__enabled=False, room__indoor=True))["traffic"]
    assert hf_share(ins) < 0.2 * hf_share(out)


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
