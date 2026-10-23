"""Named scenes for the limiting cases and their mixes, and random_scene() for augmentation: every layer switched on
or off and its intensity drawn from the seed, so a training set covers everything from one quiet source to a busy,
rainy street café."""
from collections.abc import Callable

import numpy as np

from .scene import Scene


def _only(s: Scene, *layers: str) -> Scene:
    for name in ("babble", "clatter", "kitchen", "traffic", "rain"):
        getattr(s, name).enabled = name in layers
    return s


def _tables(s: Scene, lo: int, hi: int) -> None:
    s.babble.tables = (lo, hi)
    s.clatter.rate_hz = 0.4 * (lo + hi) / 2  # clatter follows how many tables are occupied


def cafe_quiet(s):
    _only(s, "babble", "clatter")
    _tables(s, 1, 2)
    return s


def cafe_busy(s):
    _only(s, "babble", "clatter", "kitchen")
    _tables(s, 8, 14)
    return s


def cafe_open_kitchen(s):
    _only(s, "babble", "clatter", "kitchen")
    s.kitchen.open = True
    return s


def cafe_closed_kitchen(s):
    s = cafe_open_kitchen(s)
    s.kitchen.open = False
    return s


def kitchen_only(s):
    return _only(s, "kitchen")


def rain_light(s):
    s.room.indoor = False  # just the source: standing outside
    _only(s, "rain")
    s.rain.rain_mm_h = 1.5
    return s


def rain_heavy(s):
    s.room.indoor = False  # just the source: standing outside
    _only(s, "rain")
    s.rain.rain_mm_h = 30.0
    return s


def traffic_light(s):
    s.room.indoor = False  # just the source: standing outside
    _only(s, "traffic")
    s.traffic.vehicles_per_min = 3.0
    return s


def traffic_heavy(s):
    s.room.indoor = False  # just the source: standing outside
    _only(s, "traffic")
    s.traffic.vehicles_per_min = 40.0
    s.traffic.lane_distance_m = (3.0, 15.0)
    return s


def street_cafe(s):
    _only(s, "babble", "clatter", "traffic")
    s.room.indoor = False
    _tables(s, 3, 8)
    s.traffic.vehicles_per_min = 15.0
    return s


def rainy_cafe(s):
    _only(s, "babble", "clatter", "rain", "traffic")
    _tables(s, 4, 9)
    s.rain.rain_mm_h = 8.0
    s.traffic.vehicles_per_min = 6.0
    return s


PRESETS: dict[str, Callable[[Scene], Scene]] = {
    "cafe-quiet": cafe_quiet, "cafe-busy": cafe_busy, "cafe-open-kitchen": cafe_open_kitchen,
    "cafe-closed-kitchen": cafe_closed_kitchen, "kitchen-only": kitchen_only, "rain-light": rain_light,
    "rain-heavy": rain_heavy, "traffic-light": traffic_light, "traffic-heavy": traffic_heavy,
    "street-cafe": street_cafe, "rainy-cafe": rainy_cafe,
}


def preset(name: str, **scene_kw) -> Scene:
    return PRESETS[name](Scene(**scene_kw))


def random_scene(seed: int, p_layer: float = 0.5, **scene_kw) -> Scene:
    """Every layer on with probability p_layer (at least one on), indoor or outdoor, kitchen open or closed, and each
    intensity drawn log-uniformly across its realistic range, all from `seed`."""
    rng = np.random.default_rng([seed, 0xCAFE])
    s = Scene(seed=seed, **scene_kw)
    names = ["babble", "clatter", "kitchen", "traffic", "rain"]
    on = rng.uniform(size=len(names)) < p_layer
    if not on.any():
        on[rng.integers(len(names))] = True
    _only(s, *[n for n, o in zip(names, on) if o])
    s.room.indoor = bool(rng.uniform() < 0.75)
    s.kitchen.open = bool(rng.uniform() < 0.5)
    lo = int(rng.integers(1, 10))
    _tables(s, lo, lo + int(rng.integers(0, 5)))
    s.rain.rain_mm_h = float(np.exp(rng.uniform(np.log(0.5), np.log(50.0))))
    s.traffic.vehicles_per_min = float(np.exp(rng.uniform(np.log(1.0), np.log(60.0))))
    for n in names:
        getattr(s, n).level_db += rng.uniform(-6.0, 6.0)
    return s
