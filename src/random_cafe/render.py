"""render(scene) -> stems. Each layer draws from its own generator (seeded by the scene seed and the layer's name),
so switching one layer on or off never changes another. One room per scene: every indoor source shares its T60.

Levels are physical within the model: sources come out at unit level at 1 m, so distance, walls, rain rate, traffic
density and the number of occupied tables all change how loud a stem is. CAL maps each layer to scene.reference_dbfs
+ level_db for its default settings (median over seeds), so a default scene's stems sit near their nominal levels
and a busier, wetter or closed-off scene is louder or quieter than that. Recorded beds have no physical level and
are set to their nominal level directly."""
import zlib
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy.signal import resample_poly

from . import spatial
from .layers import babble, bed, clatter, kitchen, rain, traffic
from .scene import Scene

# 1 / (median stem RMS of the default scene for that layer at level_db 0), from calibrate.py; traffic and rain are
# calibrated outdoors, so indoors they come out quieter by the wall
CAL = {"babble": 0.4478, "clatter": 22.8, "kitchen": 4.534, "traffic": 22.08, "rain": 6.944}


def default_load(path: str, fs: int) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != fs:
        g = np.gcd(int(sr), fs)
        x = resample_poly(x, fs // g, int(sr) // g)
    return x.astype(np.float64)


def _rng(scene: Scene, name: str) -> np.random.Generator:
    return np.random.default_rng([scene.seed, zlib.crc32(name.encode())])


def _gain(scene: Scene, layer: str, level_db: float) -> float:
    return CAL[layer] * 10 ** ((scene.reference_dbfs + level_db) / 20)


def render(scene: Scene | None = None, talkers: Mapping[str, Sequence[str]] | None = None,
           load: Callable[[str, int], np.ndarray] = default_load) -> dict[str, np.ndarray]:
    """Stems of one scene. talkers: {speaker: [audio paths]} for babble (skipped without it); load(path, fs) -> mono
    float audio."""
    scene = scene or Scene()
    room = spatial.SharedRoom(_rng(scene, "room"), scene.room.t60_s, scene.room.drr_db, scene.fs)
    stems = {}
    if scene.babble.enabled and talkers:
        stems["babble"] = babble.render(_rng(scene, "babble"), scene, room, talkers, load) * _gain(scene, "babble", scene.babble.level_db)
    if scene.clatter.enabled:
        stems["clatter"] = clatter.render(_rng(scene, "clatter"), scene, room) * _gain(scene, "clatter", scene.clatter.level_db)
    if scene.kitchen.enabled:
        stems["kitchen"] = kitchen.render(_rng(scene, "kitchen"), scene, room) * _gain(scene, "kitchen", scene.kitchen.level_db)
    if scene.traffic.enabled:
        stems["traffic"] = traffic.render(_rng(scene, "traffic"), scene) * _gain(scene, "traffic", scene.traffic.level_db)
    if scene.rain.enabled:
        stems["rain"] = rain.render(_rng(scene, "rain"), scene, room) * _gain(scene, "rain", scene.rain.level_db)
    if scene.bed.paths:
        x = bed.render(_rng(scene, "bed"), scene, load)
        stems["bed"] = x * 10 ** ((scene.reference_dbfs + scene.bed.level_db) / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-12)
    return stems


def mix(stems: Mapping[str, np.ndarray], target_dbfs: float | None = None) -> np.ndarray:
    """The sum of the stems; with target_dbfs, scaled so the mix has that RMS (stems keep their relative levels)."""
    if not stems:
        return np.zeros(0)
    x = sum(stems.values())
    if target_dbfs is not None:
        x = x * 10 ** (target_dbfs / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-12)
    return x
