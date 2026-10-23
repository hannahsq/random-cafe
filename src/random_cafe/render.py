"""render(scene) -> stems. Each layer draws from its own generator (seeded by the scene seed and the layer's name),
so switching one layer on or off never changes another. One room per scene: every indoor source shares its T60."""
import zlib
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy.signal import resample_poly

from . import spatial
from .layers import babble, bed, clatter, rain, traffic
from .scene import Scene


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


def _level(x: np.ndarray, scene: Scene, level_db: float) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2))
    return x if rms == 0 else x * 10 ** ((scene.reference_dbfs + level_db) / 20) / rms


def render(scene: Scene | None = None, talkers: Mapping[str, Sequence[str]] | None = None,
           load: Callable[[str, int], np.ndarray] = default_load) -> dict[str, np.ndarray]:
    """Stems of one scene, each at its level relative to scene.reference_dbfs. talkers: {speaker: [audio paths]} for
    babble (skipped without it); load(path, fs) -> mono float audio."""
    scene = scene or Scene()
    room = spatial.SharedRoom(_rng(scene, "room"), scene.room.t60_s, scene.room.drr_db, scene.fs)
    stems = {}
    if scene.babble.enabled and talkers:
        stems["babble"] = _level(babble.render(_rng(scene, "babble"), scene, room, talkers, load), scene, scene.babble.level_db)
    if scene.clatter.enabled:
        stems["clatter"] = _level(clatter.render(_rng(scene, "clatter"), scene, room), scene, scene.clatter.level_db)
    if scene.traffic.enabled:
        stems["traffic"] = _level(traffic.render(_rng(scene, "traffic"), scene), scene, scene.traffic.level_db)
    if scene.rain.enabled:
        stems["rain"] = _level(rain.render(_rng(scene, "rain"), scene, room), scene, scene.rain.level_db)
    if scene.bed.paths:
        stems["bed"] = _level(bed.render(_rng(scene, "bed"), scene, load), scene, scene.bed.level_db)
    return stems


def mix(stems: Mapping[str, np.ndarray]) -> np.ndarray:
    return sum(stems.values()) if stems else np.zeros(0)
