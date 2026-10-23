"""Recorded beds (room tone, espresso machine, ambience) from caller-supplied files: each looped from a random
offset with short crossfades. No audio ships with this package; check each recording's licence."""
from collections.abc import Callable

import numpy as np

from ..scene import Scene


def render(rng: np.random.Generator, scene: Scene, load: Callable[[str, int], np.ndarray]) -> np.ndarray:
    fs, n = scene.fs, int(scene.duration_s * scene.fs)
    x = load(scene.bed.paths[int(rng.integers(len(scene.bed.paths)))], fs)
    fade = min(int(0.05 * fs), x.size // 4)
    out, pos = np.zeros(n), -int(rng.integers(0, max(x.size - 1, 1)))
    ramp = np.linspace(0, 1, fade)
    seg = x.copy()
    seg[:fade] *= ramp
    seg[-fade:] *= ramp[::-1]
    while pos < n:
        a, b = max(pos, 0), min(pos + seg.size, n)
        out[a:b] += seg[a - pos:b - pos]
        pos += seg.size - fade
    return out
