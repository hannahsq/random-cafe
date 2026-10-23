"""Babble from real talkers: tables of 2-4 talkers drawn from a caller-supplied pool, each table taking turns
(conversational gaps, sometimes overlapping), talkers facing their table so the microphone hears some of them from
behind (head-shadow directivity), every talker in the shared room at its own distance."""
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from .. import spatial
from ..scene import Scene


def render(rng: np.random.Generator, scene: Scene, room: spatial.SharedRoom,
           talkers: Mapping[str, Sequence[str]], load: Callable[[str, int], np.ndarray],
           rx: spatial.Receiver) -> np.ndarray:
    cfg, fs = scene.babble, scene.fs
    n = int(scene.duration_s * fs)
    out = rx.zeros()
    names = sorted(talkers)
    k_tables = int(rng.integers(cfg.tables[0], cfg.tables[1] + 1))
    sizes = [int(rng.integers(cfg.per_table[0], cfg.per_table[1] + 1)) for _ in range(k_tables)]
    left, fitted = len(names), []
    for s in sizes:  # a small pool: tables fill until the talkers run out (each talker at one table only)
        s = min(s, left)
        if s < 1:
            break
        fitted.append(s)
        left -= s
    picks = iter(rng.permutation(len(names)))
    for size in fitted:
        centre = spatial.floor_position(rng, cfg.radius_m, 0.0, scene.mic[:2])
        members = [names[next(picks)] for _ in range(size)]
        seats = [centre + 0.6 * np.array([np.cos(a), np.sin(a), 0.0]) for a in 2 * np.pi * (np.arange(size) + rng.uniform()) / size]
        tracks = [np.zeros(n + fs) for _ in members]
        t, last = rng.uniform(-3.0, 0.0), -1
        while t < scene.duration_s:  # turn-taking: a different talker each turn, gaps drawn (negative: overlap)
            who = int(rng.choice([i for i in range(size) if i != last] or [0]))
            files = talkers[members[who]]
            x = load(files[int(rng.integers(len(files)))], fs)
            x = x / (np.sqrt(np.mean(x ** 2)) + 1e-12)  # corpora differ in level: every utterance at unit RMS
            spatial.add_fractional(tracks[who], x, max(t, -x.size / fs) * fs)
            t += x.size / fs + rng.normal(cfg.gap_s[0], cfg.gap_s[1])
            last = who
        for seat, track in zip(seats, tracks):
            seat = seat + np.array([0, 0, 1.15])  # seated mouth height
            facing = (centre - seat)[:2] / max(np.linalg.norm((centre - seat)[:2]), 1e-9)
            theta = spatial.facing_angle(seat, np.r_[facing, 0.0], scene.mic)
            x = spatial.head_shadow(track[:n], theta, fs) * 10 ** (rng.uniform(*cfg.lombard_db) / 20)
            direct = rx.place_static(x, seat)
            out += room.apply(direct, spatial.distance(seat, scene.mic), rng) if scene.room.indoor else direct
    return out
