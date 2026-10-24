"""python -m random_cafe OUT_DIR [--preset NAME | --random] [--seed N] [--duration S] [layer switches]
[--talkers LIST.tsv] [--bed FILE ...] [--sofa HRIR.sofa [--yaw DEG]]: writes one WAV per stem and mix.wav (binaural
with --sofa). LIST.tsv: speaker<TAB>audio path, one utterance per line. Switches apply on top of a preset."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from . import Scene, mix, render, spatial
from .presets import PRESETS, preset, random_scene


def main():
    ap = argparse.ArgumentParser(prog="random_cafe")
    ap.add_argument("out")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--random", action="store_true", help="draw the layers and intensities from --seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--fs", type=int, default=16000, help="sample rate: 16000, or 44100 / 48000 for full band")
    ap.add_argument("--outdoor", action="store_true")
    ap.add_argument("--indoor", action="store_true")
    ap.add_argument("--no-babble", action="store_true")
    ap.add_argument("--no-clatter", action="store_true")
    ap.add_argument("--kitchen", choices=["open", "closed"])
    ap.add_argument("--tables", type=int, nargs=2, metavar=("MIN", "MAX"))
    ap.add_argument("--traffic", type=float, metavar="VEHICLES_PER_MIN", help="0 = none")
    ap.add_argument("--rain", type=float, metavar="MM_H", help="0 = none")
    ap.add_argument("--wind", type=float, metavar="M_S", help="wind speed (it also drives rain onto windows); 0 = none")
    ap.add_argument("--talkers", help="TSV of speaker, path")
    ap.add_argument("--bed", nargs="*", default=[])
    ap.add_argument("--sofa", help="HRIRs (.sofa) for a binaural render (needs h5py)")
    ap.add_argument("--yaw", type=float, help="binaural: the way the head faces, degrees (90: the street side)")
    a = ap.parse_args()
    kw = dict(duration_s=a.duration, fs=a.fs)
    s = random_scene(a.seed, **kw) if a.random else preset(a.preset, seed=a.seed, **kw) if a.preset else Scene(seed=a.seed, **kw)
    if a.outdoor:
        s.room.indoor = False
    if a.indoor:
        s.room.indoor = True
    if a.no_babble:
        s.babble.enabled = False
    if a.no_clatter:
        s.clatter.enabled = False
    if a.kitchen:
        s.kitchen.enabled, s.kitchen.open = True, a.kitchen == "open"
    if a.tables:
        s.babble.tables = tuple(a.tables)
    if a.traffic is not None:
        s.traffic.enabled, s.traffic.vehicles_per_min = a.traffic > 0, a.traffic
    if a.rain is not None:
        s.rain.enabled, s.rain.rain_mm_h = a.rain > 0, a.rain
    if a.wind is not None:
        s.wind.enabled, s.wind.speed_ms = a.wind > 0, a.wind
    s.bed.paths = tuple(a.bed)
    if a.yaw is not None:
        s.listener_yaw_deg = a.yaw
    talkers = {}
    if a.talkers:
        for row in csv.reader(open(a.talkers, encoding="utf-8"), delimiter="\t"):
            talkers.setdefault(row[0], []).append(row[1])
    stems = render(s, talkers or None, hrtf=spatial.Hrtf.from_sofa(a.sofa, a.fs) if a.sofa else None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, x in {**stems, "mix": mix(stems)}.items():
        wavfile.write(out / f"{name}.wav", a.fs, np.clip(x, -1, 1).astype(np.float32).T)
    print(", ".join(f"{k} {20 * np.log10(np.sqrt(np.mean(v ** 2)) + 1e-12):.1f} dBFS" for k, v in stems.items()) or "no stems")


if __name__ == "__main__":
    main()
