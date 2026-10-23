"""python -m random_cafe OUT_DIR [--seed N] [--duration S] [--outdoor] [--traffic] [--rain MM_H] [--talkers LIST.tsv]
[--bed FILE ...]: writes one WAV per stem and mix.wav. LIST.tsv: speaker<TAB>audio path, one utterance per line."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from . import Scene, mix, render


def main():
    ap = argparse.ArgumentParser(prog="random_cafe")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--outdoor", action="store_true")
    ap.add_argument("--traffic", action="store_true")
    ap.add_argument("--rain", type=float, default=0.0, help="mm/h; 0 = no rain")
    ap.add_argument("--talkers", help="TSV of speaker, path")
    ap.add_argument("--bed", nargs="*", default=[])
    a = ap.parse_args()
    s = Scene(duration_s=a.duration, fs=a.fs, seed=a.seed)
    s.room.indoor = not a.outdoor
    s.traffic.enabled = a.traffic
    s.rain.enabled, s.rain.rain_mm_h = a.rain > 0, a.rain
    s.bed.paths = tuple(a.bed)
    talkers = {}
    if a.talkers:
        for row in csv.reader(open(a.talkers, encoding="utf-8"), delimiter="\t"):
            talkers.setdefault(row[0], []).append(row[1])
    stems = render(s, talkers or None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, x in {**stems, "mix": mix(stems)}.items():
        wavfile.write(out / f"{name}.wav", a.fs, np.clip(x, -1, 1).astype(np.float32))
    print(", ".join(f"{k} {20 * np.log10(np.sqrt(np.mean(v ** 2)) + 1e-12):.1f} dBFS" for k, v in stems.items()))


if __name__ == "__main__":
    main()
