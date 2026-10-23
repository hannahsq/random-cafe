"""Render the listening demos into listen/ (gitignored): every preset plus a few variants and random scenes as mixes
(listen/scenes) and stems (listen/stems/<scene>), and, with --speech-root, read speech mixed with some of the scenes
at set SNRs (listen/speech). With --sofa, the scenes are also rendered binaurally (listen/binaural, for
headphones). Prints each render's speed in multiples of real time.

  python demos.py --talkers talkers.tsv [--speech-root LibriTTS_R/train-clean-100] [--sofa HRIR.sofa [--yaw 90]]
                  [--seed 7] [--duration 15]

talkers.tsv: speaker<TAB>audio path, for the babble. The speech targets are two readers of LibriTTS-R train-clean-100
(19, female; 26, male) so that babble from dev-clean never contains the target talker. SNR is the speech RMS over its
active frames (within 35 dB of its loudest 20 ms) against the background's RMS; every speech mix is scaled to -26 dBFS."""
import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent / "src"))
from random_cafe import mix, render, spatial  # noqa: E402
from random_cafe.presets import LAYERS, preset, random_scene  # noqa: E402
from random_cafe.render import default_load  # noqa: E402

FS = 16000
OUT = Path(__file__).parent / "listen"


def scenes(seed: int, dur: float) -> dict:
    kw = dict(seed=seed, duration_s=dur, fs=FS)
    s = {name: preset(name, **kw) for name in ("cafe-quiet", "cafe-busy", "cafe-open-kitchen", "cafe-closed-kitchen")}
    for door, p in (("shut", 0.0), ("open", 1.0)):
        x = preset("cafe-busy", **kw)
        x.traffic.enabled, x.room.p_door_open = True, p
        s[f"cafe-busy-traffic_front-door-{door}"] = x
    for name in ("street-cafe", "rainy-cafe", "rain-light", "rain-heavy", "traffic-light", "traffic-heavy",
                 "traffic-lights", "street-cafe-rain", "wind"):
        s[name] = preset(name, **kw)
    for r in (11, 12, 13, 14):
        x = random_scene(r, duration_s=dur, fs=FS)
        on = "+".join(n for n in LAYERS if getattr(x, n).enabled)
        kitchen = f"_kitchen-{'open' if x.kitchen.open else 'closed'}" if x.kitchen.enabled else ""
        s[f"random-seed{r}_{'indoor' if x.room.indoor else 'outdoor'}_{on}{kitchen}"] = x
    return s


def write(path: Path, x: np.ndarray) -> None:
    wavfile.write(path, FS, np.clip(x, -1, 1).astype(np.float32).T)


def connected(root: Path, reader: str, dur: float) -> np.ndarray:
    """consecutive sentences of one chapter, with short natural pauses, to dur seconds, at unit RMS"""
    chapter = sorted((root / reader).iterdir())[0]
    parts, n = [], 0
    for f in sorted(chapter.glob("*.wav")):
        x = default_load(str(f), FS)
        parts += [x, np.zeros(int(np.random.default_rng(len(parts)).uniform(0.15, 0.45) * FS))]
        n += x.size
        if n > dur * FS:
            break
    y = np.concatenate(parts)[: int(dur * FS)]
    return y / np.sqrt(np.mean(y ** 2))


def active_rms(x: np.ndarray) -> float:
    w = int(0.02 * FS)
    fr = np.sqrt(np.mean(x[: x.size // w * w].reshape(-1, w) ** 2, axis=1))
    return float(np.sqrt(np.mean(fr[fr > fr.max() * 10 ** (-35 / 20)] ** 2)))


def at_snr(speech: np.ndarray, bg: np.ndarray, snr_db: float) -> np.ndarray:
    x = speech + bg * active_rms(speech) / np.sqrt(np.mean(bg ** 2)) * 10 ** (-snr_db / 20)
    return x * 10 ** (-26 / 20) / np.sqrt(np.mean(x ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--talkers", required=True, help="TSV of speaker, path")
    ap.add_argument("--speech-root", type=Path, help="LibriTTS-R train-clean-100, for the speech mixes")
    ap.add_argument("--sofa", help="HRIRs (.sofa) for binaural versions of the scenes (needs h5py)")
    ap.add_argument("--yaw", type=float, default=90.0, help="binaural: the way the head faces (90: the street side)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--duration", type=float, default=15.0)
    a = ap.parse_args()
    talkers = {}
    for row in csv.reader(open(a.talkers, encoding="utf-8"), delimiter="\t"):
        talkers.setdefault(row[0], []).append(row[1])
    # every utterance loaded up front: they recur across scenes, and reading files is not what the timing measures
    cache = {p: default_load(p, FS) for paths in talkers.values() for p in paths}
    load = lambda path, fs: cache[path]  # noqa: E731
    shutil.rmtree(OUT, ignore_errors=True)
    for d in ("scenes", "stems", "speech", "binaural"):
        (OUT / d).mkdir(parents=True)
    mixes = {}
    for k, (name, scene) in enumerate(scenes(a.seed, a.duration).items(), 1):
        t = time.perf_counter()
        stems = render(scene, talkers, load)
        dt = time.perf_counter() - t
        mixes[name] = mix(stems)
        write(OUT / "scenes" / f"{k:02d}_{name}.wav", mixes[name])
        (OUT / "stems" / f"{k:02d}_{name}").mkdir()
        for stem, x in stems.items():
            write(OUT / "stems" / f"{k:02d}_{name}" / f"{stem}.wav", x)
        print(f"{k:02d} {name:60s} {a.duration / dt:6.0f}x real time")
    if a.sofa:  # the same scenes for headphones: same events, heard by two ears
        hrtf = spatial.Hrtf.from_sofa(a.sofa, FS)
        for k, (name, scene) in enumerate(scenes(a.seed, a.duration).items(), 1):
            scene.listener_yaw_deg = a.yaw
            t = time.perf_counter()
            x = mix(render(scene, talkers, load, hrtf=hrtf))
            write(OUT / "binaural" / f"{k:02d}_{name}.wav", x)
            print(f"{k:02d} {name:60s} {a.duration / (time.perf_counter() - t):6.0f}x real time, binaural")
    if a.speech_root:
        k = 1
        for tname, reader in (("female_reader19", "19"), ("male_reader26", "26")):
            s = connected(a.speech_root, reader, a.duration)
            write(OUT / "speech" / f"{k:02d}_{tname}_clean.wav", s * 10 ** (-26 / 20))
            k += 1
            for snr in (20, 10, 5, 0):  # one scene across the useful range
                write(OUT / "speech" / f"{k:02d}_{tname}_cafe-busy_snr{snr:02d}.wav", at_snr(s, mixes["cafe-busy"], snr))
                k += 1
            for name in ("cafe-quiet", "cafe-closed-kitchen", "rainy-cafe", "street-cafe", "street-cafe-rain",
                         "traffic-heavy", "rain-heavy", "cafe-busy-traffic_front-door-open"):  # each at 10 dB, a typical working level
                write(OUT / "speech" / f"{k:02d}_{tname}_{name}_snr10.wav", at_snr(s, mixes[name], 10))
                k += 1
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
