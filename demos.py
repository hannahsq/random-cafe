"""Render the listening demos into listen/ (gitignored): every preset plus a few variants and random scenes as mixes
(listen/scenes) and stems (listen/stems/<scene>); with --speech-root, read speech mixed with some of the scenes at
set SNRs (listen/speech); single vehicles cruising past, pulling away (gently and hard) and stopping, outdoors 8 m
from the kerb (listen/vehicles), one of each engine layout pulling away both ways (listen/vehicles/engines), and a
few engines part by part (listen/components/<engine>_<manoeuvre>/<part>.wav; all.wav is their sum). With --sofa, the
scenes and vehicles are also rendered binaurally (listen/binaural, listen/vehicles/binaural, for headphones). With
--fs 44100 or 48000, full-band versions go to listen/fs44100 or listen/fs48000. Prints each render's speed in
multiples of real time.

  python demos.py --talkers talkers.tsv [--speech-root LibriTTS_R/train-clean-100] [--sofa HRIR.sofa [--yaw 90]]
                  [--seed 7] [--duration 15] [--fs 16000]

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
from random_cafe import Scene, mix, render, spatial  # noqa: E402
from random_cafe.layers import traffic  # noqa: E402
from random_cafe.presets import LAYERS, preset, random_scene  # noqa: E402
from random_cafe.render import _gain, _rng, default_load  # noqa: E402

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


def trajectory(manoeuvre: str, heavy: bool, t: np.ndarray, dur: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x, speed and acceleration of one vehicle: cruising past mid-clip, pulling away from a stop line 8 m before the
    microphone after 3 s (free-road IDM acceleration, an average driver; hard: the most aggressive driver's), or
    braking to stop 8 m past it at mid-clip and idling there."""
    vmax, a, b = (45 / 3.6, 0.7, 1.5) if heavy else (50 / 3.6, 1.75, 2.25)
    if manoeuvre == "hard-pull-away":
        a = 0.9 if heavy else 2.5
    if manoeuvre == "cruise":
        return vmax * (t - dur / 2), np.full(t.size, vmax), np.zeros(t.size)
    if manoeuvre == "stop":
        tb = dur / 2 - vmax / b  # braking starts
        tau = np.clip(t - tb, 0.0, vmax / b)
        v = np.where(t < tb, vmax, vmax - b * tau)
        x = 8.0 - vmax ** 2 / (2 * b) + np.where(t < tb, vmax * (t - tb), vmax * tau - b * tau ** 2 / 2)
        return x, v, np.where((t >= tb) & (t < dur / 2), -b, 0.0)
    v, acc = np.zeros(t.size), np.zeros(t.size)
    for j in range(1, t.size):
        acc[j] = a * (1 - (v[j - 1] / vmax) ** 4) if t[j] >= 3.0 else 0.0
        v[j] = v[j - 1] + acc[j] * traffic.DT
    return -8.0 + np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * traffic.DT)]), v, acc


# the most common engine of each kind, then one of each layout (and turbo, exhaust) pulling away
TYPICAL = {"petrol": dict(layout="I4", turbo=False, high_flow=False), "diesel": dict(layout="I4", turbo=True, high_flow=False),
           "heavy": dict(layout="I6", turbo=True, high_flow=False, mass=40000.0)}  # a fully laden prime mover
ENGINES = [("petrol", "I3", True, False), ("petrol", "I4", True, False), ("petrol", "I4", True, True),
           ("petrol", "boxer4", False, False), ("petrol", "I6", True, False), ("petrol", "V6", False, False),
           ("petrol", "V6-odd", False, False), ("petrol", "V8-cross", False, True), ("petrol", "V8-flat", False, True),
           ("diesel", "V6", True, False), ("heavy", "V8-cross", True, False, 40000.0)]


def vehicle(kind: str, engine: dict, manoeuvre: str, dur: float, hrtf=None, parts=traffic.PARTS) -> np.ndarray:
    """one vehicle on an otherwise empty street (no road bed), 6 dB below the level it has in the street scenes (so
    that the loudest, alone and near, do not clip)"""
    scene = Scene(seed=1, duration_s=dur, fs=FS)
    scene.room.indoor, scene.traffic.bed_db = False, -np.inf
    t = np.arange(-10.0, dur + traffic.DT, traffic.DT)
    x, v, acc = trajectory(manoeuvre, kind == "heavy", t, dur)
    car = dict(kind=kind, aggr=1.0 if manoeuvre.startswith("hard") else 0.5, on=np.ones(t.size, bool), x=x, y=8.0, v=v, acc=acc, **engine)
    rx = spatial.Receiver(scene.mic, FS, int(dur * FS), hrtf, scene.listener_yaw_deg)
    room = spatial.SharedRoom(_rng(scene, "room"), scene.room.t60_s, scene.room.drr_db, FS)
    facade = spatial.Facade(_rng(scene, "facade"), scene.room, room, FS, rx)
    y = traffic.render(_rng(scene, "traffic"), scene, facade, rx, sim=(t, [car]), parts=parts)
    return 0.5 * y * _gain(scene, "traffic", scene.traffic.level_db)


def vehicles(dur: float, hrtf=None) -> dict:
    out = {}
    for kind, engine in TYPICAL.items():
        for manoeuvre in ("cruise", "pull-away", "hard-pull-away", "stop"):
            name = f"{kind}_{engine['layout']}{'-turbo' if engine['turbo'] else ''}_{manoeuvre}"
            out[name] = vehicle(kind, engine, manoeuvre, dur, hrtf)
    for kind, layout, turbo, high_flow, *mass in ENGINES:
        engine = dict(layout=layout, turbo=turbo, high_flow=high_flow, **(dict(mass=mass[0]) if mass else {}))
        for manoeuvre in ("pull-away", "hard-pull-away"):
            name = f"engines/{kind}_{layout}{'-turbo' if turbo else ''}{'_high-flow' if high_flow else ''}_{manoeuvre}"
            out[name] = vehicle(kind, engine, manoeuvre, dur, hrtf)
    return out


# engines heard part by part (listen/components/<engine>_<manoeuvre>/<part>.wav, with all.wav: they add up to it)
COMPONENTS = {"petrol_V8-cross_high-flow": ("petrol", dict(layout="V8-cross", turbo=False, high_flow=True)),
              "petrol_I3-1.0L-turbo": ("petrol", dict(layout="I3", turbo=True, high_flow=False, litres=1.0)),
              "petrol_I4-2.0L-turbo_high-flow": ("petrol", dict(layout="I4", turbo=True, high_flow=True, litres=2.0)),
              "heavy_I6-turbo": ("heavy", TYPICAL["heavy"])}


def components(dur: float) -> dict:
    out = {}
    for name, (kind, engine) in COMPONENTS.items():
        for manoeuvre in ("cruise", "pull-away", "hard-pull-away"):
            out[f"{name}_{manoeuvre}/all"] = vehicle(kind, engine, manoeuvre, dur)
            for part in traffic.PARTS:
                if part == "throttle" and kind != "petrol" or part == "air-brakes" and kind != "heavy" or part == "turbo" and not engine["turbo"]:
                    continue  # silent for this engine
                out[f"{name}_{manoeuvre}/{part}"] = vehicle(kind, engine, manoeuvre, dur, parts=(part,))
    return out


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
    global FS
    ap = argparse.ArgumentParser()
    ap.add_argument("--talkers", required=True, help="TSV of speaker, path")
    ap.add_argument("--speech-root", type=Path, help="LibriTTS-R train-clean-100, for the speech mixes")
    ap.add_argument("--sofa", help="HRIRs (.sofa) for binaural versions of the scenes (needs h5py)")
    ap.add_argument("--yaw", type=float, default=90.0, help="binaural: the way the head faces (90: the street side)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--fs", type=int, default=FS, help="sample rate; 44100 or 48000 for full-band demos (in listen/fs44100 ...)")
    a = ap.parse_args()
    FS = a.fs
    talkers = {}
    for row in csv.reader(open(a.talkers, encoding="utf-8"), delimiter="\t"):
        talkers.setdefault(row[0], []).append(row[1])
    # every utterance loaded up front: they recur across scenes, and reading files is not what the timing measures
    cache = {p: default_load(p, FS) for paths in talkers.values() for p in paths}
    load = lambda path, fs: cache[path]  # noqa: E731
    root = OUT if FS == 16000 else OUT / f"fs{FS}"  # other rates beside the 16 kHz demos, not over them
    for d in ("scenes", "stems", "speech", "binaural", "vehicles", "components"):
        shutil.rmtree(root / d, ignore_errors=True)
    for d in ("scenes", "stems", "speech", "binaural", "vehicles/engines", "vehicles/binaural/engines"):
        (root / d).mkdir(parents=True)
    mixes = {}
    for k, (name, scene) in enumerate(scenes(a.seed, a.duration).items(), 1):
        t = time.perf_counter()
        stems = render(scene, talkers, load)
        dt = time.perf_counter() - t
        mixes[name] = mix(stems)
        write(root / "scenes" / f"{k:02d}_{name}.wav", mixes[name])
        (root / "stems" / f"{k:02d}_{name}").mkdir()
        for stem, x in stems.items():
            write(root / "stems" / f"{k:02d}_{name}" / f"{stem}.wav", x)
        print(f"{k:02d} {name:60s} {a.duration / dt:6.0f}x real time")
    t = time.perf_counter()
    for name, x in vehicles(a.duration).items():
        write(root / "vehicles" / f"{name}.wav", x)
    print(f"single vehicles: {2 * len(ENGINES) + 12} at {(2 * len(ENGINES) + 12) * a.duration / (time.perf_counter() - t):6.0f}x real time")
    t = time.perf_counter()
    comp = components(a.duration)
    for name, x in comp.items():
        (root / "components" / name).parent.mkdir(parents=True, exist_ok=True)
        write(root / "components" / f"{name}.wav", x)
    print(f"components: {len(comp)} at {len(comp) * a.duration / (time.perf_counter() - t):6.0f}x real time")
    if a.sofa:
        for name, x in vehicles(a.duration, spatial.Hrtf.from_sofa(a.sofa, FS)).items():
            write(root / "vehicles" / "binaural" / f"{name}.wav", x)
    if a.sofa:  # the same scenes for headphones: same events, heard by two ears
        hrtf = spatial.Hrtf.from_sofa(a.sofa, FS)
        for k, (name, scene) in enumerate(scenes(a.seed, a.duration).items(), 1):
            scene.listener_yaw_deg = a.yaw
            t = time.perf_counter()
            x = mix(render(scene, talkers, load, hrtf=hrtf))
            write(root / "binaural" / f"{k:02d}_{name}.wav", x)
            print(f"{k:02d} {name:60s} {a.duration / (time.perf_counter() - t):6.0f}x real time, binaural")
    if a.speech_root:
        k = 1
        for tname, reader in (("female_reader19", "19"), ("male_reader26", "26")):
            s = connected(a.speech_root, reader, a.duration)
            write(root / "speech" / f"{k:02d}_{tname}_clean.wav", s * 10 ** (-26 / 20))
            k += 1
            for snr in (20, 10, 5, 0):  # one scene across the useful range
                write(root / "speech" / f"{k:02d}_{tname}_cafe-busy_snr{snr:02d}.wav", at_snr(s, mixes["cafe-busy"], snr))
                k += 1
            for name in ("cafe-quiet", "cafe-closed-kitchen", "rainy-cafe", "street-cafe", "street-cafe-rain",
                         "traffic-heavy", "rain-heavy", "cafe-busy-traffic_front-door-open"):  # each at 10 dB, a typical working level
                write(root / "speech" / f"{k:02d}_{tname}_{name}_snr10.wav", at_snr(s, mixes[name], 10))
                k += 1
    print(f"wrote {root}")


if __name__ == "__main__":
    main()
