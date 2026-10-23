"""Print CAL for render.py: 1 / median stem RMS of each layer's default scene (level_db 0, reference 0 dBFS) over
seeds. Babble uses synthetic talkers (harmonic tones at unit RMS), which is what the per-utterance normalisation makes
any corpus look like in level. Indoor layers are calibrated indoors, traffic, rain and wind outdoors (so indoors they are
quieter by the wall). Run after changing a layer's model: python calibrate.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
import importlib  # noqa: E402

R = importlib.import_module("random_cafe.render")  # the module (the package exports the function under this name)
from random_cafe.scene import Scene  # noqa: E402


def talkers():
    return {f"s{i}": [f"s{i}/u{j}" for j in range(4)] for i in range(40)}


def load(path, fs):
    i, j = (int(p[1:]) for p in path.split("/"))
    t = np.arange(int((1.5 + 0.4 * j) * fs)) / fs
    return sum(np.sin(2 * np.pi * k * (100 + 7 * i) * t) / k for k in range(1, 20)) * np.hanning(t.size)


R.CAL = {k: 1.0 for k in R.CAL}
rms = {k: [] for k in R.CAL}
OUTDOOR = {"traffic", "rain", "wind"}  # outdoor sources: calibrated as heard outdoors, so indoors they come out quieter
for seed in range(8):
    for indoor in (True, False):
        s = Scene(seed=seed, duration_s=8.0, reference_dbfs=0.0)
        s.room.indoor = indoor
        s.kitchen.enabled = s.traffic.enabled = s.rain.enabled = s.wind.enabled = True
        for name in ("babble", "clatter", "kitchen", "traffic", "rain", "wind"):
            getattr(s, name).level_db = 0.0
        for k, x in R.render(s, talkers(), load).items():
            if (k in OUTDOOR) != indoor:
                rms[k].append(np.sqrt(np.mean(x ** 2)))
print("CAL = {" + ", ".join(f'"{k}": {1 / np.median(v):.4g}' for k, v in rms.items()) + "}")
