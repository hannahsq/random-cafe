# random-cafe

Seeded background scenes, rendered as **separate stems** at one microphone: café babble from real talkers, clatter, passing traffic, rain, and recorded beds. Built for training and testing voice and speech models under realistic noise. Because every stem is exact, you know how much of each frame is voice and how much is background.

```python
from random_cafe import Scene, render, mix

scene = Scene(duration_s=10, fs=16000, seed=7)
scene.traffic.enabled = True          # passing vehicles + distant road bed
scene.rain.enabled = True             # drops on the ground, or on the window indoors
stems = render(scene, talkers={"spk1": ["a.wav", "b.wav"], "spk2": [...]})
background = mix(stems)               # add your target voice yourself
```

```
python -m random_cafe out/ --seed 7 --traffic --rain 8 --talkers talkers.tsv
```

## What is in a scene

| Stem | Model |
|---|---|
| `babble` | Tables of 2–4 talkers from **your** speech corpora, taking turns with conversational gaps and overlaps. Talkers face their table, so some are heard from behind (head-shadow directivity). Tables are spread uniformly over the floor area, so with 1/r gain each distance ring carries similar power: many quiet far talkers, a few near ones. |
| `clatter` | Cups, cutlery and glass as modal grains (a click exciting a few decaying resonances), synthesised, so licence-free. |
| `traffic` | Vehicles passing on a lane: tyre noise that brightens and grows with speed, plus engine harmonics. Distance and Doppler follow from geometry. Also a distant road bed. Indoors, everything passes through the wall or window (mass-law low-pass plus loss). |
| `rain` | Drop ticks and Minnaert bubbles (f = 3.26 / R) on the ground outdoors. Indoors: ticks on a window pane, plus the outdoor rain through the wall. |
| `bed` | Your recorded room tone or ambience, looped from a random offset. |

One room per scene: every indoor source shares its reverberation time, and each gets its own tail at a direct-to-reverberant ratio that falls with distance. Each layer draws from its own seeded generator, so turning one layer on or off never changes another.

## Data and licences

No audio ships with this package. You supply talkers and beds, and their licences stay yours to respect: use corpora whose terms allow your purpose, and never try to identify speakers. The code is MIT.

## Credits

The fractional delay, the Brown–Duda head shadow (used here in reverse, for talker directivity), the equal-power distance rings and the impact/bubble grains follow [umbrella-rain](https://github.com/hannahsq/umbrella-rain).

## Status and next steps

0.1, a scaffold with simple, tested models. Next:
- room simulation with pyroomacoustics (per-source RIRs from room geometry), as an option beside the statistical shared room;
- noise variation per draw (speed perturbation, tilt, layering);
- moving talkers;
- a Lombard option (raised level, f0 and F1 for talkers in noise);
- recorded beds from licence-filtered FSD50K / MUSAN lists.
