# random-cafe

Seeded background scenes, rendered as **separate stems** at one microphone: café babble from real talkers, clatter, a kitchen, passing traffic, rain and recorded beds.

**What it's for.** It's built to augment audio data for machine-learning training: mixing realistic, labelled background noise under speech or singing for denoising, ASR robustness, voice analysis and similar work. Every stem is exact, so you know how much of each frame is signal and how much is background. It can be used for any purpose, though: sound design, listening tests, games, or just background noise to work to.

**Scope.** The name stays, but scenes aren't limited to cafés: the café is the first environment. Its layers (talkers, impacts, machinery, traffic, weather, rooms, facades) are general building blocks for others, such as streets, markets, offices, transport and homes.

**Licence.** The code is MIT. **No audio ships with it: bring your own datasets.** You point it at corpora you have downloaded, and their licences stay yours to respect (see [Datasets](#datasets)).

```python
from random_cafe import Scene, render, mix
from random_cafe.presets import preset, random_scene

stems = render(preset("rainy-cafe", seed=7), talkers={"spk1": ["a.wav", "b.wav"], "spk2": ["c.wav"]})
background = mix(stems)                      # add your target voice yourself, at the SNR you want

scene = random_scene(seed=42)                # for augmentation: layers and intensities drawn from the seed
```

```
python -m random_cafe out/ --preset street-cafe --talkers talkers.tsv --seed 7
python -m random_cafe out/ --random --seed 42 --talkers talkers.tsv
python -m random_cafe out/ --kitchen closed --tables 8 12 --rain 25 --traffic 5
```

## What is in a scene

| Stem | Model |
|---|---|
| `babble` | Tables of 2–4 talkers from **your** speech corpora, taking turns with conversational gaps and overlaps. Talkers face their table, so some are heard from behind (head-shadow directivity). Tables are spread uniformly over the floor area, so with 1/r gain each distance ring carries similar power: many quiet far talkers, a few near ones. |
| `clatter` | Cups, cutlery and glass as modal grains (a click exciting a few decaying resonances), synthesised, so licence-free. Its rate follows the number of occupied tables in the presets. |
| `kitchen` | Extractor fan (rumble plus motor hum), dish and pan clatter, espresso steam bursts. **Open**, it shares the café's room. **Closed**, it's heard through a wall or door: quieter and duller. |
| `traffic` | Vehicles passing on a lane: tyre noise that brightens and grows with speed, plus engine harmonics. Some are prime movers (10%; 25% in `traffic-heavy`): trucks and buses with deep diesel harmonics, more level, and sometimes an air brake. Distance and Doppler follow from geometry. Also a distant road bed. Indoors, it comes in through the facade: the windows (about 18–28 dB of insulation, more at high frequencies) and, when the front door is open, the opening (about 8–15 dB, barely filtered), then reverberates in the café. Audible indoors, but a full house masks it. |
| `rain` | Drop ticks and Minnaert bubbles (f = 3.26 / R) on the ground outdoors. Indoors: ticks on a window pane, plus the outdoor rain through the facade (windows, and the front door when open). Louder and denser with the rain rate. |
| `bed` | Your recorded room tone or ambience, looped from a random offset. |

**Levels are physical within the model.** Sources come out at unit level at 1 m. Distance, walls, rain rate, traffic density and occupied tables change how loud each stem is, and a per-layer calibration puts the default scene near its nominal levels. Traffic and rain are calibrated outdoors, so indoors they're quieter by the facade's insulation. The facade (window loss, whether the front door is open) is drawn once per scene, so traffic and rain always agree on it. `mix(stems, target_dbfs=...)` normalises the whole mix if you want a fixed level.

**One room per scene.** Every indoor source shares its reverberation time, and each gets its own tail at a direct-to-reverberant ratio that falls with distance. Each layer draws from its own seeded generator, so turning one layer on or off never changes another.

## Limiting cases and mixes

Every layer switches on and off, so single sources (`rain-light`, `rain-heavy`, `traffic-light`, `traffic-heavy`, `kitchen-only`) are as easy as full scenes. The presets:

| Preset | Scene |
|---|---|
| `cafe-quiet` | 1–2 occupied tables; soft voices, careful cutlery (a high-SNR scene) |
| `cafe-busy` | 8–14 tables, open kitchen |
| `cafe-open-kitchen` / `cafe-closed-kitchen` | the kitchen in the room, or behind a wall |
| `kitchen-only` | just the kitchen |
| `rain-light` / `rain-heavy` | 1.5 / 30 mm/h, nothing else, outdoors (`--indoor` for rain on the windows) |
| `traffic-light` / `traffic-heavy` | 3 / 40 vehicles per minute, nothing else, outdoors (`--indoor` for traffic through the wall) |
| `street-cafe` | outdoors: tables, clatter, passing traffic |
| `rainy-cafe` | indoors: tables, rain on the windows, a little traffic outside |

`presets.SNR_DB` gives the speech-to-background SNR range where each preset is realistic: a quiet café 15–30 dB, a busy café or street 0–15 dB.

For training data, `random_scene(seed)` switches each layer on with probability 0.5 (at least one on). It picks indoor or outdoor and an open or closed kitchen, draws table counts, rain rate (0.5–50 mm/h) and traffic density (1–60 per minute) log-uniformly, and jitters each layer's level by ±6 dB. One seed gives one reproducible scene.

## Datasets

None are bundled; you download them yourself. These are the ones we plan to add built-in loaders for, chosen because their licences allow any use with attribution:

| Dataset | Use | Licence |
|---|---|---|
| LibriTTS-R | talkers (English) | CC BY 4.0 |
| VCTK 0.92 | talkers (English, many accents) | CC BY 4.0 |
| Common Voice | talkers (many languages, own microphones) | CC0 (agree not to identify speakers) |
| Multilingual LibriSpeech | talkers (8 European languages) | CC BY 4.0 |
| Zeroth-Korean | talkers (Korean) | CC BY 4.0 |
| speechocean762 | talkers (English, children and adults) | CC BY 4.0 |
| Samrómur Children | talkers (Icelandic, children 4–17) | CC BY 4.0 |
| MUSAN (noise subset) | recorded beds | CC BY 4.0 (check per-file terms for its music) |
| RIRS_NOISES (OpenSLR 28) | recorded beds and room responses | Apache 2.0 (check its source databases) |
| FSD50K | café, crowd, traffic and street beds | per clip: keep CC0 / CC BY only |

Never try to identify speakers in any corpus, and credit each one you use.

## Credits

The fractional delay, the Brown–Duda head shadow (used here in reverse, for talker directivity), the equal-power distance rings and the impact/bubble grains follow [umbrella-rain](https://github.com/hannahsq/umbrella-rain).

## Status and next steps

0.1, a scaffold with simple, tested models. `python calibrate.py` recomputes the level calibration after a model change. Next:
- built-in loaders for the datasets above;
- room simulation with pyroomacoustics (per-source room responses from room geometry), as an option beside the statistical shared room;
- noise variation per draw (speed perturbation, tilt, layering);
- moving talkers;
- a Lombard option (raised level, f0 and F1 for talkers in noise);
- **rain**: revisit against umbrella-rain's best demo parameters (the current drops sound over-processed, "deep fried"); on hold until those command lines are found;
- a **wind** layer: diffuse low-frequency turbulence, the character the distant traffic bed already has, for outdoor scenes;
- licence-filtered recorded-bed lists for FSD50K and MUSAN.
