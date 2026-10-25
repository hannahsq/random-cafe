# random-cafe

Seeded background scenes, rendered as **separate stems** at one microphone: café babble from real talkers, clatter, a kitchen, passing traffic, rain, wind and recorded beds.

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
| `traffic` | A simulated street: one or two lanes (the second running the other way), vehicles following each other by the Intelligent Driver Model, each driver with a desired speed, a time headway and an aggressiveness (how hard they accelerate and brake, how late they change up); the arrival rate drifts, so platoons form; optionally a pedestrian crossing whose lights stop the traffic. Each vehicle's speed and acceleration drive its engine through a gearbox: cars are petrol or diesel automatics (idling at 650–850 rpm, slipping as they pull away), prime movers (10%; 25% in `traffic-heavy`; 12–40 t, laden or not) are manual diesels that go flat out whenever they accelerate, double-clutched: about 1.3 s off the throttle at each shift (from a recording) while the engine falls (or, shifting down, is blipped) to the next gear's speed, skipping gears if they have gained speed meanwhile; sometimes an air-brake dump as they stop. Every engine has a full-load torque curve (a big diesel's governor cuts it steeply near rated speed). Each vehicle draws an engine layout (I3, I4, boxer 4, I6, V6, odd-fire V6, flat- or cross-plane V8), a turbo or not and a stock or high-flow exhaust. The exhaust is blowdown pulses at the layout's firing angles, bank by bank (so a cross-plane V8 burbles), soft at light throttle and sharp at full throttle in petrol engines, alike at every load in diesels, smeared by a turbine, each cylinder's pulse a little early or late and a little stronger or weaker and no two firings alike (so lines fill in between the firing harmonics, as recorded), through a silencer (for a stock petrol car, the measured response of a 2023 BMW X1's: resonances at ~110, 205, 307, 377 Hz and up to ~8.7 kHz, a cliff above ~390 Hz, a pass-band at 1–1.4 kHz; its low resonances driven harder as the throttle opens; the harmonics rolling off by ~2 kHz over white-ish source noise that fills out to 8 kHz); then the flow noise made at the tailpipe and in the silencer itself, after its attenuation (broadband, roughly flat per octave to a few kHz, power as gas speed⁸ like jet noise at first, then levelling off to grow like the tones (fitted to the X1 revving to 6000 rpm), as measured on a straight-piped and a stock Hayabusa and a Cappuccino; a modern consumer car's stock exhaust 6 dB quieter, fitted to a 2023 BMW X1 1.5 L); the intake is its own source (inlet pulses through a resonant plastic airbox, throttle hiss for petrol); a turbo, sized to the engine's rated power (shaft speed as power^-0.5: ~9 kHz for a big truck, ~25 kHz for a kei car), adds the compressor's blade-pass whistle over a broadband pedestal and, for a car's splitter-bladed compressor, a line at half the blade-pass only ~3 dB weaker (all measured on recordings), louder with the shaft's speed (as boost², i.e. speed⁴), which follows the torque (the turbine runs on the exhaust's enthalpy and the compressor's work also goes with the flow, so the shaft traces the fuelling: arcs that droop near the top of each gear), after a lag, and drops at each shift (at 16 kHz the whistle of a small car turbo near full boost is above Nyquist); the block adds mechanical noise, much of it in clicks: each cylinder's combustion and its valves seating, the cylinders and the cycles differing, so a diesel idle ticks at the cycle rate (as a recorded truck idle does) as well as the firing rate; tyre noise grows steeply with speed. A vehicle is heard from two places, its exhaust at the back and the rest from the engine bay, each with its own distance and Doppler, so the two comb against each other as the revs sweep (as in the X1 recording). Also a distant road bed. Indoors, it comes in through the facade: the windows (about 18–28 dB of insulation, more at high frequencies) and, when the front door is open, the opening (about 8–15 dB, barely filtered), then reverberates in the café. Audible indoors, but a full house masks it. |
| `rain` | Drops on the ground outdoors, sized by Marshall–Palmer: each an impact click plus a splash coloured by its contact time and, at 44.1 and 48 kHz, a short burst near 11 kHz at the first instant of contact (the bump at 10–13 kHz every rain recording shows; fitted), and large drops landing in puddles ring a Minnaert bubble (f = 3.26 / R). The nearest drops are rendered one by one, up to a budget; the dense rain further out is noise of the same colour at the power the rings carry. The rain rate drifts in slow gusts. Indoors: drops on a window pane, plus the outdoor rain through the facade (windows, and the front door when open). Wind drives rain onto the glass (a pane facing the wind takes wind speed / fall speed of the ground's flux; the facade's angle to the wind is drawn per scene), so a windward window is a rain bed of its own. Louder and denser with the rain rate. |
| `wind` | Outdoors, turbulence buffeting the microphone (dynamic pressure, so level grows as speed²; flat up to speed / eddy size, then about f^-4.3 as the microphone's body averages the smaller eddies away; eddy size and the device's high-pass default to a phone's, fitted to a recording) plus the rush of wind around buildings and trees, both in gusts. Indoors, only the rush, through the facade. The wind speed is scene weather: rain reads it even when this layer is off. |
| `bed` | Your recorded room tone or ambience, looped from a random offset. |

**Levels are physical within the model.** Sources come out at unit level at 1 m. Distance, walls, rain rate, traffic density and occupied tables change how loud each stem is, and a per-layer calibration puts the default scene near its nominal levels. Traffic, rain and wind are calibrated outdoors, so indoors they're quieter by the facade's insulation. The facade (window loss, whether the front door is open) is drawn once per scene, so traffic and rain always agree on it. `mix(stems, target_dbfs=...)` normalises the whole mix if you want a fixed level.

**Sample rates.** 16 kHz by default (speech models); 44.1 and 48 kHz for full-band models, diagnostics and tuning (`Scene(fs=48000)`, `--fs 48000`): a small turbo's whistle near full boost, the top of rain and hiss. Every layer's level agrees across these rates (tested), so one calibration serves them all.

**Binaural previews.** For listening (to judge where things are, say), `render(scene, ..., hrtf=spatial.Hrtf.from_sofa("subject.sofa", fs))` or `--sofa subject.sofa` renders every stem for two ears at the microphone, facing `--yaw` degrees (90 by default: the street side, so traffic passes left to right in front). Point sources use the nearest measured HRIR, with the interaural delay; moving vehicles switch HRIRs frame by frame; nearby rain drops are grouped into direction sectors; room tails, dense rain, wind and the distant road bed are diffuse (coherent across the head below 700 Hz, decorrelated above); outdoor sound indoors arrives from the facade's opening. The HRIRs are diffuse-field equalised, so binaural stems keep the mono levels. Mono stems are bit-identical with or without this. Reading SOFA needs `h5py` (`pip install random-cafe[binaural]`); nothing else does.

**One room per scene.** Every indoor source shares its reverberation time, and each gets its own tail at a direct-to-reverberant ratio that falls with distance. Each layer draws from its own seeded generator, so turning one layer on or off never changes another.

## Limiting cases and mixes

Every layer switches on and off, so single sources (`rain-light`, `rain-heavy`, `traffic-light`, `traffic-heavy`, `kitchen-only`, `wind`) are as easy as full scenes. The presets:

| Preset | Scene |
|---|---|
| `cafe-quiet` | 1–2 occupied tables; soft voices, careful cutlery (a high-SNR scene) |
| `cafe-busy` | 8–14 tables, open kitchen |
| `cafe-open-kitchen` / `cafe-closed-kitchen` | the kitchen in the room, or behind a wall |
| `kitchen-only` | just the kitchen |
| `rain-light` / `rain-heavy` | 1.5 / 30 mm/h, nothing else, outdoors (`--indoor` for rain on the windows) |
| `traffic-light` / `traffic-heavy` | 3 / 40 vehicles per minute, nothing else, outdoors (`--indoor` for traffic through the wall) |
| `traffic-lights` | 20 vehicles per minute stopping and starting at a pedestrian crossing within 30 m |
| `street-cafe` | outdoors: tables, clatter, passing traffic |
| `street-cafe-rain` | the same, in 8 mm/h rain and a 5 m/s wind (the regulars stay out in it) |
| `wind` | 6 m/s wind, nothing else, outdoors |
| `rainy-cafe` | indoors: tables, rain on the windows, a little traffic outside |

`presets.SNR_DB` gives the speech-to-background SNR range where each preset is realistic: a quiet café 15–30 dB, a busy café or street 0–15 dB.

For training data, `random_scene(seed)` switches each layer on with probability 0.5 (at least one on). It picks indoor or outdoor and an open or closed kitchen, draws table counts, rain rate (0.5–50 mm/h), traffic density (1–60 per minute) and wind speed (1–12 m/s) log-uniformly, puts lights on the street 30% of the time, and jitters each layer's level by ±6 dB. One seed gives one reproducible scene.

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

Traffic flow follows the Intelligent Driver Model (Treiber, Hennecke & Helbing 2000). The fractional delay, the Brown–Duda head shadow (used here in reverse, for talker directivity), the equal-power distance rings, the gust envelope, the rain model (drop sizes and fall speeds, click and splash grains, Minnaert bubbles) and the SOFA HRIR handling follow [umbrella-rain](https://github.com/hannahsq/umbrella-rain).

## Status and next steps

0.1, a scaffold with simple, tested models. `python calibrate.py` recomputes the level calibration after a model change, and `python demos.py --talkers talkers.tsv [--speech-root DIR] [--sofa HRIR.sofa]` renders listening demos into `listen/` (gitignored; binaural versions with `--sofa`), printing each render's speed. Next:
- built-in loaders for the datasets above;
- room simulation with pyroomacoustics (per-source room responses from room geometry), as an option beside the statistical shared room;
- noise variation per draw (speed perturbation, tilt, layering);
- moving talkers;
- a Lombard option (raised level, f0 and F1 for talkers in noise);
- **rain**: compare by ear with umbrella-rain's best demos once their command lines are found; the window pane's level is a heuristic (`Rain.pane_db`);
- speed: dense traffic (about 15× real time at 40 vehicles a minute) and large babble scenes are the slowest layers;
- traffic, next stages: tuning engines and drivers by ear; more vehicle kinds (mopeds under 50 cc; sports bikes: 1–4 cylinders, 0.125–1.4 L, high rpm; loud low-cylinder-count choppers; electric cars; buses pulling in); CVT transmissions (belt or chain whine at the pulley speeds, gliding as the ratio changes); trucks coasting through their shift gaps (the flow model doesn't know the gearbox); transmission whine (an automatic's oil pump at engine speed, planetary gear meshes at each gear's own orders, torque-converter slip, the final drive's ring and pinion at road speed: gear meshes sit at non-integer engine orders, which tells them from the engine in a recording); motorcycle chain noise (sprocket teeth x sprocket speed, plus slap); exhaust resonances (a fixed-frequency line excited by the flow under load, ~1.0-1.3 kHz on the X1, rising as the gas heats: sound speed as sqrt(temperature)); a turbo's power drawn per vehicle (tuned cars run bigger turbos, which whistle lower), not only from its displacement; separate intake and exhaust positions (front and rear); overrun pops and blow-off valves;
- rain, next stages: more surfaces for it to fall on: roofs (corrugated iron, corrugated polycarbonate, tiles; a concrete building's roof is near silent), and outdoors heavy tarps, vinyl awnings and umbrellas; downpipes and drains; wet roads (tyre noise rising on a wet surface, splashes through standing water); a window pane that rings at its own modes
- wind, next stages: noise on the microphone set by its shelter (little in a closed café, more with the doors open), howling and whistling around openings and edges; the buffeting's corner following the gusts (it follows the mean speed, so a gust is louder but not brighter)
- other environmental sources (some may be best from recordings): magnetic ballasts of fluorescent tubes, ceiling and pedestal fans, gas burners, pedestrian-crossing signals (the safe-to-cross sounds)
- licence-filtered recorded-bed lists for FSD50K and MUSAN.
