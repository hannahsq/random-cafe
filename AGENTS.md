# AGENTS.md — random-cafe

Seeded background scenes rendered as separate stems at one microphone: babble from real talkers, clatter, kitchen, traffic, rain, wind, recorded beds. It's for augmenting ML audio data, though usable for anything. Code MIT; no audio ships (bring your own corpora).

**Scope:** not limited to cafés. The café is the first environment, and new environments should reuse and generalise the layers (talkers, impacts, machinery, traffic, weather, rooms, facades) rather than fork them.

## Layout

- `src/random_cafe/scene.py`: dataclasses for every parameter (`Scene`, `Room`, one per layer).
- `src/random_cafe/render.py`: `render(scene, talkers, load) -> stems`, `mix()`, and the per-layer calibration `CAL`.
- `src/random_cafe/spatial.py`: geometry (fractional delay, 1/r, Doppler), head-shadow directivity, floor positions, `SharedRoom`, `Facade`, `transmission`; `Receiver` (the microphone, or two ears with an `Hrtf`) and `diffuse` for binaural previews.
- `src/random_cafe/layers/`: one module per stem (`noise.py`: coloured and OU noise they share). Traffic is simulated first (`traffic.simulate`, IDM), then sounded (`drive`: gearbox, rpm and load; `sound`).
- `src/random_cafe/presets.py`: named scenes and `random_scene(seed)`.
- `calibrate.py`: recomputes `CAL`.
- `demos.py`: renders listening demos (scenes, stems, speech at set SNRs, single vehicles, binaural with `--sofa`) into `listen/` (gitignored), with each render's speed.
- `tests/test_render.py`.

## Running

- Tests: `python -m pytest -q` (pytest finds `src` via `pyproject.toml`). The local `.venv` has the test, audio and binaural extras: `pip install -e .[test,audio,binaural]`.
- Demo: `python -m random_cafe out/ --preset rainy-cafe --talkers talkers.tsv` (TSV: speaker, path).
- Sample rates: 16 kHz default; 44.1 and 48 kHz supported (`scene.SAMPLE_RATES`, tested to keep every layer's level, so `CAL` is shared). Calibrate at 16 kHz.
- `data/` (gitignored) holds reference recordings for eval only: no licence, never commit them.

## Invariants (tested; keep them)

- **Deterministic:** the same scene and seed give bit-identical stems.
- **Layers are independent.** Each layer draws only from its own generator (`_rng(scene, name)`), including its room tails. Turning one layer on or off never changes another. Shared scene state (the room's T60, the facade) is drawn once from its own generator.
- **Levels are physical within the model.** Sources come out at unit level at 1 m. Distance, walls, rates and counts change loudness. Don't normalise stems individually. **After changing a layer's model, run `python calibrate.py`** and paste the result into `CAL` (traffic, rain and wind are calibrated outdoors).
- **Mono doesn't depend on binaural.** Layers place sound through the `Receiver`; binaural-only randomness (decorrelation, rain-drop azimuths) comes from `rng.spawn`, which doesn't advance the layer's stream, so mono stems are bit-identical with or without an HRTF. `h5py` stays an optional extra.
- **Facade losses are insulation** (outdoor minus indoor level). Don't add separate spreading loss on top.

## Open items

- Rain was rebuilt on umbrella-rain's ground model (the old one sounded over-processed). Compare by ear with umbrella-rain's best demos once Hannah has found their command lines. The window pane's radiation gain (`Rain.pane_db`) and the wind's buffeting and rush constants are heuristics, not yet checked by ear.

- Exhaust (on hold for stock and straight-piped references): a Suzuki Cappuccino dyno pull (`data/cappuccino_2.flac`, aftermarket ~75 mm exhaust, mic ~2 m behind the outlet, on axis) shows firing harmonics relative to the firing order at about -15 to -18 dB at 400-800 Hz, -17 to -21 at 0.8-1.6 kHz, then ~12 dB/octave. That is a shelf, not our steep fourth-order low-pass: even the high-flow silencer is ~20-30 dB too dark above 800 Hz. The second firing harmonic matches the first; half-orders rise to near the firing ones above cycle order ~10 at high rpm (cylinders' pulses arriving a few degrees apart; ours vary only in amplitude); and the block's noise should be partly locked to engine orders (the truck idle shows the same). Plan once references arrive: a silencer shelf (corner, floor, gentle roll-off) fitted per exhaust type, pulse arrival offsets with cycle-to-cycle jitter, order-locked block noise; allow for on-axis beaming above ~1.5 kHz.

## Conventions

- New corpora must have licences that allow any use with attribution (CC0, CC BY, Apache). Never ship audio. Never try to identify speakers.
- Keep models simple and named for their physics, and mark heuristics as heuristics. Credit borrowed methods (umbrella-rain) in docstrings and the README.
- This repository is an optional submodule of voice-analyser (`external/random-cafe`). Keep its interface (`render`, `mix`, `Scene`, presets) stable.
