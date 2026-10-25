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

- Rain was rebuilt on umbrella-rain's ground model (the old one sounded over-processed). Compare by ear with umbrella-rain's best demos once Hannah has found their command lines. The window pane's radiation gain (`Rain.pane_db`) and the wind's rush constants are heuristics, not yet checked by ear. The buffeting's shape is fitted to a phone recording (eddy size and high-pass in `Wind`); its absolute level and the rush are not (rain masked the rush).

- Exhaust (references in `data/`, eval only): Hayabusa mk3 straight-piped and stock, a Suzuki Cappuccino on a dyno, a 2023 BMW X1 1.5 L turbo I3 (stationary revving 1 m from the tailpipe, in mono and stereo; three pulls). The stock petrol exhaust is fitted to the X1 by ear and by measurement through an emulation of the camera's automatic gain (`demos.agc`; the recording's level range is ~12 dB because of it): the silencer's response (`traffic.SHELF_DB`, `SHELF_PEAKS`), steep pulse fronts, cylinder-to-cylinder and firing-to-firing differences, the low resonances driven by the throttle (revving on the spot, load is the crank's acceleration), source noise, and exhaust and engine bay heard from two places. Its overall level (`X1_GAIN`) keeps stock petrol cars as loud as before the fit. Open: diesels, trucks and high-flow exhausts keep the simpler silencer (no references yet); the X1's fixed ~1.0-1.3 kHz whistle under load and its pulls (on the exhaust's axis) are not yet used; the throttle boost is capped at half load beyond the revving it was fitted to. Scratch tools for this (order tracking, A/B switches) are not in the repo; `demos.py` renders the revving comparison (`listen/vehicles/petrol_I3-1.5L-turbo_revving-1m*.wav`).

## Conventions

- New corpora must have licences that allow any use with attribution (CC0, CC BY, Apache). Never ship audio. Never try to identify speakers.
- Keep models simple and named for their physics, and mark heuristics as heuristics. Credit borrowed methods (umbrella-rain) in docstrings and the README.
- This repository is an optional submodule of voice-analyser (`external/random-cafe`). Keep its interface (`render`, `mix`, `Scene`, presets) stable.
