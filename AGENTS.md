# AGENTS.md — random-cafe

Seeded background scenes rendered as separate stems at one microphone: babble from real talkers, clatter, kitchen, traffic, rain, recorded beds. It's for augmenting ML audio data, though usable for anything. Code MIT; no audio ships (bring your own corpora).

**Scope:** not limited to cafés. The café is the first environment, and new environments should reuse and generalise the layers (talkers, impacts, machinery, traffic, weather, rooms, facades) rather than fork them.

## Layout

- `src/random_cafe/scene.py`: dataclasses for every parameter (`Scene`, `Room`, one per layer).
- `src/random_cafe/render.py`: `render(scene, talkers, load) -> stems`, `mix()`, and the per-layer calibration `CAL`.
- `src/random_cafe/spatial.py`: geometry (fractional delay, 1/r, Doppler), head-shadow directivity, floor positions, `SharedRoom`, `Facade`, `transmission`.
- `src/random_cafe/layers/`: one module per stem.
- `src/random_cafe/presets.py`: named scenes and `random_scene(seed)`.
- `calibrate.py`: recomputes `CAL`.
- `tests/test_render.py`.

## Running

- Tests: `python -m pytest -q` (pytest finds `src` via `pyproject.toml`).
- Demo: `python -m random_cafe out/ --preset rainy-cafe --talkers talkers.tsv` (TSV: speaker, path).

## Invariants (tested; keep them)

- **Deterministic:** the same scene and seed give bit-identical stems.
- **Layers are independent.** Each layer draws only from its own generator (`_rng(scene, name)`), including its room tails. Turning one layer on or off never changes another. Shared scene state (the room's T60, the facade) is drawn once from its own generator.
- **Levels are physical within the model.** Sources come out at unit level at 1 m. Distance, walls, rates and counts change loudness. Don't normalise stems individually. **After changing a layer's model, run `python calibrate.py`** and paste the result into `CAL` (traffic and rain are calibrated outdoors).
- **Facade losses are insulation** (outdoor minus indoor level). Don't add separate spreading loss on top.

## Open items

- Rain sounds over-processed next to umbrella-rain's demos: revisit with its best demo parameters (Hannah is finding the command lines) before tuning further.

## Conventions

- New corpora must have licences that allow any use with attribution (CC0, CC BY, Apache). Never ship audio. Never try to identify speakers.
- Keep models simple and named for their physics, and mark heuristics as heuristics. Credit borrowed methods (umbrella-rain) in docstrings and the README.
- This repository is an optional submodule of voice-analyser (`external/random-cafe`). Keep its interface (`render`, `mix`, `Scene`, presets) stable.
