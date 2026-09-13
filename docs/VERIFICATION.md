# Local verification record

Verified on **2026-09-12**, Windows, Python **3.12.13**, CPU. These checks establish
software execution, not biological validity or convergence of a learned policy.

## Follow-up: learning diagnostics, 2026-09-13

On branch `codex/fix-learning-stagnation`, **90 offline tests passed**, including
moving-turn energy accounting, legacy turn behavior, difficulty-aware shaping,
discounted potential telescoping, per-episode action counters, validation seed
separation, stochastic evaluation reproducibility/RNG restoration, and validation
artifact checks for all four PPO architectures. Ruff lint and formatting passed;
`pip check` reported no broken requirements.

The revised-motion synthetic smoke config `configs/learning_smoke.yaml` completed
all six experiment paths and generated **14 plots**, including two new behavioral
training curves. The neuron-damage figure was visually inspected. A headless
100-step Pygame demo using `configs/learning.yaml` rendered a valid frame, which
was also inspected. These checks do not establish learning; explicit no-foraging
warnings in short smoke runs are expected.

The original seed-42 baseline and connectome checkpoints were loaded with their
saved configurations and re-evaluated on the original paired seeds/maps. Episode
reward, food, distance and survival reproduced the recorded CSV values to a
1e-8 tolerance. The stochastic `fly-ai evaluate` CLI and a 100-step trained-baseline
checkpoint demo also completed. The publishable-file audit found no file over
1 MiB; real datasets and generated results remained excluded.

See [the learning investigation](LEARNING_FIX.md) for measured real-data pilot
outcomes, preserved old results, the revised protocol and its limitations.
The revised real-data comparison subsequently completed four 32,768-step runs
(baseline/connectome, seeds 42/43) and 40 final evaluation episodes. This supersedes
the earlier short-run-only scope below; it still does not establish convergence.

## Follow-up: authenticated real-data integration

Later on the same date, a user-supplied token enabled actual authenticated access
to `https://neuprint.janelia.org`, dataset **`male-cns:v1.0`**. The token was passed
through a hidden process prompt into that process's environment, not saved in
project files. The local cache contains **2,000 measured neurons and 348,394
directed edges**, with retrieval time and CSV integrity hashes in its manifest.

Default preprocessing selected a weakly connected **500-neuron, 35,603-edge**
induced subgraph. A real-data forward/backward check produced finite `[1, 32]`
features and gradients for existing edge weights. Then the actual CLI completed
**256 PPO steps on CPU** and saved
`results/models/connectome_seed42_6b1b518056bb/model.zip`.

This supersedes the initial absent-token integration limitation below. It is a
short integration check, not a converged model or a biological performance result.
The real cache and checkpoint are ignored by Git. Cached training now works without
another token or API request.

## Checks actually executed

| Check | Observed result |
|---|---|
| `python -m pytest -q` | **70 passed**, no warnings in the final run |
| Gymnasium `check_env` | Passed as part of the tests |
| `python -m ruff check src scripts tests` | All checks passed |
| `python -m ruff format --check src scripts tests` | 49 Python files formatted |
| `python -m pip check` | No broken requirements |
| All six workflow script `--help` entry points | Passed subprocess tests |
| Actual PPO updates: baseline, graph, fixed-edge graph, random sparse | Passed training/checkpoint tests |
| Same-seed training with two vector environments | Parameters identical on this CPU/runtime |
| Neuron/edge masks and checkpoint round-trip | Passed |
| Headless Pygame RGB demo and saved frame | Executed and image inspected |
| Pygame human-render/event path with SDL dummy display | Executed; no physical GUI interaction asserted |
| 100-node graph sample figure | Generated and image inspected |
| Real downloader, absent token | Clear expected CLI failure; no silent fallback |
| Downloader cache, explicit fallback, mocked ROI aggregation | Passed offline tests |
| Saved checkpoint reuse, standalone evaluation, checkpoint demo | Executed |
| Git exclusions | Virtualenv, secrets, downloaded data and generated outputs excluded |

## End-to-end smoke run

Executed:

```bash
python scripts/run_experiments.py --config configs/smoke.yaml
python scripts/generate_plots.py --output results/smoke
```

The run trained four models at 128 PPO steps each, using a **64-node synthetic
subgraph**, one training seed and short 64-step episodes. Six experiment CSVs
contain actual evaluations:

| CSV | Episode rows |
|---|---:|
| comparison | 4 |
| generalization | 20 |
| sensor_noise | 8 |
| neuron_damage | 16 |
| edge_damage | 8 |
| ablation | 8 |

Six summary CSVs and **12 PNG plots** were generated. A damage plot was visually
inspected for readable axes/labels and an explicit synthetic-data label. Undefined
retention ratios remain missing, rather than fabricated values. These very short
runs are unsuitable for estimating comparative learning or damage resilience.

## Intended default-size CPU check

Executed:

```bash
python scripts/run_experiments.py --config configs/verification.yaml --experiment comparison
```

The baseline and **500-node synthetic** graph model each completed **1,024 PPO
steps** and paired evaluation. The run recorded 26,950 trainable baseline parameters
and 4,086 graph-model parameters, confirming the documented capacity mismatch.
Both runs produced TensorBoard event files and saved checkpoints. The checkpoints
are generated artifacts under `results/verification/`, ignored by Git.

Key installed versions were Torch 2.14.0, Stable-Baselines3 2.9.0, Gymnasium 1.2.3,
NumPy 2.5.3 and neuprint-python 0.6.3. Full installed versions are captured in
`requirements-lock-windows-py312.txt`; each training run also records its own
source/config fingerprint and package versions.

## Initial verification limits and remaining unverified claims

- Initially no `NEUPRINT_TOKEN` was available. The real MaleCNS download was not completed then;
  API schemas were checked against official documentation and the client contract
  was tested with mocks. The authenticated follow-up above now verifies this path.
- Beyond the 256-step real-data integration check, no full research-budget multi-seed experiment,
  biological comparison, convergence result, or statistical superiority is claimed.
- CUDA execution, physical desktop window interaction, other operating systems,
  GitHub-hosted CI and the Docker build were not exercised locally. Their supplied
  configurations should be validated in their target environments.
- The repository is initialized locally on `main`; no remote publication is implied.

The local Git publishable-file audit found no file larger than 1 MB. The only
included raster preview is a small actual heuristic demo image. Large dependencies
live in the ignored `.venv`; measured datasets/checkpoints are not bundled.
