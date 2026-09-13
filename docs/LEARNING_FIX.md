# Learning stagnation investigation

Local investigation on 2026-09-13, Windows/Python 3.12, CPU. This records observed
failures as well as changes; it is not a claim of convergence or biological benefit.

## Reproduced problem

The original saved comparison had zero food collection and zero distance traveled
for baseline/connectome agents. They selected idle or turn-in-place actions.
The corresponding local seed-42 training logs also had very little food collection
near the end. An optimization progress bar reaching 100% had hidden the lack of
useful behavior. Old checkpoints and result files were preserved.

There were three stationary actions out of five, lower energy drain while idle
or turning, weak food-distance shaping, and substantial collision/death penalties.
These are plausible contributors to exploration failure, not a proof that the
original task's globally optimal policy is stationary.

## Implemented changes

- Optional `environment.turn_speed_fraction` enables turning while moving and
  charges movement energy. Absent/zero preserves old checkpoint semantics.
- Optional bounded food-heading potential adds orientation information to
  `gamma * Phi(next) - Phi(current)`. It preserves the terminal correction and
  does not pay an action bonus for repeated turns.
- Predator safety shaping now uses the actual difficulty-scaled sensing radius.
- Episode logs and TensorBoard record all five action counts and actual stationary
  steps, plus the existing food/distance/survival metrics. Progress includes recent
  food collection and distance rather than only a timestep counter.
- Post-training validation records deterministic and sampled policies on separate
  development seeds, with four non-learning controls. Evaluation saves/restores
  Torch RNG state and isolates action sampling from environment/noise randomness.
- Intact-policy diagnostics warn on stationary or zero-foraging evaluations.
  Damage outcomes are still measured without censoring failed models.
- Plots accept `--config` so their output directory follows the selected protocol.
  Distance/stationary training plots reveal failures that reward curves can hide.

These changes do not modify measured graph edges or replace the graph with an MLP.
PPO, the selected 500-neuron/35,603-edge real MaleCNS graph, and graph I/O mapping
remain the same across this investigation's connectome runs.

## Development pilots actually run

Each row below is the mean of five deterministic development episodes, one
training seed (42), familiar map layouts, using actual saved checkpoints.

| Protocol | Steps per model | Baseline food | Connectome food | Baseline distance | Connectome distance |
|---|---:|---:|---:|---:|---:|
| Motion/energy/reward pilot | 16,384 | 0.0 | 0.2 | 12.56 | 60.93 |
| Heading-potential pilot | 32,768 | 7.0 | 0.2 | 135.42 | 76.46 |

The second row changes both training budget and heading shaping, so the observed
difference cannot be attributed to heading alone. The connectome pilot remained
weak. A nonzero food count is not proof of learning above random exploration.

Pilot configs are `learning_pilot.yaml` and `learning_pilot_heading.yaml`.
Their raw CSVs/checkpoints remain locally under matching `results/` subdirectories
and are Git-ignored. Every run stores its fully resolved configuration; later
changes to inherited defaults do not retroactively change that saved record.

## Revised protocol and commands

`configs/learning.yaml` opts into the heading pilot's parameters, uses development
seeds 20001–20005 and reserves 30001–30010 for final evaluation. It trains seeds
42 and 43 at 32,768 steps each. This remains a small exploratory budget, especially
for the graph architecture. It does not promise successful convergence.

```bash
python -m pytest -q
python scripts/run_experiments.py --config configs/learning.yaml --experiment comparison
python scripts/generate_plots.py --config configs/learning.yaml
```

The real cache must already be available; otherwise follow the README downloader
setup. Results are written to `results/learning_v2/`. Before running the full
robustness suite, inspect each model's `learning_health.json`, `validation.csv`,
`reference_validation.csv`, and the comparison's `csv/learning_health.csv`.
There is deliberately no automatic checkpoint/seed selection based on validation.

## Completed two-seed real-data check

The revised comparison completed on CPU: four checkpoints, 32,768 steps each,
and 10 paired final evaluation episodes per checkpoint (40 measured episodes).
The final episode seeds were not used to select the revised settings.

| Model | Training seed | Mean food | Mean distance | Mean survival (seconds) | Stationary step fraction |
|---|---:|---:|---:|---:|---:|
| Baseline | 42 | 3.5 | 80.79 | 26.00 | 0.487 |
| Baseline | 43 | 1.8 | 45.41 | 19.32 | 0.444 |
| Connectome | 42 | 0.5 | 84.53 | 27.74 | 0.376 |
| Connectome | 43 | 1.1 | 80.30 | 23.18 | 0.522 |

Across the two training seeds, mean food collection was **2.65 for baseline** and
**0.80 for connectome**. No evaluated episode was entirely stationary, but the
stationary-step fractions still show substantial periods of no displacement.
These are descriptive pilot measurements, not significance or convergence claims.

On the five development episodes, the uniform-random control collected 0.4 food
on average; idle and either constant-turn control collected zero. Deterministic
baseline means were 7.0/2.2 across seeds 42/43; connectome means were 0.2/0.6.
Thus aggregate connectome development food collection matched the random control.
The connectome's movement and occasional food events do **not** establish reliable
learned foraging. A larger prespecified budget, graph signal/gradient diagnostics,
and more independent training seeds remain necessary before robustness claims.
Any further tuning should use development seeds and reserve a new final seed set.

Raw episode/summary CSVs and the checkpoint manifest are under
`results/learning_v2/`. The saved checkpoints are:

```text
baseline_seed42_add72b0f2292
baseline_seed43_cf47bf09834b
connectome_seed42_79edda5ad7c1
connectome_seed43_f17b6232f78d
```

Each folder contains the resolved config, source fingerprint, data provenance,
validation reports and model binary. No scientific advantage is inferred from
the connectome's slightly longer survival in this small, differently foraging sample.

## Offline execution check

For a fast offline execution check of all six experiment paths:

```bash
python scripts/run_experiments.py --config configs/learning_smoke.yaml
python scripts/generate_plots.py --config configs/learning_smoke.yaml
```

This uses a 64-neuron synthetic fixture and 128 training steps; warnings about
non-foraging are expected and appropriate. Passing software checks is separate
from demonstrating learned foraging, generalization, or damage resilience.
