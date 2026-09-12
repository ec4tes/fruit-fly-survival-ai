# Experiment protocol and interpretation

This is a connectome-constrained artificial architecture experiment, not a brain
simulation. Commit a protocol/configuration before looking at held-out scores.

## Paired comparison

- The same Gymnasium class, 13 observations, five actions, reward function, PPO
  implementation, rollout budget and optimizer settings apply to every model.
- Default training uses map seeds 11/22/33/44/55, with varied episode spawns.
  Evaluation uses separate episode RNG seeds. Unseen layouts use a disjoint map
  seed list; layout generation has its own RNG.
- The suite trains all requested architectures on each training seed, then
  evaluates every perturbation on the same episode seeds and map assignments.
  Noise uses a separate RNG, so enabling noise cannot change environment RNG draws.
  Different actions can still consume food and hence spawn RNG at different times.
- PPO rounds a requested budget up to a whole rollout. Requested and actual steps,
  parameter counts, training time, device, code hash and versions are saved.
- Equal interaction budgets do **not** equal equal parameter counts or compute.
  The default dense MLP and graph model are intentionally not capacity matched.
  Report this; use additional MLP widths and compute-matched budgets for strong claims.

## Experiment definitions

| Experiment | Perturbation after training | Reference |
|---|---|---|
| comparison | none | familiar map layouts, held-out episode RNG |
| generalization | layout, food distribution, predator speed, then all combined | in-distribution episodes |
| sensor_noise | Gaussian noise per feature, clipped to feature bounds | zero noise |
| neuron_damage | persistent activation lesions, random and high-degree | intact same model |
| edge_damage | persistent zero-weight masks, random | intact same model |
| ablation | trainable graph, fixed-edge graph, degree-preserving random control, dense MLP | equal PPO steps |

Noise level is standard deviation as a fraction of the feature's numeric range:
distance/energy/speed/rays span [0,1], angular sin/cos span [-1,1]. Components receive
independent noise, so noisy sin/cos pairs need not lie on the unit circle. Noise
changes policy input only; rewards use uncorrupted world state.

Generalization modifiers are evaluated separately as well as jointly. Fixed maps
can be supplied as rectangles in `environment.fixed_obstacles`; in that mode map
seeds alone do not create unseen layouts. Use procedural maps for the default OOD
protocol. Avoid training on `map_seeds: null` when claiming strict layout holdout.

## Lesions

For N hidden units, floor(fraction*N) units are disabled. Masks remain unchanged
through the whole evaluation condition and across episodes, and reset before the
next condition. Reusing a lesion seed creates nested lesions across fractions.

Graph units include mapped input and output neurons; biases and injected sensory
signals cannot reactivate a damaged neuron. Baseline lesions cover all hidden
layers, sampled globally rather than independently rounding each layer. Neither
architecture's final actor/value readout is lesioned. A 100% hidden lesion therefore
leaves a bias-only policy, not undefined actions.

High-degree means unweighted in-degree plus out-degree. Dense layers use their
incident hidden/input edges, excluding final heads; ties are seed-randomized.
Graph ties use stable node order. Random neuron lesions are the primary
cross-architecture comparison; degree attacks differ structurally between models.

Graph edge damage affects measured recurrent edges only. Dense edge damage affects
feature-extractor matrices, including observation-to-first-hidden connections.
Both exclude final heads; graph sensory encoder weights are also excluded. This
boundary mismatch makes edge damage an exploratory comparison, not a clean test
of identical biological injury. Fractional lesions also remove different absolute
numbers of parameters and multiply-used graph edges participate in K updates.

## Sparse control

Directed double-edge swaps preserve exact per-node in/out degree, neuron count,
edge count, and each target's incoming weight multiset. No new self-loops or
duplicate edges are introduced. The target accepted-swap count is 10 times E,
with bounded attempts and the achieved count recorded. This is a practical null
model, not a proof of uniform sampling over degree-constrained graphs.

Input/output body IDs are copied from the original graph. Rewiring may disconnect
an output or move it beyond the configured K microsteps; it is not remapped to
hide this effect. `reachable_output_count` is stored in the policy spec. The
control does not preserve motifs, path lengths or weak/strong components. Compare
those diagnostics before assigning an effect specifically to biological topology.

## Statistics and retention

Episode-level CSVs are primary records. Summaries first average episodes and lesion
replicates within each trained model seed, then compute mean, sample SD and SEM
across training seeds. With one seed SD/SEM are undefined and no band is drawn.
Bands are SEM, not 95% confidence intervals. Default three seeds/five episodes
are a small exploratory pilot; increase them and use a seed-level paired bootstrap
or appropriate hierarchical analysis for publication-level uncertainty.

Retention is 100*perturbed/intact, where intact is the paired condition mean for
that trained model and lesion seed. Reward ratios are only emitted for a positive
intact mean; food/survival ratios require a nonzero positive mean. Otherwise they
are missing values, not zeros. Raw scores and score changes are always retained.
Curves with all-missing ratios are deliberately blank; inspect raw-score plots.

Energy efficiency means food events per energy spent, including idle metabolism;
it is not thermodynamic efficiency. Escape rate is completed exits from the
detection zone divided by zone entries; a no-encounter episode reports 0 alongside
an encounter count of 0. Capture rate is the mean of the capture indicator.
Survival time is observed time up to capture, depletion or timeout; timeouts are
censored observations, not proof of indefinite survival. Training episode curves
versus steps/time are exploratory learning-speed diagnostics, not held-out
sample-complexity estimates or proof of convergence.
