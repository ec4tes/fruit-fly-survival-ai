# Contributing

Install with `pip install -r requirements.txt`. Before a pull request, run:

```bash
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m pytest -q
python scripts/run_experiments.py --config configs/smoke.yaml
```

Tests must remain offline. Keep credentials, checkpoints and downloaded datasets
out of commits. New experimental knobs belong in YAML, with validation and a
documented definition. Preserve directed topology and provenance throughout any
preprocessing change. Never describe synthetic fixtures as biological measurements.

Research result submissions should include resolved configuration, code hash,
package versions, original data source/version, trained-model seeds, episode-level
CSV files, compute budgets and uncertainty over independent training seeds.
State failures, selection bias and negative results. Do not count episodes from
the same trained model as independent training replicates.
