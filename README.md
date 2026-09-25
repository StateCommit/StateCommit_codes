# StateCommit

Anonymous public reference implementation for StateReturn.

## Included

- Public Text and Visual dataset validation.
- Typed state schemas and public transition contracts.
- A restricted, versioned Runtime that commits one event-local StatePatch at a time.
- Text StateCommit inference with a local frozen language model.
- Visual Fast Path replay from externally supplied public Grounder facts.
- Causal next-frame job construction and prediction-format validation without target access.

## Release status

This review-period repository is intentionally limited to public-input protocols and reference Runtime semantics. It is not a complete artifact for reproducing reported official-test tables.

### TODO: planned after review

- Visual Grounder training pipeline and released checkpoints.
- State-to-render training pipeline and released checkpoints.
- Full baseline training and evaluation artifacts.
- Official evaluator assets, hidden test labels, and scoring scripts.
- External-provider experiment runners and frozen prediction records.

## Installation

```bash
python -m pip install -e .
```

Public validators and Visual Fast Path replay require no third-party runtime dependencies. To run the local language-model Text reference path, install:

```bash
python -m pip install -r requirements-optional.txt
```

## Public interfaces

```bash
statereturn-public verify-text --dataset-root /path/to/StateReturn-Text-v1.0
statereturn-public verify-visual --dataset-root /path/to/StateReturn-Visual-v1.0-development
statereturn-public --help
```

Text StateCommit reference inference:

```bash
python scripts/run_text_method.py \
  --dataset-root /path/to/StateReturn-Text-v1.0 \
  --output-root /path/to/text_run \
  --split validation \
  --representation structured \
  --method wcm \
  --model-path /path/to/local/model
```

Visual Fast Path replay from public Grounder facts:

```bash
statereturn-public run-visual-statecommit \
  --dataset-root /path/to/StateReturn-Visual-v1.0-development \
  --split validation \
  --facts /path/to/grounder_facts.jsonl \
  --output-root /path/to/visual_run
```

## License

Released under the [MIT License](LICENSE).
