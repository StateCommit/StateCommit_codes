# StateReturn

Minimal reference code for StateReturn: transactional world-state maintenance
and state-consistent next-frame prediction.

This is a minimal reference implementation of the public-input protocol,
transactional state-maintenance Runtime, and inference components. It is not
an end-to-end reproduction package. Datasets, model weights, training
pipelines, predictions, evaluator assets, and experiment outputs are
intentionally not included.

Released under the MIT License.

## Install

```bash
python -m pip install -e .
```

The public protocol command has no third-party runtime dependencies:

```bash
statereturn-public --help
```

Install the learned-method dependencies only when needed:

```bash
python -m pip install -r requirements-optional.txt
```

## Public-input checks

```bash
statereturn-public verify-text --dataset-root /path/to/StateReturn-Text-v1.0
statereturn-public verify-visual --dataset-root /path/to/StateReturn-Visual-v1.0-development
```

The `methods/` directory contains the Text and Visual reference
implementations.  The resumable Text-method runner is available as:

```bash
python scripts/run_text_method.py --help
```

The three history renderers (`recent4`, `full_history`, and `implicit_gru`)
also support a query-conditioned control through
`RendererConfig(query_slot_conditioning=True)`.  It receives only the public
`entity_id` and `component` of the queried slot, never an Active-State value,
Visual Binding, or binding projection.  It changes the renderer architecture,
so it must be trained from scratch and is not interchangeable with an
unconditioned checkpoint.

## Visual Fast Path

`run-visual-statecommit` uses the same executable chain as the Visual method:

```text
Grounder fact -> deterministic restricted Programmer -> Grounded Patch Auditor -> versioned Runtime
```

```bash
statereturn-public run-visual-statecommit \
  --dataset-root /path/to/StateReturn-Visual-v1.0-development \
  --split validation \
  --facts /path/to/grounder_facts.jsonl \
  --output-root /path/to/visual_run
```

Each Grounder-fact row is a JSON object with exactly `stream_id`,
`frame_index`, and `raw_output`.  The Visual Table 2 main path uses the
deterministic restricted Programmer; its Qwen-Coder variant belongs to a
separate state-to-render control.

The released Visual package contains only public development data.  Official
test scoring requires the separately held evaluator; this source release does
not claim to expose test labels, counterfactual targets, masks, or simulator
metadata.  The benchmark implementation commits one event-local StatePatch
per public state-changing frame. Accordingly, the evaluated Base Track is the
singleton transaction specialization: each public event has at most one
tracked-slot write, and a rejected patch leaves the Active State unchanged.
