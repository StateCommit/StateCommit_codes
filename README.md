<div align="center">

# StateCommit

### Maintaining active world state for persistent world models

[Project Page](https://statecommit.github.io/) · [StateReturn Benchmark](https://huggingface.co/datasets/StateCommit/StateReturn) · [Code](https://github.com/StateCommit/StateCommit_codes)

</div>

## Overview

StateCommit studies a basic failure mode of persistent world models: an interaction may change the world, then leave the current observation. A model must retain what changed rather than rely on the most recent view alone.

<p align="center">
  <img src="assets/readme/persistence-gap.png" alt="The same current observation can arise from different past states and lead to different futures." width="100%" />
</p>

The method maintains an explicit Active State. For each verified interaction, the Visual path grounds a candidate change, compiles it into a restricted state patch, checks it, and commits it to a versioned Runtime. The stored state can then answer later queries or condition a future renderer.

This repository contains the public reference implementation for StateCommit and StateReturn. It is intentionally a compact method release: training code, model weights, official evaluation assets, and hidden test labels are not included.

> **Review-period release.** The public repository currently provides the protocols, Runtime reference, Text method, and Visual Fast Path interface. Visual Grounder/renderer training and checkpoints, complete baseline artifacts, and the official private evaluator are TODO for a post-review release.

## StateReturn

StateReturn evaluates whether state persists through intervening interactions in both text and visual environments.

<p align="center">
  <img src="assets/readme/statereturn-construction.png" alt="Construction of the StateReturn text and visual tracks." width="100%" />
</p>

- **Text:** AI2-THOR interaction histories in structured and natural-language forms.
- **Visual:** partially observed MiniGrid worlds with state-query and next-frame tasks.
- **Horizons:** `K ∈ {0, 1, 2, 4, 8, 16}` intervening interactions.

The full benchmark, including public development data, is available on [Hugging Face](https://huggingface.co/datasets/StateCommit/StateReturn). See the [project page](https://statecommit.github.io/) for examples and results.

## Installation

```bash
git clone https://github.com/StateCommit/StateCommit_codes.git
cd StateCommit_codes

python -m pip install -e .
```

The public validators and Visual Fast Path do not require third-party runtime dependencies. Install the optional dependencies when running the local Qwen-based Text method:

```bash
python -m pip install -r requirements-optional.txt
```

## Quick start

First validate the downloaded public data:

```bash
statereturn-public verify-text \
  --dataset-root /path/to/StateReturn-Text-v1.0

statereturn-public verify-visual \
  --dataset-root /path/to/StateReturn-Visual-v1.0-development
```

Run the Text StateCommit method (`wcm`) on one representation:

```bash
python scripts/run_text_method.py \
  --dataset-root /path/to/StateReturn-Text-v1.0 \
  --output-root /path/to/text_run \
  --split validation \
  --representation structured \
  --method wcm \
  --model-path /path/to/local/model \
  --device cuda:0
```

For Visual state queries, provide public Grounder facts and run the transactional Fast Path:

```bash
statereturn-public run-visual-statecommit \
  --dataset-root /path/to/StateReturn-Visual-v1.0-development \
  --split validation \
  --facts /path/to/grounder_facts.jsonl \
  --output-root /path/to/visual_run
```

The CLI also provides `build-next-frame-jobs` and `validate-next-frame-predictions` for the public causal next-frame interface. Run `statereturn-public --help` for all options.

## Repository structure

```text
src/statereturn_public/     Public CLI and dataset validators
methods/text/               StateCommit text method and public contracts
methods/visual/             Visual Runtime, Fast Path, and public interfaces
scripts/                    Text runner
assets/readme/              README figures
```

## Citation

```bibtex
@inproceedings{anonymous2027statecommit,
  title     = {StateCommit: Maintaining Active World State for Persistent World Models},
  author    = {Anonymous Authors},
  booktitle = {International Conference on Learning Representations},
  year      = {2027}
}
```

Citation metadata will be updated after the review process.

## License

Released under the [MIT License](LICENSE).
