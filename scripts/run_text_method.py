
"""Run or resume one public-input Text method without evaluator access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "text"))

from baseline.factory import METHODS, make_factory
from baseline.runner import run_online


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--representation", choices=("structured", "natural"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--model-path", help="Local frozen text model; required by prompted methods and by path-sanitized IRMem metadata.")
    parser.add_argument("--irmem-checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--history-budget-events", type=int, default=4)
    parser.add_argument("--summary-chars", type=int, default=1600)
    parser.add_argument("--limit-episodes", type=int, help="Non-reportable smoke subset; included in the run manifest.")
    args = parser.parse_args()
    try:
        factory, options = make_factory(
            method=args.method,
            model_path=args.model_path,
            device=args.device,
            representation=args.representation,
            history_budget_events=args.history_budget_events,
            summary_chars=args.summary_chars,
            irmem_checkpoint=args.irmem_checkpoint,
        )
        result = run_online(
            dataset_root=args.dataset_root,
            split=args.split,
            output_root=args.output_root,
            method_factory=factory,
            options=options,
            representation=args.representation,
            limit_episodes=args.limit_episodes,
        )
    except (RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
