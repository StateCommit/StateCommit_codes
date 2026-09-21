
"""Train Text IRMem on public train/validation labels; test is never opened."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "text"))

from baseline.irmem import train_irmem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-path", required=True, help="Local frozen text model matching the recorded encoder fingerprint.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--representation", choices=("structured", "natural"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--lr-scheduler-patience", type=int, default=5)
    args = parser.parse_args()
    try:
        result = train_irmem(
            dataset_root=args.dataset_root,
            model_path=args.model_path,
            output_path=args.output,
            cache_root=args.cache_root,
            representation=args.representation,
            device=args.device,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            key_dim=args.key_dim,
            label_smoothing=args.label_smoothing,
            lr_scheduler_patience=args.lr_scheduler_patience,
        )
    except (RuntimeError, ValueError, FileExistsError) as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
