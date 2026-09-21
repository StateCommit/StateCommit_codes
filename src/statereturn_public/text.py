from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SPLITS = ("train", "validation", "test")
REPRESENTATIONS = ("structured", "natural")


class PublicProtocolError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicProtocolError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise PublicProtocolError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise PublicProtocolError(f"missing public file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_release_fingerprint(dataset_root: str | Path) -> dict[str, object]:
    root = Path(dataset_root).resolve()
    manifest = _read_json(root / "metadata" / "release_manifest.json")
    if (
        manifest.get("benchmark") != "StateReturn-Text"
        or manifest.get("release_version") != "StateReturn/v1.0"
        or manifest.get("package_kind") != "public"
        or manifest.get("representations") != list(REPRESENTATIONS)
    ):
        raise PublicProtocolError("not a frozen StateReturn/v1.0 public Text release")
    if (root / "labels" / "test.jsonl").exists():
        raise PublicProtocolError("public Text release must not contain test labels")
    return {
        "benchmark": manifest["benchmark"],
        "release_version": manifest["release_version"],
        "manifest_sha256": _sha256(root / "metadata" / "release_manifest.json"),
        "public_data_files": {
            representation: {
                split: _sha256(root / "data" / representation / f"{split}.jsonl")
                for split in SPLITS
            }
            for representation in REPRESENTATIONS
        },
        "labels_opened": False,
    }
