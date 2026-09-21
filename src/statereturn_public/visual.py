"""Public Visual protocol entry points backed by the canonical Fast Path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from e0.public_protocol import PublicBenchmark, PublicProtocolError, load_public_benchmark
from e0_5.grounder_protocol import GrounderFact, GrounderProtocolError, parse_grounder_output
from e1.runner import run_visual_wcm_state_queries
from e1.tracked_state_protocol import TrackedStateProtocolError, load_public_tracked_state_scopes
from e1.visual_wcm_runtime import RuntimeViolation


class VisualPublicError(ValueError):
    """A released Visual input or supplied Grounder output violates the protocol."""


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VisualPublicError(f"cannot read public JSON: {path}") from error
    if not isinstance(value, dict):
        raise VisualPublicError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise VisualPublicError(f"cannot read Grounder output file: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise VisualPublicError(f"blank JSONL line at {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise VisualPublicError(f"invalid JSONL at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise VisualPublicError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _release_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).resolve()
    manifest = _read_json(root / "metadata" / "release_manifest.json")
    if manifest.get("benchmark") != "StateReturn-Visual" or manifest.get("release_version") != "StateReturn/v1.0":
        raise VisualPublicError("not a StateReturn-Visual/v1.0 release")
    if manifest.get("package_kind") not in {"public", "public_development"}:
        raise VisualPublicError("Visual benchmark is not a supported public package")
    return root


def load_visual_benchmark(dataset_root: str | Path) -> PublicBenchmark:
    """Validate a released Visual package and return its public method inputs."""

    root = _release_root(dataset_root)
    try:
        return load_public_benchmark(root / "public")
    except PublicProtocolError as error:
        raise VisualPublicError(str(error)) from error


def _fact_index(path: str | Path) -> dict[tuple[str, int], str]:
    facts: dict[tuple[str, int], str] = {}
    for row in _read_jsonl(Path(path)):
        if set(row) != {"stream_id", "frame_index", "raw_output"}:
            raise VisualPublicError("Grounder output requires stream_id, frame_index, and raw_output exactly")
        stream_id, frame_index, raw_output = row["stream_id"], row["frame_index"], row["raw_output"]
        if not isinstance(stream_id, str) or not stream_id or not isinstance(frame_index, int) or frame_index < 1:
            raise VisualPublicError("Grounder output has an invalid stream_id or frame_index")
        if not isinstance(raw_output, str):
            raise VisualPublicError("Grounder raw_output must be a string")
        key = (stream_id, frame_index)
        if key in facts:
            raise VisualPublicError(f"duplicate Grounder output: {key}")
        facts[key] = raw_output
    return facts


class _RecordedVisualGrounder:
    name = "recorded-public-grounder"

    def __init__(self, *, benchmark: PublicBenchmark, split: str, facts_path: str | Path) -> None:
        facts = _fact_index(facts_path)
        all_transitions: set[tuple[str, int]] = set()
        selected_transitions: dict[Path, str] = {}
        for stream_id, stream in benchmark.streams.items():
            for frame in stream.frames[1:]:
                if frame.action not in {"drop", "toggle"}:
                    continue
                key = (stream_id, frame.frame_index)
                all_transitions.add(key)
                if stream.split == split:
                    if key not in facts:
                        raise VisualPublicError(f"missing Grounder output for public transition {key}")
                    after_path = (benchmark.public_root / frame.rgb_path).resolve()
                    selected_transitions[after_path] = facts[key]
        unknown = sorted(set(facts) - all_transitions)
        if unknown:
            raise VisualPublicError(f"Grounder output refers to a non-transition frame: {unknown[:3]}")
        self._raw_by_after_path = selected_transitions
        self.invalid_fact_count = 0

    def ground(
        self, *, before_image: str | Path, after_image: str | Path, action: str, entity_catalog: Any
    ) -> GrounderFact | None:
        after_path = Path(after_image).resolve()
        raw_output = self._raw_by_after_path.get(after_path)
        if raw_output is None:
            raise VisualPublicError(f"no recorded Grounder output for public image {after_path}")
        try:
            return parse_grounder_output(raw_output, action=action, entity_catalog=entity_catalog)
        except GrounderProtocolError as error:
            self.invalid_fact_count += 1
            raise VisualPublicError(f"invalid recorded Grounder output for {after_path}: {error}") from error


def run_visual_statecommit(dataset_root: str | Path, *, split: str, facts_path: str | Path, output_root: str | Path) -> dict[str, object]:
    """Run the canonical Visual Grounder→Programmer→Auditor→Runtime Fast Path."""

    if split not in {"train", "validation", "test"}:
        raise VisualPublicError("unsupported split")
    benchmark = load_visual_benchmark(dataset_root)
    if not benchmark.queries_for(split=split, task="state_query"):
        raise VisualPublicError("selected split has no public Visual state queries")
    try:
        scopes = load_public_tracked_state_scopes(benchmark.public_root, benchmark)
        grounder = _RecordedVisualGrounder(benchmark=benchmark, split=split, facts_path=facts_path)
        report = run_visual_wcm_state_queries(
            benchmark=benchmark,
            split=split,
            grounder=grounder,
            output_root=output_root,
            tracked_state_scopes=scopes,
        )
    except (PublicProtocolError, GrounderProtocolError, RuntimeViolation, TrackedStateProtocolError) as error:
        raise VisualPublicError(str(error)) from error
    report["grounder_source"] = "recorded public fact file"
    return report


def visual_release_fingerprint(dataset_root: str | Path) -> dict[str, object]:
    benchmark = load_visual_benchmark(dataset_root)
    try:
        scopes = load_public_tracked_state_scopes(benchmark.public_root, benchmark)
    except TrackedStateProtocolError as error:
        raise VisualPublicError(str(error)) from error
    scope_path = benchmark.public_root / "tracked_state_scopes.json"
    return {
        "benchmark": "StateReturn-Visual",
        "release_version": "StateReturn/v1.0",
        "public_scope_sha256": _sha256(scope_path),
        "public_scope_count": len(scopes.scopes),
        "stream_count": len(benchmark.streams),
        "query_count": len(benchmark.queries),
        "private_data_opened": False,
    }
