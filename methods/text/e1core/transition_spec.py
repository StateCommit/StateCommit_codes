"""Frozen, public transition contracts for Structured and Natural E1.

The Structured contract names canonical actions.  The Natural contract names
only the resultative English facts and outcome sentences that occur in the
public Natural view.  Neither contract contains a private label or C1 trace.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .protocol import StructuredEvent, canonical_json, read_json, sha256_file


STRUCTURED_SPEC_PATH = Path(__file__).resolve().parent / "config" / "public_transition_spec_v1.json"
NATURAL_SPEC_PATH = Path(__file__).resolve().parent / "config" / "natural_event_transition_spec_v1.json"


class TransitionSpecError(ValueError):
    """A public event cannot be interpreted by the frozen public contract."""


@lru_cache(maxsize=1)
def load_public_transition_spec() -> dict[str, Any]:
    spec = read_json(STRUCTURED_SPEC_PATH)
    if spec.get("spec_version") != "wcm-e1-public-transition/v1" or not isinstance(spec.get("structured_actions"), Mapping):
        raise TransitionSpecError("unsupported Structured transition specification")
    return spec


@lru_cache(maxsize=1)
def load_natural_transition_spec() -> dict[str, Any]:
    spec = read_json(NATURAL_SPEC_PATH)
    if spec.get("spec_version") != "wcm-textworld-natural-event-transition/v1":
        raise TransitionSpecError("unsupported Natural transition specification")
    return spec


def spec_provenance(representation: str = "structured") -> dict[str, str]:
    if representation == "structured":
        spec, path = load_public_transition_spec(), STRUCTURED_SPEC_PATH
    elif representation == "natural":
        spec, path = load_natural_transition_spec(), NATURAL_SPEC_PATH
    else:
        raise TransitionSpecError(f"unsupported representation {representation!r}")
    return {"representation": representation, "spec_version": str(spec["spec_version"]), "sha256": sha256_file(path), "relative_path": f"config/{path.name}"}


def bind_dataset_transition_spec(dataset_root: str | Path, representation: str) -> dict[str, str] | None:
    """Bind Natural prompt semantics to the matching public release document.

    The dataset carries the Natural transition contract because it defines the
    language-only setting.  Refuse a run if a local prompt specification has
    drifted semantically from that public document.  Structured v2.4 inherits
    its separately versioned public action contract from this E1 package.
    """

    if representation == "structured":
        return None
    if representation != "natural":
        raise TransitionSpecError(f"unsupported representation {representation!r}")
    source = Path(dataset_root) / "schema" / "natural_event_transition_spec_v1.json"
    dataset_spec = read_json(source)
    local_spec = load_natural_transition_spec()
    if canonical_json(dataset_spec) != canonical_json(local_spec):
        raise TransitionSpecError("local Natural transition spec differs from the public dataset contract")
    return {"relative_path": "schema/natural_event_transition_spec_v1.json", "sha256": sha256_file(source), "spec_version": str(dataset_spec["spec_version"])}


def public_transition_prompt(representation: str = "structured") -> str:
    """Render exactly the public contract given to prompted E1 methods."""

    if representation == "natural":
        spec = load_natural_transition_spec()
        return "\n".join(
            [
                f"PUBLIC NATURAL-EVENT TRANSITION SPECIFICATION ({spec['spec_version']})",
                "Use only event_id, description, outcome_text, and the typed query.",
                "Legal state values:",
                '- openness: "open" or "closed".',
                '- toggle_state: "on" or "off".',
                '- place: {"kind":"on"|"inside"|"held_by","target":"<stable entity ID>"}.',
                "Outcome rule: the failure/no-change outcome sentences never update tracked state.",
                "A successful sentence explicitly saying opened/closed, on/off, picked up, placed on, or placed inside licenses only that stated current fact.",
                "Illustrative public-language examples:",
                '- "The agent opened drawer_1." licenses openness(drawer_1) = "open".',
                '- "Power was switched on for faucet_1." licenses toggle_state(faucet_1) = "on".',
                '- "The agent picked up mug_1." licenses place(mug_1) = {"kind":"held_by","target":"agent_1"}.',
                '- "The agent placed mug_1 on table_1." licenses place(mug_1) = {"kind":"on","target":"table_1"}.',
                '- "The agent moved mug_1 to the inside of drawer_1." licenses place(mug_1) = {"kind":"inside","target":"drawer_1"}.',
                "A failed/no-change outcome sentence licenses no update even if its description mentions an action.",
                "Never infer an unspoken relation or use a canonical action API, hidden Structured record, Oracle answer, old value, or private trace.",
            ]
        )
    spec = load_public_transition_spec()
    action_lines = []
    for action, rule in spec["structured_actions"].items():
        changed = rule["success_changed"]
        result = "set place to exactly the public place_effect" if changed.get("new_source") == "place_effect" else f"set {changed['component']} to {canonical_json(changed['new'])}"
        action_lines.append(f"- {action} + success_changed: {result}.")
    return "\n".join(
        [
            f"PUBLIC WORLD TRANSITION SPECIFICATION ({spec['spec_version']})",
            "This specification is public and is the only action semantics available to you.",
            "Legal state values:",
            '- openness: "open" or "closed".',
            '- toggle_state: "on" or "off".',
            '- place: {"kind":"on"|"inside"|"held_by","target":"<stable entity ID>"}.',
            "Outcome rule: success_no_change and failure_no_change never update tracked state.",
            "Success-changed action rules:", *action_lines,
            "Never use a hidden simulator state, an Oracle answer, an unobserved old value, or a private trace.",
        ]
    )


def expected_change(event: StructuredEvent) -> tuple[str, Any] | None:
    """Return the sole Structured state delta licensed by public evidence."""

    spec = load_public_transition_spec()
    if event.outcome in {"failure_no_change", "success_no_change"}:
        return None
    if event.outcome != "success_changed":
        raise TransitionSpecError(f"unsupported public outcome {event.outcome!r}")
    rule = spec["structured_actions"].get(event.action_type)
    if not isinstance(rule, Mapping):
        raise TransitionSpecError(f"unsupported Structured E1 action {event.action_type}")
    changed = rule.get("success_changed")
    if not isinstance(changed, Mapping) or not isinstance(changed.get("component"), str):
        raise TransitionSpecError(f"malformed public rule for {event.action_type}")
    component = str(changed["component"])
    if changed.get("new_source") == "place_effect":
        if event.place_effect is None:
            raise TransitionSpecError(f"{event.action_type} lacks required public place_effect")
        return component, {"kind": event.place_effect.kind, "target": event.place_effect.target}
    if "new" not in changed:
        raise TransitionSpecError(f"public rule for {event.action_type} lacks new value")
    return component, changed["new"]
