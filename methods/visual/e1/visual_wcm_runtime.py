"""Public-input executable runtime for E1 Visual WCM.

The runtime owns a versioned Active State, a public Visual Binding Store, and
an append-only transaction log.  It never reads evaluator labels or simulator
metadata.  A learned visual Grounder proposes one typed EventFact; the runtime
compiles it into a guarded StatePatch and atomically commits it only when the
schema, expected-old value, and version all agree.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

from e0_5.grounder_protocol import GrounderFact, GrounderProtocolError


UNTRACKED = "__WCM_UNTRACKED__"
RUNTIME_VERSION = "visual-wcm-runtime/e1-v2-fast-path"


class RuntimeViolation(ValueError):
    """A proposed fact or patch violates the executable world contract."""


class WorldCodeSyntaxError(RuntimeViolation):
    """A Coding Programmer response is not a legal restricted StatePatch."""


def _copy_value(value: str | dict[str, str]) -> str | dict[str, str]:
    return deepcopy(value)


@dataclass(frozen=True, slots=True)
class StatePatch:
    event_id: str
    base_version: int
    entity_id: str
    component: str
    expected_old: str | dict[str, str]
    new: str | dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "base_version": self.base_version,
            "entity_id": self.entity_id,
            "component": self.component,
            "expected_old": _copy_value(self.expected_old),
            "new": _copy_value(self.new),
        }


def _remove_one_outer_code_fence(code: str) -> str:
    """Remove presentation-only outer fencing, without repairing program text."""

    text = code.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        raise WorldCodeSyntaxError("unterminated Markdown code fence")
    return "\n".join(lines[1:-1]).strip()


def parse_state_patch_code(raw_code: str) -> StatePatch:
    """Parse one literal-only ``set_state(...)`` patch without executing code.

    The programming interface deliberately admits a single function call with
    six named literal fields.  The Runtime remains the authority for types,
    versions, and expected-old semantics; this parser guarantees that an LLM
    cannot run arbitrary Python while attempting to manage memory.
    """

    if not isinstance(raw_code, str) or not raw_code.strip():
        raise WorldCodeSyntaxError("world code must be a non-empty string")
    try:
        expression = ast.parse(_remove_one_outer_code_fence(raw_code), mode="eval").body
    except SyntaxError as error:
        raise WorldCodeSyntaxError("world code is not one valid expression") from error
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name) or expression.func.id != "set_state":
        raise WorldCodeSyntaxError("world code must be exactly set_state(...)")
    if expression.args:
        raise WorldCodeSyntaxError("set_state positional arguments are forbidden")
    values: dict[str, object] = {}
    for keyword in expression.keywords:
        if keyword.arg is None or keyword.arg in values:
            raise WorldCodeSyntaxError("set_state keyword arguments are malformed")
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError) as error:
            raise WorldCodeSyntaxError("set_state arguments must be Python literals") from error
    expected_keys = {"event_id", "base_version", "entity", "component", "expected_old", "new"}
    if set(values) != expected_keys:
        raise WorldCodeSyntaxError("set_state requires exactly event_id, base_version, entity, component, expected_old, new")
    if not isinstance(values["event_id"], str) or not values["event_id"]:
        raise WorldCodeSyntaxError("event_id must be a non-empty string")
    if not isinstance(values["base_version"], int) or values["base_version"] < 0:
        raise WorldCodeSyntaxError("base_version must be a non-negative integer")
    if not isinstance(values["entity"], str) or not values["entity"] or not isinstance(values["component"], str):
        raise WorldCodeSyntaxError("entity and component must be non-empty strings")
    for field in ("expected_old", "new"):
        value = values[field]
        if not isinstance(value, (str, dict)):
            raise WorldCodeSyntaxError(f"{field} must be a string or literal dictionary")
        if isinstance(value, dict) and not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
            raise WorldCodeSyntaxError(f"{field} dictionary must map strings to strings")
    return StatePatch(
        event_id=values["event_id"],
        base_version=values["base_version"],
        entity_id=values["entity"],
        component=values["component"],
        expected_old=_copy_value(values["expected_old"]),
        new=_copy_value(values["new"]),
    )


@dataclass(frozen=True, slots=True)
class CommitReceipt:
    event_id: str
    version_before: int
    version_after: int
    entity_id: str
    component: str

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "version_before": self.version_before,
            "version_after": self.version_after,
            "entity_id": self.entity_id,
            "component": self.component,
        }


class EntityIdentityBindingStore:
    """Public persistent entity identity bindings for one stream.

    MiniGrid v2 guarantees one visible ``type:color`` key per catalog entity;
    this store records that public binding so StatePatches use stable entity
    IDs rather than an anonymous visual class.
    """

    def __init__(self, entity_catalog: Iterable[dict[str, object]]) -> None:
        self._entities: dict[str, dict[str, object]] = {}
        self._visual_keys: dict[tuple[str, str], str] = {}
        self._zone_ids: set[str] = set()
        for entry in entity_catalog:
            entity_id, entity_type = entry.get("entity_id"), entry.get("entity_type")
            if not isinstance(entity_id, str) or not isinstance(entity_type, str) or entity_id in self._entities:
                raise RuntimeViolation("malformed or duplicate public entity catalog entry")
            copied = dict(entry)
            self._entities[entity_id] = copied
            if entity_type == "zone":
                self._zone_ids.add(entity_id)
                continue
            color = entry.get("color")
            if not isinstance(color, str) or (entity_type, color) in self._visual_keys:
                raise RuntimeViolation("visible entity type/color bindings must be unique")
            self._visual_keys[(entity_type, color)] = entity_id
        if not self._entities or not self._zone_ids:
            raise RuntimeViolation("catalog must contain entities and zones")

    def entity_type(self, entity_id: str) -> str:
        try:
            return str(self._entities[entity_id]["entity_type"])
        except KeyError as error:
            raise RuntimeViolation(f"unknown public entity: {entity_id}") from error

    def is_zone(self, entity_id: str) -> bool:
        return entity_id in self._zone_ids

    def zone_ids(self) -> tuple[str, ...]:
        """Public catalog zone IDs, in deterministic order."""

        return tuple(sorted(self._zone_ids))

    def snapshot(self) -> dict[str, object]:
        return {"entity_count": len(self._entities), "zone_count": len(self._zone_ids), "visual_binding_count": len(self._visual_keys)}


class ActiveStateRuntime:
    """Typed, versioned current-world state with atomic patch commits."""

    def __init__(self, *, entity_catalog: Iterable[dict[str, object]]) -> None:
        self.binding_store = EntityIdentityBindingStore(entity_catalog)
        self.version = 0
        self._state: dict[tuple[str, str], str | dict[str, str]] = {}
        self._seen_event_ids: set[str] = set()
        self.transaction_log: list[CommitReceipt] = []

    def _current(self, *, entity_id: str, component: str) -> str | dict[str, str]:
        return _copy_value(self._state.get((entity_id, component), UNTRACKED))

    def current_value(self, *, entity_id: str, component: str) -> str | dict[str, str]:
        """Expose the current typed value to the Programmer, not any Oracle."""

        self.binding_store.entity_type(entity_id)
        if component not in {"place", "openness", "toggle_state"}:
            raise RuntimeViolation(f"unsupported schema component: {component!r}")
        return self._current(entity_id=entity_id, component=component)

    def compile_fact(self, *, fact: GrounderFact, event_id: str) -> StatePatch:
        if not event_id or event_id in self._seen_event_ids:
            raise RuntimeViolation(f"duplicate or empty event id: {event_id!r}")
        return StatePatch(
            event_id=event_id,
            base_version=self.version,
            entity_id=fact.entity_id,
            component=fact.component,
            expected_old=self._current(entity_id=fact.entity_id, component=fact.component),
            new=_copy_value(fact.new),
        )

    def _validate_typed_value(self, patch: StatePatch) -> None:
        entity_type = self.binding_store.entity_type(patch.entity_id)
        if patch.component == "place":
            if entity_type not in {"ball", "key"}:
                raise RuntimeViolation("place applies only to ball/key entities")
            if not isinstance(patch.new, dict) or patch.new.get("kind") != "in_zone" or not isinstance(patch.new.get("target"), str):
                raise RuntimeViolation("place must be {'kind':'in_zone','target':'zone_i'}")
            if not self.binding_store.is_zone(patch.new["target"]):
                raise RuntimeViolation("place target must be a public zone")
        elif patch.component == "openness":
            if entity_type != "door" or patch.new not in {"open", "closed"}:
                raise RuntimeViolation("openness applies only to doors with open/closed values")
        elif patch.component == "toggle_state":
            if entity_type != "switch" or patch.new not in {"on", "off"}:
                raise RuntimeViolation("toggle_state applies only to switches with on/off values")
        else:
            raise RuntimeViolation(f"unsupported schema component: {patch.component!r}")

    def preflight(self, patch: StatePatch) -> None:
        if patch.base_version != self.version:
            raise RuntimeViolation("stale base version")
        if patch.event_id in self._seen_event_ids:
            raise RuntimeViolation("duplicate event id")
        self._validate_typed_value(patch)
        current = self._current(entity_id=patch.entity_id, component=patch.component)
        if current != patch.expected_old:
            raise RuntimeViolation("expected-old mismatch")

    def commit(self, patch: StatePatch) -> CommitReceipt:
        self.preflight(patch)
        before = self.version
        self._state[(patch.entity_id, patch.component)] = _copy_value(patch.new)
        self._seen_event_ids.add(patch.event_id)
        self.version += 1
        receipt = CommitReceipt(patch.event_id, before, self.version, patch.entity_id, patch.component)
        self.transaction_log.append(receipt)
        return receipt

    def apply_fact(self, *, fact: GrounderFact, event_id: str) -> CommitReceipt:
        return self.commit(self.compile_fact(fact=fact, event_id=event_id))

    def answer(self, *, entity_id: str, component: str) -> str | dict[str, str]:
        value = self._current(entity_id=entity_id, component=component)
        if value == UNTRACKED:
            raise RuntimeViolation(f"query slot is untracked: {entity_id}|{component}")
        return value

    def state_dump(self, *, slots: Iterable[str]) -> dict[str, str | dict[str, str]]:
        """Return a complete typed snapshot over an explicitly public scope.

        The caller supplies only slot *names* (``entity_id|component``), never
        values.  A missing slot is an observable method failure, rather than an
        opportunity for the runtime to fabricate a default from simulator
        state.  This is the visual counterpart of Textual E1's active-state
        probe.
        """

        result: dict[str, str | dict[str, str]] = {}
        seen: set[str] = set()
        for slot in slots:
            if not isinstance(slot, str) or slot.count("|") != 1:
                raise RuntimeViolation("state dump slot must use entity_id|component")
            entity_id, component = slot.split("|", 1)
            if not entity_id or component not in {"place", "openness", "toggle_state"}:
                raise RuntimeViolation("state dump slot has invalid entity/component")
            if slot in seen:
                raise RuntimeViolation("state dump slots must be unique")
            seen.add(slot)
            result[slot] = self.answer(entity_id=entity_id, component=component)
        if not result:
            raise RuntimeViolation("state dump scope must be non-empty")
        return result

    def active_state_snapshot(self) -> dict[str, str | dict[str, str]]:
        """Return the Runtime's complete *method-produced* current state.

        This is intentionally distinct from a simulator snapshot: it contains
        only slots that were previously accepted through this Runtime's public
        Grounder → Patch → Auditor path.  E2's Program-to-Frame compiler may
        select a query-relevant projection from this object.
        """

        return {
            f"{entity_id}|{component}": _copy_value(value)
            for (entity_id, component), value in sorted(self._state.items())
        }

    def snapshot(self) -> dict[str, object]:
        return {
            "runtime_version": RUNTIME_VERSION,
            "version": self.version,
            "tracked_slot_count": len(self._state),
            "transaction_count": len(self.transaction_log),
            "visual_bindings": self.binding_store.snapshot(),
        }
