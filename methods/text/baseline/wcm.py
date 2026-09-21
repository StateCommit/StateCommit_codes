"""Multi-agent World Code Memory for representation-isolated StateReturn Text.

Structured WCM uses the published canonical-action transition contract.  In
Natural WCM, semantics are intentionally *not* recovered by a deterministic
parser: a Grounder proposes a typed ``EventFact`` from the Natural sentence,
then a Programmer compiles it into restricted World Code.  The Natural Runtime
checks only schema/transaction invariants and public entity mentions; an LLM
Evidence Auditor is the semantic second opinion.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from e1core.answer_codec import AnswerCodecError, UNTRACKED, normalize_value, unwrap_single_markdown_fence
from e1core.protocol import NaturalEvent, PublicEvent, QueryExample, REPRESENTATIONS, StructuredEvent, canonical_json
from e1core.transition_spec import TransitionSpecError, expected_change, public_transition_prompt

from .base import Generation, MemoryMethod, TextBackend, state_digest


class PatchError(ValueError):
    pass


class EventFactError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EventFact:
    """Grounder's typed, public-language interpretation; never an Oracle fact."""

    entity_id: str
    component: str
    new: Any
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"entity_id": self.entity_id, "component": self.component, "new": self.new}


@dataclass(frozen=True, slots=True)
class Patch:
    patch_id: str
    base_version: int
    entity_id: str
    component: str
    expected_old: Any
    new: Any
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"patch_id": self.patch_id, "base_version": self.base_version, "entity_id": self.entity_id, "component": self.component, "expected_old": self.expected_old, "new": self.new, "source": self.source}


def _literal_call(source: str, *, function: str, required: set[str], error_type: type[ValueError]) -> dict[str, Any]:
    source = unwrap_single_markdown_fence(source)
    if len(source.encode("utf-8")) > 4096:
        raise error_type("generated program exceeds 4 KB")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise error_type(f"invalid syntax: {exc.msg}") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        raise error_type("must contain exactly one expression")
    call = tree.body[0].value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != function:
        raise error_type(f"must be {function}(...)")
    if call.args or any(item.arg is None for item in call.keywords):
        raise error_type(f"{function} accepts keyword literals only")
    try:
        values = {str(item.arg): ast.literal_eval(item.value) for item in call.keywords}
    except (TypeError, ValueError) as exc:
        raise error_type("values must be Python literals") from exc
    if set(values) != required:
        raise error_type(f"{function} needs exactly {sorted(required)}")
    return values


def parse_event_fact(source: str) -> EventFact | None:
    """Parse a Grounder fact without executing model-produced code.

    This function checks only ID/string/type syntax.  It does not map language
    templates to components or values and therefore cannot be a semantic
    oracle for Natural WCM.
    """

    if unwrap_single_markdown_fence(source) == "NO_FACT":
        return None
    values = _literal_call(source, function="event_fact", required={"entity", "component", "new"}, error_type=EventFactError)
    entity, component = values["entity"], values["component"]
    if not isinstance(entity, str) or not entity.strip() or entity.strip() != entity or not isinstance(component, str):
        raise EventFactError("EventFact entity/component must be stable strings")
    try:
        new = normalize_value(component, values["new"])
    except AnswerCodecError as exc:
        raise EventFactError(str(exc)) from exc
    return EventFact(entity, component, new, unwrap_single_markdown_fence(source))


def parse_patch(source: str, *, patch_id: str, base_version: int) -> Patch:
    """Parse one literal-only DSL call; it is never executed as Python."""

    values = _literal_call(source, function="set_state", required={"entity", "component", "expected_old", "new"}, error_type=PatchError)
    entity, component = values["entity"], values["component"]
    if not isinstance(entity, str) or not entity.strip() or entity.strip() != entity or not isinstance(component, str):
        raise PatchError("patch entity/component must be stable strings")
    try:
        old = values["expected_old"] if values["expected_old"] == UNTRACKED else normalize_value(component, values["expected_old"])
        new = normalize_value(component, values["new"])
    except AnswerCodecError as exc:
        raise PatchError(str(exc)) from exc
    return Patch(patch_id, base_version, entity, component, old, new, unwrap_single_markdown_fence(source))


def _structured_expected(event: StructuredEvent) -> tuple[str, str, Any] | None:
    try:
        changed = expected_change(event)
    except TransitionSpecError:

        return None
    return None if changed is None else (str(event.arguments["object_id"]), changed[0], changed[1])


def _patch_skeleton(*, entity_id: str, component: str, current: Any, new: Any) -> str:
    """Render a literal-only Patch repair from public/runtime-visible values.

    This is prompt text, not an executable patch and not a parser fallback:
    the Coding Agent must still emit the line and ``parse_patch`` must still
    accept it.  For Structured events all fields come only from the published
    action contract plus the Runtime's current Active State; for Natural
    events, callers use the Grounder's public typed EventFact.
    """

    old_literal = repr(current) if current == UNTRACKED else canonical_json(current)
    return (
        f"set_state(entity={entity_id!r}, component={component!r}, "
        f"expected_old={old_literal}, new={canonical_json(new)})"
    )


class TrustedRuntime:
    """Versioned Active-State authority with no Natural semantic parser."""

    def __init__(self, *, representation: str = "structured") -> None:
        if representation not in REPRESENTATIONS:
            raise ValueError(f"unsupported representation {representation!r}")
        self.representation = representation
        self.entities: set[str] = {"agent_1"}
        self.values: dict[str, Any] = {}
        self.version = 0
        self.log: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.entities, self.values, self.version, self.log = {"agent_1"}, {}, 0, []

    def register(self, event: PublicEvent) -> None:
        before = set(self.entities)
        self.entities.update(event.entity_ids())
        added = sorted(self.entities - before)
        if added:
            self.log.append({"operation": "register_entities", "event_id": event.event_id, "added": added})

    def read(self, *, entity_id: str, component: str) -> Any:
        return self.values.get(f"{entity_id}|{component}", UNTRACKED)

    def _natural_gate(self, event: NaturalEvent, patch: Patch, fact: EventFact | None) -> None:
        """Schema/provenance checks only—never parse Natural event semantics."""

        if fact is None:
            raise PatchError("Natural patch requires a typed Grounder EventFact")
        if fact.entity_id not in event.entity_ids():
            raise PatchError("Grounder entity does not occur in the public Natural event")
        if fact.component not in {"openness", "place", "toggle_state"}:
            raise PatchError("Grounder component is outside E1 Base schema")
        if fact.component == "place" and fact.new["target"] not in event.entity_ids():
            raise PatchError("Grounder place target does not occur in the public Natural event")
        if patch.entity_id != fact.entity_id or patch.component != fact.component or canonical_json(patch.new) != canonical_json(fact.new):
            raise PatchError("patch must compile the typed Grounder EventFact exactly")

    def preflight(self, event: PublicEvent, patch: Patch, *, fact: EventFact | None = None) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        try:
            if patch.base_version != self.version:
                raise PatchError(f"base_version {patch.base_version} != current {self.version}")
            if patch.entity_id not in self.entities:
                raise PatchError(f"unregistered patch entity {patch.entity_id}")
            if patch.component == "place" and patch.new["target"] not in self.entities:
                raise PatchError(f"unregistered place target {patch.new['target']}")
            if isinstance(event, StructuredEvent):
                expected = _structured_expected(event)
                if expected is None:
                    raise PatchError("this Structured event has no publicly licensed Base-state patch")
                entity_id, component, new = expected
                if patch.entity_id != entity_id or patch.component != component or canonical_json(patch.new) != canonical_json(new):
                    raise PatchError("patch contradicts the published Structured transition contract")
            else:
                self._natural_gate(event, patch, fact)
            actual = self.read(entity_id=patch.entity_id, component=patch.component)
            if canonical_json(actual) != canonical_json(patch.expected_old):
                raise PatchError("expected_old does not match current Active State")
        except PatchError as exc:
            reasons.append(str(exc))
        return not reasons, reasons

    def commit(self, event: PublicEvent, patch: Patch, *, fact: EventFact | None = None) -> dict[str, Any]:
        accepted, reasons = self.preflight(event, patch, fact=fact)
        before = self.version
        if not accepted:
            receipt = {"event_id": event.event_id, "status": "rejected", "before_version": before, "after_version": before, "patch_id": patch.patch_id, "reasons": reasons}
        else:
            self.values[f"{patch.entity_id}|{patch.component}"] = patch.new
            self.version += 1
            receipt = {"event_id": event.event_id, "status": "committed", "before_version": before, "after_version": self.version, "patch_id": patch.patch_id}
        self.log.append({"operation": "submit", "event": event.to_dict(), "event_fact": None if fact is None else fact.to_dict(), "patch": patch.to_dict(), "receipt": receipt})
        return receipt

    def record_no_write(self, event: PublicEvent, *, status: str, details: Mapping[str, Any] | None = None) -> None:
        entry: dict[str, Any] = {"operation": "no_write", "event": event.to_dict(), "receipt": {"event_id": event.event_id, "status": status, "before_version": self.version, "after_version": self.version}}
        if details is not None:
            entry["details"] = dict(details)
        self.log.append(entry)

    def dump(self, scope: Sequence[str]) -> dict[str, Any]:


        return {key: self.values[key] for key in sorted(scope) if key in self.values}

    def public_view(self) -> dict[str, Any]:
        return {"representation": self.representation, "state_version": self.version, "registered_entity_ids": sorted(self.entities), "tracked_values": [{"slot": key, "value": value} for key, value in sorted(self.values.items())], "untracked_marker": UNTRACKED}


def _ground_prompt(runtime: TrustedRuntime, event: NaturalEvent) -> str:
    return "\n".join([public_transition_prompt("natural"), "You are the semantic Grounder. Convert the public Natural event into one typed EventFact whenever it explicitly states a successful Base-state result. Do not wait for the Runtime to supply a state delta.", "General English state rule: shut is a synonym of closed for an openable entity, so it establishes openness = 'closed'. Active and passive voice express the same current state effect (for example, 'the agent shut drawer_2' and 'drawer_2 was shut by the agent').", "Required output grammar:", "event_fact(entity='<stable ID appearing in the event>', component='<openness|toggle_state|place>', new=<typed value>)", "Format examples (illustrative only):", "Input: The agent swung cabinet_1 open. / The interaction completed successfully.", "Output: event_fact(entity='cabinet_1', component='openness', new='open')", "Input: drawer_2 was shut by the agent. / The interaction completed successfully.", "Output: event_fact(entity='drawer_2', component='openness', new='closed')", "Input: faucet_1 was set down on counter_1 by the agent. / The interaction completed successfully.", "Output: event_fact(entity='faucet_1', component='place', new={'kind':'on','target':'counter_1'})", "Input: The agent attempted to open cabinet_1. / The interaction failed. No change to the world was observed.", "Output: NO_FACT", "Current public Natural event:", json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True), "Output exactly one EventFact or NO_FACT. No prose."])


def _program_prompt(runtime: TrustedRuntime, event: PublicEvent, *, fact: EventFact | None = None) -> str:
    lines = [public_transition_prompt(runtime.representation), "Current executable Active State:", json.dumps(runtime.public_view(), ensure_ascii=False, sort_keys=True), "Current public event:", json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)]
    if fact is not None:
        lines += ["Typed Grounder EventFact (already schema-checked, but not an Oracle):", json.dumps(fact.to_dict(), ensure_ascii=False, sort_keys=True), "Compile this EventFact exactly into one patch."]
    else:
        lines += ["The Runtime invokes you only after it has found one explicitly licensed Base-state update. Therefore return a patch, not NO_PATCH."]
    lines += [
        "Write exactly one literal-only World Code patch and no prose.",
        "Your output must be one Python-like function call beginning with set_state(; it is not JSON.",
        "Never output a JSON object, JSON array, key-value list, or explanatory text.",
        "Required grammar:",
        "set_state(entity='<stable_id>', component='<place|openness|toggle_state>', expected_old=<current value or '__UNTRACKED__'>, new=<typed value>)",
        "Format-only example:",
        "set_state(entity='drawer_1', component='openness', expected_old='__UNTRACKED__', new='open')",
    ]
    if isinstance(event, StructuredEvent):
        expected = _structured_expected(event)
        if expected is None:
            raise PatchError("Structured Programmer received an event without a licensed Base-state patch")
        entity_id, component, new = expected
        skeleton = _patch_skeleton(
            entity_id=entity_id,
            component=component,
            current=runtime.read(entity_id=entity_id, component=component),
            new=new,
        )
        lines += [
            "STRUCTURED FAST-PATH COMPILATION:",
            "The public action contract and current Runtime state determine the following complete patch. This is source code, not a JSON schema or an illustrative example.",
            "Copy this exact line as your entire answer:",
            skeleton,
            "The first character of your answer must be s in set_state. Do not return JSON, a JSON array, NO_PATCH, or a Markdown explanation.",
        ]
    return "\n".join(lines)


def _review_prompt(runtime: TrustedRuntime, event: PublicEvent, source: str, reasons: Sequence[str], *, fact: EventFact | None = None) -> str:
    lines = ["You are a World Code reviewer. Repair the candidate using only public evidence.", public_transition_prompt(runtime.representation), "Current Active State:", json.dumps(runtime.public_view(), sort_keys=True), "Public event:", json.dumps(event.to_dict(), sort_keys=True)]
    if fact is not None:
        slot = f"{fact.entity_id}|{fact.component}"
        current = runtime.read(entity_id=fact.entity_id, component=fact.component)
        skeleton = _patch_skeleton(entity_id=fact.entity_id, component=fact.component, current=current, new=fact.new)
        lines += ["Typed Grounder EventFact:", json.dumps(fact.to_dict(), sort_keys=True), f"Current slot value: {slot} = {canonical_json(current)}", "Candidate patch:", source, "Runtime rejection reasons:", json.dumps(list(reasons)), "A full runtime-valid repair is required for this typed EventFact. Do not return NO_PATCH.", "Copy this complete repair skeleton exactly, changing nothing except syntax if necessary:", skeleton, "Return only that one complete set_state(...) call. Never return JSON, a JSON array, or prose."]
        return "\n".join(lines)
    if isinstance(event, StructuredEvent):
        expected = _structured_expected(event)
        if expected is None:
            raise PatchError("Structured reviewer received an event without a licensed Base-state patch")
        entity_id, component, new = expected
        current = runtime.read(entity_id=entity_id, component=component)
        skeleton = _patch_skeleton(entity_id=entity_id, component=component, current=current, new=new)
        lines += [
            "Candidate patch:", source,
            "Runtime rejection reasons:", json.dumps(list(reasons)),
            "This is a Structured fast-path repair. The following slot and new value are derived only from the published Structured action contract and the current Runtime state; they are not private labels.",
            f"Current slot value: {entity_id}|{component} = {canonical_json(current)}",
            "Required exact repair:", skeleton,
            "Copy the required exact repair as your entire answer. Its first character must be s in set_state; do not return NO_PATCH, JSON, a JSON array, or prose.",
        ]
        return "\n".join(lines)
    lines += ["Candidate patch:", source, "Runtime rejection reasons:", json.dumps(list(reasons)), "Return either exactly NO_PATCH (only if no safe patch exists) or one complete literal-only patch with all four required fields:", "set_state(entity='<stable_id>', component='<place|openness|toggle_state>', expected_old=<current value or '__UNTRACKED__'>, new=<typed value>)", "Do not omit expected_old. Do not return prose."]
    return "\n".join(lines)


def _audit_prompt(representation: str, event: PublicEvent, patch: Patch, *, fact: EventFact | None = None) -> str:
    lines = ["You are an evidence auditor. Decide whether the candidate patch is supported by this public event.", public_transition_prompt(representation), "Event:", json.dumps(event.to_dict(), sort_keys=True)]
    if fact is not None:
        lines += ["Grounder EventFact:", json.dumps(fact.to_dict(), sort_keys=True)]
    lines += ["Candidate:", json.dumps(patch.to_dict(), sort_keys=True), 'Return exactly {"verdict":"support"} or {"verdict":"reject"}.']
    return "\n".join(lines)


class WorldCodeMemory(MemoryMethod):
    name = "wcm"

    def __init__(self, backend: TextBackend, *, representation: str = "structured") -> None:
        if representation not in REPRESENTATIONS:
            raise ValueError("unsupported WCM representation")
        self.backend, self.representation = backend, representation
        self.runtime = TrustedRuntime(representation=representation)
        self.role_counts: dict[str, int] = {}
        self.reviewer_stats: dict[str, int] = {}
        self.grounder_stats: dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        self.runtime.reset()
        self.role_counts = {"grounder": 0, "programmer": 0, "reviewer": 0, "auditor": 0}
        self.reviewer_stats = {"reviewer_invocation_count": 0, "reviewer_parse_failure_count": 0, "reviewer_repair_success_count": 0, "reviewer_repair_failure_count": 0}
        self.grounder_stats = {"grounder_parse_failure_count": 0, "grounder_no_fact_count": 0, "grounder_fact_count": 0}

    def _parse_or_none(self, source: str, *, patch_id: str) -> Patch | None:
        return None if unwrap_single_markdown_fence(source) == "NO_PATCH" else parse_patch(source, patch_id=patch_id, base_version=self.runtime.version)

    def _audit_and_commit(self, event: PublicEvent, patch: Patch, calls: list[Generation], *, fact: EventFact | None = None) -> None:
        auditor = self.backend.generate(role="evidence_auditor", system="You audit public evidence conservatively using only the published transition specification.", user=_audit_prompt(self.representation, event, patch, fact=fact), max_new_tokens=32)
        self.role_counts["auditor"] += 1
        calls.append(auditor)
        try:
            support = isinstance((verdict := json.loads(unwrap_single_markdown_fence(auditor.raw))), Mapping) and verdict == {"verdict": "support"}
        except (AnswerCodecError, json.JSONDecodeError):
            support = False
        if support:
            self.runtime.commit(event, patch, fact=fact)
        else:
            self.runtime.log.append({"operation": "auditor_rejected", "event_id": event.event_id, "event_fact": None if fact is None else fact.to_dict(), "patch": patch.to_dict(), "auditor_raw": auditor.raw})

    def _review(self, event: PublicEvent, *, fact: EventFact | None, candidate: str, reasons: Sequence[str], calls: list[Generation]) -> Patch | None:
        reviewer_system = (
            "You are a compiler for restricted World Code. Return exactly one literal-only set_state(...) repair using only the published transition specification; never return JSON."
            if isinstance(event, StructuredEvent)
            else "You review and repair restricted World Code using only the published transition specification."
        )
        reviewer = self.backend.generate(role="program_reviewer", system=reviewer_system, user=_review_prompt(self.runtime, event, candidate, reasons, fact=fact), max_new_tokens=192)
        self.role_counts["reviewer"] += 1
        self.reviewer_stats["reviewer_invocation_count"] += 1
        calls.append(reviewer)
        try:
            repaired = self._parse_or_none(reviewer.raw, patch_id=f"{event.event_id}-review")
        except (PatchError, AnswerCodecError):
            self.reviewer_stats["reviewer_parse_failure_count"] += 1
            self.reviewer_stats["reviewer_repair_failure_count"] += 1
            return None
        if repaired is None:
            self.reviewer_stats["reviewer_repair_failure_count"] += 1
            return None
        accepted, _repair_reasons = self.runtime.preflight(event, repaired, fact=fact)
        if accepted:
            self.reviewer_stats["reviewer_repair_success_count"] += 1
            return repaired
        self.reviewer_stats["reviewer_repair_failure_count"] += 1
        return None

    def _run_patch_pipeline(self, event: PublicEvent, *, fact: EventFact | None, calls: list[Generation]) -> None:
        programmer = self.backend.generate(role="world_programmer", system="You write one safe literal-only Python-like set_state(...) World Code patch using only the published transition specification. Never return JSON or a JSON array.", user=_program_prompt(self.runtime, event, fact=fact), max_new_tokens=192)
        self.role_counts["programmer"] += 1
        calls.append(programmer)
        try:
            patch = self._parse_or_none(programmer.raw, patch_id=f"{event.event_id}-program")
            accepted, reasons = (False, ["programmer returned NO_PATCH"]) if patch is None else self.runtime.preflight(event, patch, fact=fact)
        except (PatchError, AnswerCodecError) as exc:
            patch, accepted, reasons = None, False, [str(exc)]
        if accepted and patch is not None:
            if self.representation == "structured":
                self.runtime.commit(event, patch, fact=fact)
            else:
                self._audit_and_commit(event, patch, calls, fact=fact)
            return
        repaired = self._review(event, fact=fact, candidate=programmer.raw if patch is None else patch.source, reasons=reasons, calls=calls)
        if repaired is None:
            self.runtime.log.append({"operation": "proposal_refused", "event_id": event.event_id, "event_fact": None if fact is None else fact.to_dict(), "reasons": list(reasons)})
            return
        self._audit_and_commit(event, repaired, calls, fact=fact)

    def _step_natural(self, event: NaturalEvent) -> Sequence[Generation]:
        calls: list[Generation] = []
        grounder = self.backend.generate(role="entity_grounder", system="You convert public Natural event descriptions into typed facts without using any private state.", user=_ground_prompt(self.runtime, event), max_new_tokens=128)
        self.role_counts["grounder"] += 1
        calls.append(grounder)
        try:
            fact = parse_event_fact(grounder.raw)
        except (EventFactError, AnswerCodecError) as exc:
            self.grounder_stats["grounder_parse_failure_count"] += 1
            self.runtime.record_no_write(event, status="grounder_parse_failure", details={"error": str(exc), "grounder_raw": grounder.raw})
            return tuple(calls)
        if fact is None:
            self.grounder_stats["grounder_no_fact_count"] += 1
            self.runtime.record_no_write(event, status="grounder_no_fact", details={"grounder_raw": grounder.raw})
            return tuple(calls)


        if fact.entity_id not in event.entity_ids() or (fact.component == "place" and fact.new["target"] not in event.entity_ids()):
            self.grounder_stats["grounder_parse_failure_count"] += 1
            self.runtime.record_no_write(event, status="grounder_entity_provenance_failure", details={"event_fact": fact.to_dict()})
            return tuple(calls)
        self.grounder_stats["grounder_fact_count"] += 1
        self._run_patch_pipeline(event, fact=fact, calls=calls)
        return tuple(calls)

    def step(self, event: PublicEvent) -> Sequence[Generation]:
        self.runtime.register(event)
        if isinstance(event, NaturalEvent):
            return self._step_natural(event)
        if _structured_expected(event) is None:
            self.runtime.record_no_write(event, status=event.outcome if event.outcome != "success_changed" else "ignored_out_of_scope")
            return ()
        calls: list[Generation] = []
        self._run_patch_pipeline(event, fact=None, calls=calls)
        return tuple(calls)

    def answer(self, example: QueryExample) -> tuple[Any, Sequence[Generation]]:
        return self.runtime.read(entity_id=example.target_entity_id, component=example.target_component), ()

    def dump(self, scope: Sequence[str]) -> tuple[Mapping[str, Any], Sequence[Generation]]:
        return self.runtime.dump(scope), ()

    def trace(self) -> Mapping[str, Any]:
        committed = sum(item.get("receipt", {}).get("status") == "committed" for item in self.runtime.log)
        rejected = sum(item.get("receipt", {}).get("status") == "rejected" for item in self.runtime.log)
        return {"method": self.name, "representation": self.representation, "runtime": self.runtime.public_view(), "transaction_log_digest": state_digest({str(i): item for i, item in enumerate(self.runtime.log)}), "role_call_count": dict(self.role_counts), **self.grounder_stats, **self.reviewer_stats, "commit_count": committed, "runtime_rejection_count": rejected}
