"""Visual Fast-Path code synthesis and auditing.

An external visual Grounder supplies a typed :class:`GrounderFact`. The
Programmer expresses that fact as restricted world code, and the Auditor
checks that the code cannot change its visual evidence before Runtime commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from e0_5.grounder_protocol import GrounderFact
from .visual_wcm_runtime import ActiveStateRuntime, RuntimeViolation, StatePatch


FAST_PATH_AGENT_PROTOCOL_VERSION = "visual-wcm-fast-path-agents/v1"


class VisualGrounder(Protocol):
    def ground(
        self, *, before_image: str | Path, after_image: str | Path, action: str, entity_catalog: Iterable[dict[str, object]]
    ) -> GrounderFact | None:
        ...


class WorldCodeProgrammer(Protocol):
    name: str

    def propose(self, *, fact: GrounderFact, runtime: ActiveStateRuntime, event_id: str) -> str:
        ...


@dataclass(frozen=True, slots=True)
class AuditReceipt:
    verdict: str
    reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {"verdict": self.verdict, "reason": self.reason}


def _literal(value: object) -> str:
    """Stable Python-literal serialization for prompts and deterministic code."""

    return repr(value)


def _programmer_context(*, fact: GrounderFact, runtime: ActiveStateRuntime, event_id: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "base_version": runtime.version,
        "entity": fact.entity_id,
        "component": fact.component,
        "expected_old": runtime.current_value(entity_id=fact.entity_id, component=fact.component),
        "new": fact.new,
    }


def _code_from_context(context: dict[str, object]) -> str:
    return (
        "set_state(\n"
        f"    event_id={_literal(context['event_id'])},\n"
        f"    base_version={_literal(context['base_version'])},\n"
        f"    entity={_literal(context['entity'])},\n"
        f"    component={_literal(context['component'])},\n"
        f"    expected_old={_literal(context['expected_old'])},\n"
        f"    new={_literal(context['new'])},\n"
        ")"
    )


class DeterministicWorldCodeProgrammer:
    """Reference compiler used only to test executable Fast-Path semantics."""

    name = "deterministic-world-code-compiler"

    def propose(self, *, fact: GrounderFact, runtime: ActiveStateRuntime, event_id: str) -> str:
        return _code_from_context(_programmer_context(fact=fact, runtime=runtime, event_id=event_id))


class GroundedPatchAuditor:
    """Evidence + program auditor for the Fast Path.

    It deliberately does not resolve visual truth by itself.  It validates
    that the Coding Programmer preserved the Grounder's evidence exactly, and
    asks Runtime preflight to validate the current executable-state context.
    Thus a malformed or stale patch is rejected before it can pollute Active
    State.
    """

    name = "grounded-patch-auditor"

    def audit(self, *, fact: GrounderFact, patch: StatePatch, runtime: ActiveStateRuntime, event_id: str) -> AuditReceipt:
        if patch.event_id != event_id:
            return AuditReceipt("reject", "event_id_mismatch")
        if (patch.entity_id, patch.component, patch.new) != (fact.entity_id, fact.component, fact.new):
            return AuditReceipt("reject", "grounded_evidence_mismatch")
        try:
            runtime.preflight(patch)
        except RuntimeViolation as error:
            return AuditReceipt("reject", f"runtime_preflight:{error}")
        return AuditReceipt("support", None)
