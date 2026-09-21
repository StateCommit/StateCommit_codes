"""Visual WCM Fast-Path agents: grounded code synthesis and auditing.

The learned/Qwen visual Grounder is responsible for *perception* and emits a
typed :class:`GrounderFact`.  The Coding Programmer is separately responsible
for expressing that fact as restricted executable world code.  The Auditor
ensures code cannot silently change the visual evidence before the Runtime
performs its version/type/invariant checks and atomically commits it.

No class in this module imports evaluator labels, simulator metadata, plans,
or reveal frames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from e0_5.grounder_protocol import GrounderFact, GrounderProtocolError, parse_grounder_output
from e0_5.qwen_grounder import QwenFormalGrounder

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


class ParsedQwenVisualGrounder:
    """Adapt Qwen-VL raw text into the exact same GrounderFact interface."""

    name = "qwen2.5-vl-grounder"

    def __init__(
        self,
        *,
        model_path: str | Path,
        max_new_tokens: int = 96,
        prompt_version: str = "v3",
        use_change_crop: bool = False,
        device: str = "cuda",
    ) -> None:
        self._backend = QwenFormalGrounder(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            prompt_version=prompt_version,
            use_change_crop=use_change_crop,
            device=device,
        )
        self.invalid_fact_count = 0

    def ground(
        self, *, before_image: str | Path, after_image: str | Path, action: str, entity_catalog: Iterable[dict[str, object]]
    ) -> GrounderFact | None:
        raw = self._backend.ground(
            before_image=before_image, after_image=after_image, action=action, entity_catalog=entity_catalog
        )
        try:
            fact = parse_grounder_output(raw, action=action, entity_catalog=entity_catalog)
        except GrounderProtocolError:



            self.invalid_fact_count += 1
            return None
        return fact


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


def build_programmer_prompt(*, context: dict[str, object]) -> list[dict[str, str]]:
    """Frozen Qwen-Coder prompt: programming, not visual semantic inference."""

    specification = """You are the Coding Programmer in a persistent visual world-memory Fast Path.
The visual Grounder has already supplied a typed candidate fact. Do not reinterpret it, rename its entity or component, or infer any unobserved state. Compile the supplied fields into exactly one executable State Patch.

Return exactly one Python literal call with no Markdown and no explanation:
set_state(event_id=<str>, base_version=<int>, entity=<str>, component=<str>, expected_old=<str-or-dict>, new=<str-or-dict>)

All six fields are mandatory. Copy their values exactly. expected_old and new must be Python literals; string values must be quoted. No other functions, imports, variables, comments, code fences, or prose are permitted."""
    payload = json.dumps(context, ensure_ascii=False, sort_keys=True)
    return [
        {"role": "system", "content": specification},
        {"role": "user", "content": f"Compile this public grounded transition and current Runtime context:\n{payload}"},
    ]


class QwenCoderWorldCodeProgrammer:
    """Frozen Qwen2.5-Coder agent that writes one constrained State Patch."""

    name = "qwen2.5-coder-world-programmer"

    def __init__(self, *, model_path: str | Path, max_new_tokens: int = 192, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.backends.cudnn.enabled = False
        self._torch = torch
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Coding Programmer model path does not exist: {self.model_path}")
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype="auto", device_map={"": device}, local_files_only=True
        ).eval()
        self.device = device

    def propose(self, *, fact: GrounderFact, runtime: ActiveStateRuntime, event_id: str) -> str:
        context = _programmer_context(fact=fact, runtime=runtime, event_id=event_id)
        messages = build_programmer_prompt(context=context)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            generated = self.model.generate(
                **tokens,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output = generated[0, tokens.input_ids.shape[1] :]
        return self.tokenizer.decode(output, skip_special_tokens=True).strip()


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
