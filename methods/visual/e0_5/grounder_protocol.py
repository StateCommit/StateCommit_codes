"""Public-only Visual Grounder protocol for formal WCM-Grid v2.

The parser validates syntax, public entity identity, component/value *types*,
and action/schema compatibility.  It never determines which fact is true for
an image pair; that remains the visual model's responsibility and is scored
only by evaluator-side labels.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Iterable


class GrounderProtocolError(ValueError):
    """A malformed or schema-incompatible Grounder output."""


@dataclass(frozen=True, slots=True)
class GrounderFact:
    entity_id: str
    component: str
    new: str | dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {"entity_id": self.entity_id, "component": self.component, "new": self.new}


def _catalog_index(entity_catalog: Iterable[dict[str, object]]) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for entry in entity_catalog:
        entity_id = entry.get("entity_id")
        entity_type = entry.get("entity_type")
        if not isinstance(entity_id, str) or not isinstance(entity_type, str):
            raise GrounderProtocolError("public entity catalog is malformed")
        if entity_id in indexed:
            raise GrounderProtocolError(f"duplicate public entity ID: {entity_id}")
        indexed[entity_id] = entry
    return indexed


def _literal_call(text: str) -> dict[str, object]:
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as error:
        raise GrounderProtocolError("output is not a valid literal event_fact expression") from error
    body = tree.body
    if not isinstance(body, ast.Call) or not isinstance(body.func, ast.Name) or body.func.id != "event_fact":
        raise GrounderProtocolError("output must be exactly event_fact(...)")
    if body.args:
        raise GrounderProtocolError("event_fact positional arguments are forbidden")
    values: dict[str, object] = {}
    for keyword in body.keywords:
        if keyword.arg is None or keyword.arg in values:
            raise GrounderProtocolError("event_fact keyword arguments are malformed")
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError) as error:
            raise GrounderProtocolError("event_fact arguments must be literals") from error
    if set(values) != {"entity", "component", "new"}:
        raise GrounderProtocolError("event_fact requires exactly entity, component, new")
    return values


def parse_grounder_output(
    raw_output: str,
    *,
    action: str,
    entity_catalog: Iterable[dict[str, object]],
) -> GrounderFact | None:
    """Parse one public-schema fact or ``NO_FACT`` without executing code."""

    text = raw_output.strip()
    if text == "NO_FACT":
        return None
    values = _literal_call(text)
    entity_id, component, new = values["entity"], values["component"], values["new"]
    if not isinstance(entity_id, str) or not isinstance(component, str):
        raise GrounderProtocolError("entity and component must be strings")
    catalog = _catalog_index(entity_catalog)
    entry = catalog.get(entity_id)
    if entry is None:
        raise GrounderProtocolError(f"fact refers to non-public entity: {entity_id}")
    entity_type = entry["entity_type"]
    if component == "place":
        if entity_type not in {"ball", "key"}:
            raise GrounderProtocolError("only ball/key entities can have place")
        if not isinstance(new, dict) or set(new) != {"kind", "target"}:
            raise GrounderProtocolError("place value must be exactly {'kind','target'}")
        if new.get("kind") != "in_zone" or not isinstance(new.get("target"), str):
            raise GrounderProtocolError("place value must use kind='in_zone' and a string target")
        target = catalog.get(new["target"])
        if target is None or target.get("entity_type") != "zone":
            raise GrounderProtocolError("place target must be a public zone entity")
        normalized: str | dict[str, str] = {"kind": "in_zone", "target": new["target"]}
    elif component == "openness":
        if entity_type != "door" or new not in {"open", "closed"}:
            raise GrounderProtocolError("openness must be open/closed on a public door")
        normalized = new
    elif component == "toggle_state":
        if entity_type != "switch" or new not in {"on", "off"}:
            raise GrounderProtocolError("toggle_state must be on/off on a public switch")
        normalized = new
    else:
        raise GrounderProtocolError(f"unsupported component: {component!r}")



    if action == "drop" and component != "place":
        raise GrounderProtocolError("drop can only propose place")
    if action == "toggle" and component not in {"openness", "toggle_state"}:
        raise GrounderProtocolError("toggle can only propose openness/toggle_state")
    if action not in {"drop", "toggle"}:
        raise GrounderProtocolError(f"the formal Grounder is never called for action {action!r}")
    return GrounderFact(entity_id, component, normalized)


def _catalog_lines(entity_catalog: Iterable[dict[str, object]]) -> tuple[dict[str, dict[str, object]], str]:
    catalog = _catalog_index(entity_catalog)
    lines: list[str] = []
    for entity_id in sorted(catalog):
        entry = catalog[entity_id]
        entity_type = entry["entity_type"]
        if entity_type == "zone":
            lines.append(f"- {entity_id}: zone with {entry.get('pattern')} floor pattern")
        else:
            lines.append(f"- {entity_id}: {entry.get('color')} {entity_type}")
    return catalog, "\n".join(lines)


def build_grounder_prompt(*, action: str, entity_catalog: Iterable[dict[str, object]], version: str = "v1") -> str:
    """Create a frozen public prompt; no Oracle value or candidate label enters it."""

    _, entity_lines = _catalog_lines(entity_catalog)
    if version == "v3":
        return _build_grounder_prompt_v3(action=action, entity_lines=entity_lines)
    if version == "v2":
        return _build_grounder_prompt_v2(action=action, entity_lines=entity_lines)
    if version != "v1":
        raise GrounderProtocolError(f"unsupported Grounder prompt version: {version!r}")
    if action == "drop":
        rule = """
The executed action is DROP. Compare the same local scene before and after.
If one visible ball or key is newly placed on a recognizable zone floor, emit
that object's place fact with exactly {'kind':'in_zone','target':'zone_i'}.
Identify the object by its visible color and type, and identify the zone by
its public floor pattern. If no such placement is visually supported, output
NO_FACT.
""".strip()
    elif action == "toggle":
        rule = """
The executed action is TOGGLE. Inspect the same interacted cell before and
after. A door is open when its solid colored door glyph becomes a thin frame
with a passable opening; it is closed for the reverse change. A switch is on
when its colored inner panel is bright and contains a white center, and off
when the inner panel is dim/dark. Emit exactly one supported door openness or
switch toggle_state fact. If no corresponding visual change is supported,
output NO_FACT.
""".strip()
    else:
        raise GrounderProtocolError(f"unsupported Grounder action: {action!r}")
    return f"""
You are a visual transition Grounder for a persistent world-state system.

You receive Image 1 (BEFORE), one executed raw action, and Image 2 (AFTER).
The action indicates only an attempt; visual evidence must identify the
entity and resulting state. Do not describe the scene, infer hidden state,
or explain your reasoning.

PUBLIC ENTITY CATALOG
{entity_lines}

TRACKED WORLD SCHEMA
- movable ball/key: place = {{'kind':'in_zone','target':'zone_i'}}
- door: openness = 'open' or 'closed'
- switch: toggle_state = 'on' or 'off'

{rule}

OUTPUT CONTRACT
Return exactly one of:

NO_FACT

event_fact(entity='<public entity id>', component='<place|openness|toggle_state>', new=<typed value>)

Use Python literal quoting exactly as shown. No Markdown and no prose.
""".strip()


def _build_grounder_prompt_v2(*, action: str, entity_lines: str) -> str:
    """More explicit, action-specific visual procedure for validation-only v2."""

    if action == "drop":
        procedure = """
SILENT VISUAL PROCEDURE FOR DROP
1. Compare Image 1 and Image 2. Locate the one floor cell where a ball or
   key appears after the action. Ignore the two-tone grey agent marker.
2. In Image 2, identify that new object by BOTH shape/type (ball or key) and
   color. Match that pair to exactly one public entity ID; never choose an
   unrelated catalog entity merely because it has a similar color.
3. Read the floor pattern directly beneath/around that object in Image 2 and
   map it to its zone ID. Pattern guide: checker=alternating small squares;
   vertical=parallel vertical bars; diagonal=diagonal stripes; dots=repeated
   small dots; solid=uniform floor; border=outlined floor.
4. Emit that object's place fact. Its new value must use the AFTER zone.
""".strip()
    elif action == "toggle":
        procedure = """
SILENT VISUAL PROCEDURE FOR TOGGLE
1. Locate the one changed non-agent cell by comparing Image 1 and Image 2.
2. Decide its TYPE before its state: a door is a tall colored door glyph with
   a dark handle/opening; a switch is a dark square control with a smaller
   colored inner panel.
3. For a DOOR: a closed door is a solid colored barrier with a dark handle;
   an open door is a thin colored frame/edge with a passable dark/grey gap.
   Output the state visible in Image 2, not Image 1.
4. For a SWITCH: ON has a bright colored inner panel and white center circle;
   OFF has a dim colored panel and dark center. Output the state in Image 2.
5. Match the changed glyph's type plus color to the public entity catalog.
   Do not select a different nearby door/switch solely because it shares a
   color or has a more familiar name.
""".strip()
    else:
        raise GrounderProtocolError(f"unsupported Grounder action: {action!r}")
    return f"""
You are the Visual Grounder for a persistent world-state system.

Image 1 is BEFORE. The action between images is `{action}`. Image 2 is AFTER.
The action is only an attempt: make a fact only when the corresponding visual
change is visible. Reason silently using the procedure below, then output one
single exact record and nothing else.

PUBLIC ENTITY CATALOG
{entity_lines}

TRACKED SCHEMA
- ball/key: component='place', new={{'kind':'in_zone','target':'zone_i'}}
- door: component='openness', new='open' or new='closed'
- switch: component='toggle_state', new='on' or new='off'

{procedure}

VALID OUTPUT EXAMPLES (illustrative IDs only; do not copy them unless visible)
event_fact(entity='red_ball_1', component='place', new={{'kind':'in_zone','target':'zone_3'}})
event_fact(entity='blue_door_1', component='openness', new='open')
event_fact(entity='green_switch_1', component='toggle_state', new='off')

If the required visual evidence is absent or ambiguous, output exactly NO_FACT.
Otherwise output exactly one `event_fact(entity=..., component=..., new=...)`.
Do not output JSON, a list, Markdown, a bare value, or an explanation.
""".strip()


def _build_grounder_prompt_v3(*, action: str, entity_lines: str) -> str:
    """Full-frame-only visual procedure focused on the action-relevant change.

    This is intentionally a prompt-only revision: it receives exactly the two
    public RGB frames, the raw action, and the public catalog.  In particular,
    it does not use a crop, a difference image, a tile coordinate, simulator
    metadata, or an evaluator-side state label.
    """

    if action == "drop":
        procedure = """
VISUAL CHECKLIST — DROP
First compare the complete BEFORE and AFTER images silently. Ignore the fixed
two-tone grey agent marker and static walls. Focus only on a colored ball or
key whose visible location changes because of the DROP action.

Then use the AFTER image as the source of truth:
1. Identify the moved object by its exact visible color AND type (ball/key),
   then bind it to the matching public entity ID.
2. Inspect the floor cell directly beneath that object. Match its visible
   pattern to the catalog zone: checker, vertical bars, diagonal stripes,
   dots, solid fill, or border.
3. Output exactly that object's `place` value using the AFTER zone.

Do not infer a destination from the action name or choose a catalog object
that is not visibly involved in the before/after change.
""".strip()
    elif action == "toggle":
        procedure = """
VISUAL CHECKLIST — TOGGLE
First compare the complete BEFORE and AFTER images silently. Ignore the fixed
two-tone grey agent marker and static walls. Find the changed non-agent glyph
that is consistent with a TOGGLE action; do not choose a different same-color
object elsewhere in the image.

Classify the changed glyph before reading its state:
1. A DOOR is a tall vertical colored barrier positioned in a wall passage.
   Read ONLY its AFTER appearance: a solid colored barrier closing the passage
   means `openness='closed'`; a thin colored edge/frame leaving a visible
   passable gap means `openness='open'`.
2. A SWITCH is a compact square control with a smaller colored inner panel.
   Read ONLY its AFTER appearance: a bright panel with a white center means
   `toggle_state='on'`; a dim/dark panel with a dark center means
   `toggle_state='off'`.
3. Bind the changed door or switch using BOTH its type and color in the public
   entity catalog, then output its one supported fact.

The raw action is only an attempted toggle. If no corresponding changed door
or switch can be visually verified, output NO_FACT.
""".strip()
    else:
        raise GrounderProtocolError(f"unsupported Grounder action: {action!r}")

    return f"""
You are the Visual Grounder for a persistent world-state system.

You receive exactly two full public RGB images: Image 1 (BEFORE) and Image 2
(AFTER). The executed raw action between them is `{action}`. No crop, pixel
difference image, tile coordinate, simulator metadata, or hidden state is
available. Reason silently using the checklist below, then emit one exact
record and nothing else.

PUBLIC ENTITY CATALOG
{entity_lines}

TRACKED SCHEMA
- ball/key: component='place', new={{'kind':'in_zone','target':'zone_i'}}
- door: component='openness', new='open' or new='closed'
- switch: component='toggle_state', new='on' or new='off'

{procedure}

OUTPUT CONTRACT
Return exactly one of:
NO_FACT
event_fact(entity='<public entity id>', component='<place|openness|toggle_state>', new=<typed value>)

Examples of exact syntax only (do not copy IDs unless visually supported):
event_fact(entity='red_ball_1', component='place', new={{'kind':'in_zone','target':'zone_3'}})
event_fact(entity='blue_door_1', component='openness', new='open')
event_fact(entity='green_switch_1', component='toggle_state', new='off')

Do not output JSON, Markdown, a list, a bare value, scene description, or an
explanation.
""".strip()
