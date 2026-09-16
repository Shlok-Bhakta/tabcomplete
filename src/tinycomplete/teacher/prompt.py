"""Teacher prompt builder: narrow region-rewrite labeling instructions."""

from __future__ import annotations

from .base import TeacherRequest

__all__ = ["SYSTEM_PROMPT", "RESPONSE_SCHEMA", "build_messages"]

SYSTEM_PROMPT = (
    """You are producing training labels for a small code next-edit model.
You are NOT writing an explanation.
Given repository context, recent edits, current file context, and one editable
region, predict plausible next versions of that editable region.
Rules:
1. Treat recent edits as intentional unless they are syntactically impossible.
2. Do not undo the user's immediately preceding edit merely because an older version looked cleaner.
3. Preserve unrelated code.
4. Do not edit outside the supplied editable region.
"""
    "5. Do not invent project APIs not present in the supplied context unless they are "
    "obvious standard-library APIs.\n"
    """6. Prefer small likely edits over broad rewrites.
7. NOOP is valid when no edit is justified.
8. Return exactly the requested JSON schema.
9. Do not include Markdown.
10. Do not include rationale or prose.
11. Produce distinct candidates when multiple candidates are requested."""
)

RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["replace", "noop"]},
                    "replacement": {"type": "string"},
                },
                "required": ["action", "replacement"],
                "additionalProperties": False,
            },
            "minItems": 1,
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


def build_messages(request: TeacherRequest) -> list[dict[str, str]]:
    recent = "\n".join(request.recent_edits) if request.recent_edits else "(none)"
    user = (
        f"Repository context:\n{request.repo_context or '(none)'}\n\n"
        f"Recent edits:\n{recent}\n\n"
        f"Current file context (serialized editor state):\n{request.serialized_state}\n\n"
        f"Editable region in {request.region.path or 'file'} "
        f"[bytes {request.region.start}:{request.region.end}]:\n"
        f"<<<REGION\n{request.region.text}\nREGION>>>\n\n"
        f"Produce exactly {request.num_candidates} distinct candidate(s) as JSON."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
