"""Deterministic fake teacher: full pipeline without spending money."""

from __future__ import annotations

import json
import time

from .base import Candidate, TeacherRequest, TeacherResponse, ValidationSummary, raw_sha256_of
from .validate import validate_response

__all__ = ["FakeProvider"]


class FakeProvider:
    provider_name = "fake"

    def __init__(self, model: str = "fake-teacher-1", **kwargs) -> None:
        self.model = model

    async def predict(self, request: TeacherRequest) -> TeacherResponse:
        region = request.region.text
        cands: list[Candidate] = []
        if region:
            cands.append(Candidate(action="replace", replacement=region + "  # fake-edit"))
        cands.append(Candidate(action="noop", replacement=""))
        chosen = cands[: request.num_candidates]
        raw = json.dumps({"candidates": [c.model_dump() for c in chosen]}, sort_keys=True)
        _, rejected = validate_response(tuple(chosen), request)
        reasons: list[str] = [reason for record in rejected for reason in record["reasons"]]
        return TeacherResponse(
            provider=self.provider_name,
            model=self.model,
            timestamp=time.time(),
            state_id=request.state_id,
            candidates=tuple(chosen),
            prompt_tokens=len(request.serialized_state) // 4,
            output_tokens=sum(len(c.replacement) for c in chosen) // 4,
            reported_cost_usd=0.0,
            request_id=f"fake-{request.state_id}",
            raw_sha256=raw_sha256_of(raw),
            validation=ValidationSummary(ok=not reasons, reasons=tuple(reasons)),
        )
