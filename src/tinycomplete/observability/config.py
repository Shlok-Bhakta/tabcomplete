"""Configuration for live, offline, and disabled telemetry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ObservabilityConfig:
    enabled: bool = False
    mode: Literal["disabled", "live", "offline"] = "disabled"
    service_name: str = "tabcomplete"
    service_version: str = "0.1.0"
    deployment_environment: str = "development"
    otlp_endpoint: str = "http://127.0.0.1:4318"
    export_timeout_seconds: float = 5.0
    artifact_root: Path = Path(".tabcomplete-observability/artifacts")
    artifact_max_payload_bytes: int = 8 * 2**20
    capture_content: bool = False
    offline_bundle: Path = Path(".tabcomplete-observability/offline.jsonl")
    offline_max_bytes: int = 256 * 2**20
    heartbeat_seconds: float = 15.0

    @classmethod
    def from_env(cls) -> ObservabilityConfig:
        requested_mode = os.environ.get("TABCOMPLETE_OBSERVABILITY_MODE", "live").lower()
        enabled = _truthy(os.environ.get("TABCOMPLETE_OBSERVABILITY_ENABLED"))
        if not enabled:
            mode: Literal["disabled", "live", "offline"] = "disabled"
        elif requested_mode in {"live", "offline"}:
            mode = "live" if requested_mode == "live" else "offline"
        else:
            raise ValueError("TABCOMPLETE_OBSERVABILITY_MODE must be live or offline")
        return cls(
            enabled=enabled,
            mode=mode,
            service_name=os.environ.get("OTEL_SERVICE_NAME", "tabcomplete"),
            service_version=os.environ.get("TABCOMPLETE_SERVICE_VERSION", "0.1.0"),
            deployment_environment=os.environ.get(
                "TABCOMPLETE_DEPLOYMENT_ENVIRONMENT", "development"
            ),
            otlp_endpoint=os.environ.get(
                "OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318"
            ).rstrip("/"),
            export_timeout_seconds=float(
                os.environ.get("TABCOMPLETE_OBSERVABILITY_EXPORT_TIMEOUT_SECONDS", "5")
            ),
            artifact_root=Path(
                os.environ.get(
                    "TABCOMPLETE_OBSERVABILITY_ARTIFACT_ROOT",
                    ".tabcomplete-observability/artifacts",
                )
            ),
            artifact_max_payload_bytes=int(
                os.environ.get("TABCOMPLETE_OBSERVABILITY_ARTIFACT_MAX_BYTES", str(8 * 2**20))
            ),
            capture_content=_truthy(os.environ.get("TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT")),
            offline_bundle=Path(
                os.environ.get(
                    "TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE",
                    ".tabcomplete-observability/offline.jsonl",
                )
            ),
            offline_max_bytes=int(
                os.environ.get("TABCOMPLETE_OBSERVABILITY_OFFLINE_MAX_BYTES", str(256 * 2**20))
            ),
            heartbeat_seconds=float(
                os.environ.get("TABCOMPLETE_OBSERVABILITY_HEARTBEAT_SECONDS", "15")
            ),
        )

    def __post_init__(self) -> None:
        if self.mode == "disabled" and self.enabled:
            raise ValueError("enabled observability needs live or offline mode")
        if self.mode != "disabled" and not self.enabled:
            raise ValueError("disabled observability cannot select an exporter mode")
        if self.export_timeout_seconds <= 0:
            raise ValueError("export timeout must be positive")
        if self.artifact_max_payload_bytes <= 0 or self.offline_max_bytes <= 0:
            raise ValueError("storage limits must be positive")
        if self.heartbeat_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
