"""Deterministic, executable causal code-completion benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import signal
import statistics
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, field_validator

CheckStatus = Literal["pass", "fail", "timeout", "error", "unavailable", "not_run"]


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("benchmark files must use a safe relative path")
    return str(path)


class CheckSpec(BaseModel):
    compile: list[str] | None = None
    test: list[str] | None = None
    files: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    container_image: str | None = None

    @field_validator("files")
    @classmethod
    def validate_files(cls, value: dict[str, str]) -> dict[str, str]:
        return {_safe_relative_path(path): content for path, content in value.items()}


class BenchmarkCase(BaseModel):
    id: str
    language: Literal[
        "python", "typescript", "javascript", "java", "cpp", "rust", "go", "c", "csharp"
    ]
    path: str
    prefix: str
    suffix: str = ""
    expected: str = Field(min_length=1)
    context_files: dict[str, str] = Field(default_factory=dict)
    check: CheckSpec = Field(default_factory=CheckSpec)
    category: str = "unspecified"
    repository_context: bool = False

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _safe_relative_path(value)

    @field_validator("context_files")
    @classmethod
    def validate_context_files(cls, value: dict[str, str]) -> dict[str, str]:
        return {_safe_relative_path(path): content for path, content in value.items()}


class Prediction(BaseModel):
    case_id: str
    completion: str
    latency_seconds: float | None = None
    generated_tokens: int | None = None


class CheckResult(BaseModel):
    status: CheckStatus
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    seconds: float | None = None


class BenchmarkResult(BaseModel):
    case_id: str
    language: str
    category: str
    repository_context: bool
    exact_match: bool
    normalized_exact_match: bool
    nonempty: bool
    compile_configured: bool
    test_configured: bool
    parse: CheckResult
    compile: CheckResult
    test: CheckResult
    working_tree_sha256: str
    latency_seconds: float | None = None
    generated_tokens: int | None = None


def load_suite(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            case = BenchmarkCase.model_validate_json(line)
            if case.id in seen:
                raise ValueError(f"duplicate benchmark id at line {line_number}: {case.id}")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError("benchmark suite is empty")
    return cases


def _parse(code: str, language: str) -> CheckResult:
    grammar = {"csharp": "csharp"}.get(language, language)
    try:
        from tree_sitter_language_pack import get_parser

        root = get_parser(grammar).parse(code.encode()).root_node
    except Exception as exc:
        return CheckResult(status="unavailable", stderr=type(exc).__name__)
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            return CheckResult(status="fail")
        stack.extend(node.children)
    return CheckResult(status="pass")


def _limits(timeout_seconds: float, *, limit_address_space: bool = True):
    def apply() -> None:
        cpu_seconds = max(1, int(timeout_seconds) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        if limit_address_space:
            resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20, 512 * 2**20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 2**20, 8 * 2**20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
        except (OSError, ValueError):
            pass

    return apply


def _bounded_text(path: Path, limit: int = 16_384) -> str:
    data = path.read_bytes()[:limit]
    return data.decode("utf-8", errors="replace")


def _run_trusted(
    command: list[str],
    cwd: Path,
    timeout_seconds: float,
    *,
    limit_address_space: bool = True,
    apply_resource_limits: bool = True,
    preserve_environment: bool = False,
) -> CheckResult:
    if not command or shutil.which(command[0]) is None:
        return CheckResult(status="unavailable")
    env = (
        os.environ.copy()
        if preserve_environment
        else {
            "HOME": str(cwd),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONHASHSEED": "0",
        }
    )
    with (
        tempfile.NamedTemporaryFile(dir=cwd) as stdout,
        tempfile.NamedTemporaryFile(dir=cwd) as stderr,
    ):
        import time

        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            preexec_fn=(
                _limits(timeout_seconds, limit_address_space=limit_address_space)
                if apply_resource_limits
                else None
            ),
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
            status: CheckStatus = "pass" if returncode == 0 else "fail"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            returncode = None
            status = "timeout"
        stdout.flush()
        stderr.flush()
        return CheckResult(
            status=status,
            returncode=returncode,
            stdout=_bounded_text(Path(stdout.name)),
            stderr=_bounded_text(Path(stderr.name)),
            seconds=time.perf_counter() - started,
        )


def container_command(
    *,
    runtime: str,
    image: str,
    work_root: Path,
    fixture_command: list[str],
    timeout_seconds: float,
) -> list[str]:
    """Build a no-network, rootless OCI command for untrusted generated code."""
    del timeout_seconds  # enforced both by OCI resource flags and the parent watchdog
    return [
        runtime,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=all",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=768m",
        "--cpus=1",
        "--userns=keep-id",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
        f"--volume={work_root.resolve()}:/workspace:rw,Z",
        "--workdir=/workspace",
        image,
        *fixture_command,
    ]


def _container_runtime() -> str | None:
    for name in ("podman", "docker"):
        if shutil.which(name):
            return name
    return None


def _run_container(
    command: list[str], cwd: Path, timeout_seconds: float, image: str | None
) -> CheckResult:
    runtime = _container_runtime()
    if runtime is None or image is None:
        return CheckResult(status="unavailable", stderr="container runtime/image unavailable")
    inspect = subprocess.run(
        [runtime, "image", "inspect", image],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if inspect.returncode:
        return CheckResult(status="unavailable", stderr=f"container image not installed: {image}")
    container_timeout = max(30.0, timeout_seconds)
    return _run_trusted(
        container_command(
            runtime=runtime,
            image=image,
            work_root=cwd,
            fixture_command=command,
            timeout_seconds=container_timeout,
        ),
        cwd,
        container_timeout + 2,
        limit_address_space=False,
        apply_resource_limits=False,
        preserve_environment=True,
    )


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_file(root: Path, relative: str, content: str) -> None:
    path = root / _safe_relative_path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def evaluate_prediction(
    case: BenchmarkCase,
    prediction: Prediction,
    *,
    work_root: Path,
    execution_backend: Literal["none", "container", "trusted-host"] = "none",
) -> BenchmarkResult:
    if prediction.case_id != case.id:
        raise ValueError(f"prediction id {prediction.case_id!r} does not match case {case.id!r}")
    work_root.mkdir(parents=True, exist_ok=True)
    for path, content in case.context_files.items():
        _write_file(work_root, path, content)
    for path, content in case.check.files.items():
        _write_file(work_root, path, content)
    assembled = case.prefix + prediction.completion + case.suffix
    _write_file(work_root, case.path, assembled)
    fixture_sha256 = _tree_hash(work_root)
    parse = _parse(assembled, case.language)
    compile_result = CheckResult(status="not_run")
    test_result = CheckResult(status="not_run")
    if execution_backend in {"container", "trusted-host"}:
        if case.check.compile:
            if execution_backend == "container":
                compile_result = _run_container(
                    case.check.compile,
                    work_root,
                    case.check.timeout_seconds,
                    case.check.container_image,
                )
            else:
                compile_result = _run_trusted(
                    case.check.compile, work_root, case.check.timeout_seconds
                )
        if case.check.test and compile_result.status in {"pass", "not_run"}:
            if execution_backend == "container":
                test_result = _run_container(
                    case.check.test,
                    work_root,
                    case.check.timeout_seconds,
                    case.check.container_image,
                )
            else:
                test_result = _run_trusted(case.check.test, work_root, case.check.timeout_seconds)
    return BenchmarkResult(
        case_id=case.id,
        language=case.language,
        category=case.category,
        repository_context=case.repository_context,
        exact_match=prediction.completion == case.expected,
        normalized_exact_match=(prediction.completion.strip() == case.expected.strip()),
        nonempty=bool(prediction.completion.strip()),
        compile_configured=case.check.compile is not None,
        test_configured=case.check.test is not None,
        parse=parse,
        compile=compile_result,
        test=test_result,
        working_tree_sha256=fixture_sha256,
        latency_seconds=prediction.latency_seconds,
        generated_tokens=prediction.generated_tokens,
    )


def _rate(results: list[BenchmarkResult], attribute: str) -> float | None:
    values = [bool(getattr(result, attribute)) for result in results]
    return sum(values) / len(values) if values else None


def _check_rate(
    results: list[BenchmarkResult], attribute: str, configured_attribute: str
) -> float | None:
    eligible = [row for row in results if getattr(row, configured_attribute)]
    if not eligible:
        return None
    return sum(getattr(row, attribute).status == "pass" for row in eligible) / len(eligible)


def _availability_rate(
    results: list[BenchmarkResult], attribute: str, configured_attribute: str
) -> float | None:
    eligible = [row for row in results if getattr(row, configured_attribute)]
    if not eligible:
        return None
    return sum(
        getattr(row, attribute).status not in {"unavailable", "not_run"} for row in eligible
    ) / len(eligible)


def summarize_results(results: list[BenchmarkResult]) -> dict:
    def summary(rows: list[BenchmarkResult]) -> dict:
        latencies = [row.latency_seconds for row in rows if row.latency_seconds is not None]
        generated_tokens = [
            row.generated_tokens for row in rows if row.generated_tokens is not None
        ]
        return {
            "total": len(rows),
            "exact_match_rate": _rate(rows, "exact_match"),
            "normalized_exact_match_rate": _rate(rows, "normalized_exact_match"),
            "nonempty_rate": _rate(rows, "nonempty"),
            "parse_pass_rate": sum(row.parse.status == "pass" for row in rows) / len(rows)
            if rows
            else None,
            "compile_pass_rate": _check_rate(rows, "compile", "compile_configured"),
            "compile_availability_rate": _availability_rate(rows, "compile", "compile_configured"),
            "test_pass_rate": _check_rate(rows, "test", "test_configured"),
            "test_availability_rate": _availability_rate(rows, "test", "test_configured"),
            "median_latency_seconds": statistics.median(latencies) if latencies else None,
            "median_generated_tokens": (
                statistics.median(generated_tokens) if generated_tokens else None
            ),
            "timeout_count": sum(
                check.status == "timeout" for row in rows for check in (row.compile, row.test)
            ),
        }

    overall = summary(results)
    overall["by_language"] = {
        language: summary([row for row in results if row.language == language])
        for language in sorted({row.language for row in results})
    }
    overall["by_repository_context"] = {
        "repository_context": summary([row for row in results if row.repository_context]),
        "standalone": summary([row for row in results if not row.repository_context]),
    }
    overall["by_category"] = {
        category: summary([row for row in results if row.category == category])
        for category in sorted({row.category for row in results})
    }
    return overall


def write_results(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.model_dump(mode="json"), sort_keys=True) + "\n")
