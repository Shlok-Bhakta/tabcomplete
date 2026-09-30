from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_sweep_comparison.py"
_SPEC = importlib.util.spec_from_file_location("run_sweep_comparison", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)


def test_vmstat_parser_reads_page_counters(tmp_path: Path) -> None:
    path = tmp_path / "vmstat"
    path.write_text("nr_free_pages 10\npswpin 123\npswpout 456\n", encoding="utf-8")

    assert sweep._proc_page_fields(path, {"pswpin", "pswpout"}) == {
        "pswpin": 123,
        "pswpout": 456,
    }


def test_memory_report_uses_process_swap_field() -> None:
    server = object.__new__(sweep.Server)
    server.samples = [
        {
            "rss_bytes": 100,
            "rss_high_water_bytes": 110,
            "pss_bytes": 90,
            "anonymous_bytes": 50,
            "private_bytes": 40,
            "process_swap_bytes": 7,
        },
        {
            "rss_bytes": 120,
            "rss_high_water_bytes": 130,
            "pss_bytes": 100,
            "anonymous_bytes": 55,
            "private_bytes": 45,
            "process_swap_bytes": 9,
        },
    ]

    report = server.memory_report()

    assert report["peak"]["process_swap_bytes"] == 9
    assert "swap_bytes" not in report["peak"]
    assert report["post_request_retained"]["process_swap_bytes"] == 9


def test_runtime_backend_requires_log_confirmed_cuda_offload(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    gpu = {"available": True, "devices": "0, Tesla T4, 8000, 15360"}
    log.write_text(
        "ggml_cuda_init: found 1 CUDA devices:\nllama_model_load: offloaded 29/29 layers to GPU\n",
        encoding="utf-8",
    )

    evidence = sweep._runtime_backend_evidence(log, gpu)
    assert evidence["backend"] == "CUDA"
    assert evidence["cuda_device_count"] == 1
    assert evidence["offloaded_layers"] == 29
    assert evidence["offloadable_layers"] == 29
    assert evidence["nvidia_smi_snapshot"] == gpu

    log.write_text("llama_model_load: offloaded 0/29 layers to GPU\n", encoding="utf-8")
    no_offload = sweep._runtime_backend_evidence(log, gpu)
    assert no_offload["backend"] == "unverified"
    assert no_offload["offloaded_layers"] == 0


@pytest.mark.parametrize("offloaded", [True, False])
def test_startup_enables_backend_logs_and_preserves_guard_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offloaded: bool
) -> None:
    import httpx

    stopped = []

    class Process:
        pid = 12345

        def poll(self):
            return None

        def terminate(self):
            stopped.append(True)

        def wait(self, timeout):
            return 0

    def launch(argv, **kwargs):
        level = int(argv[argv.index("--log-verbosity") + 1])
        if level >= 4:
            layers = 29 if offloaded else 0
            kwargs["stdout"].write(
                (f"ggml_cuda_init: found 1 CUDA devices:\n"
                 f"llama_model_load: offloaded {layers}/29 layers to GPU\n").encode()
            )
            kwargs["stdout"].flush()
        return Process()

    monkeypatch.setattr(sweep.subprocess, "Popen", launch)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: SimpleNamespace(status_code=200))
    monkeypatch.setattr(sweep, "_host_snapshot", lambda: {})
    monkeypatch.setattr(sweep, "_gpu_snapshot", lambda: {"available": True})
    monkeypatch.setattr(sweep, "deadline_remaining", lambda: 6000)
    monkeypatch.setattr(sweep.Server, "sample", lambda self: None)
    args = SimpleNamespace(runtime=tmp_path, output=tmp_path, gpu_layers=99)
    if offloaded:
        with sweep.Server(args, tmp_path / "model.gguf") as server:
            assert server.backend_evidence["offloaded_layers"] == 29
    else:
        with pytest.raises(RuntimeError, match="layers offloaded"):
            with sweep.Server(args, tmp_path / "model.gguf"):
                pytest.fail("an unverified backend must not enter inference")
    evidence = list(tmp_path.glob("*.backend-startup.json"))
    assert len(evidence) == 1
    record = json.loads(evidence[0].read_text())
    assert record["backend_evidence"]["offloaded_layers"] == (29 if offloaded else 0)
    assert stopped == [True]


def test_swap_delta_retains_page_units_and_negative_reset_evidence() -> None:
    assert sweep._swap_activity_delta(
        {"swap_activity_pages": {"pswpin": 10, "pswpout": 20}},
        {"swap_activity_pages": {"pswpin": 12, "pswpout": 29}},
    ) == {"pswpin": 2, "pswpout": 9}
    assert sweep._swap_activity_delta(
        {"swap_activity_pages": {"pswpin": 10}},
        {"swap_activity_pages": {"pswpin": 2}},
    ) == {"pswpin": -8}


def test_full_file_mapping_requires_unchanged_utf8_byte_range() -> None:
    current = "name = '雪'\nkeep = True\n"
    start = len(b"name = ")
    end = start + len("'雪'".encode())

    mapped = sweep.map_full_file(current, "name = '月'\nkeep = True\n", start, end)
    assert mapped["mapping"] == "within_editable_range"
    assert mapped["replacement"] == "'月'"
    assert (
        sweep.map_full_file(current, "changed = 1\nkeep = True\n", start, end)["mapping"]
        == "out_of_range"
    )
    assert sweep.map_full_file(current, current, start + 3, end)["mapping"] == (
        "invalid_utf8_boundary"
    )


def test_publisher_prompt_order_is_explicit() -> None:
    prompt = sweep.build_sweep_prompt(
        {
            "context_files": {"lib.py": "helper()"},
            "recent_diffs": [{"file_path": "main.py", "original": "old", "updated": "new"}],
            "file_path": "main.py",
            "original_content": "before",
            "current_content": "after",
        }
    )
    assert prompt == (
        "<|file_sep|>lib.py\nhelper()\n"
        "<|file_sep|>main.py.diff\noriginal:\nold\nupdated:\nnew\n"
        "<|file_sep|>original/main.py\nbefore\n"
        "<|file_sep|>current/main.py\nafter\n"
        "<|file_sep|>updated/main.py"
    )


@pytest.mark.parametrize(
    ("stop_type", "truncated", "tokens", "terminal", "capped"),
    [
        ("eos", False, 512, True, False),
        ("word", False, 6, True, False),
        ("limit", True, 512, False, True),
        (None, False, 512, False, True),
        (None, False, 20, False, False),
    ],
)
def test_explicit_termination_is_distinct_from_token_cap(
    stop_type: str | None,
    truncated: bool,
    tokens: int,
    terminal: bool,
    capped: bool,
) -> None:
    assert sweep.terminal_status(stop_type, truncated, tokens, 512) == (terminal, capped)


def test_token_budget_ledger_is_idempotent_and_rejects_mutation(tmp_path: Path) -> None:
    ledger = tmp_path / "tokens.json"
    rows = [{"key": "q8/case-1", "input_tokens": 100, "output_ceiling": 96}]

    assert sweep.add_token_budget(ledger, "plan", rows) == 196
    assert sweep.add_token_budget(ledger, "plan", rows) == 196
    assert json.loads(ledger.read_text(encoding="utf-8"))["entries"] == {
        "q8/case-1": {"input_tokens": 100, "output_ceiling": 96}
    }
    with pytest.raises(ValueError, match="charge changed"):
        sweep.add_token_budget(
            ledger,
            "plan",
            [{"key": "q8/case-1", "input_tokens": 101, "output_ceiling": 96}],
        )


def test_streamed_generation_has_real_trace_ids_and_captures_public_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tinycomplete.observability.artifacts import ArtifactStore
    from tinycomplete.observability.bootstrap import initialize_observability
    from tinycomplete.observability.config import ObservabilityConfig
    from tinycomplete.observability.context import RunContext

    prompt = "<|file_sep|>sample.py\nprint('hello')"
    response_event = {
        "content": "print('hello')\n",
        "tokens": [10, 11, 12],
        "stop": True,
        "stop_type": "eos",
        "truncated": False,
        "timings": {"prompt_n": 8, "predicted_n": 3},
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self):
            yield "data: " + json.dumps(response_event)

    import httpx

    monkeypatch.setattr(httpx, "stream", lambda *_args, **_kwargs: Response())
    offline_bundle = tmp_path / "telemetry.jsonl"
    artifact_root = tmp_path / "artifacts"
    runtime = initialize_observability(
        ObservabilityConfig(
            enabled=True,
            mode="offline",
            offline_bundle=offline_bundle,
            artifact_root=artifact_root,
            capture_content=True,
        )
    )
    context = RunContext.new(campaign_id="campaign-synthetic").for_case("synthetic/sample")
    try:
        with runtime.activate():
            result = sweep._generate_sweep_case(
                SimpleNamespace(url="http://127.0.0.1:19000"),
                prompt,
                8,
                precision="q8_0",
                context=context,
                case_id="synthetic/sample",
            )
    finally:
        runtime.shutdown()

    records = [json.loads(line) for line in offline_bundle.read_text().splitlines()]
    model_span = next(record for record in records if record["name"] == "model.generate")
    assert result["observability_ids"]["request_id"] == context.request_id
    assert result["observability_ids"]["trace_id"] == model_span["trace_id"]
    assert len(result["observability_ids"]["trace_id"]) == 32
    assert model_span["attributes"]["tabcomplete.request_id"] == context.request_id
    assert model_span["attributes"]["tabcomplete.artifact.input.status"] == "captured"
    assert model_span["attributes"]["tabcomplete.artifact.output.status"] == "captured"
    store = ArtifactStore(artifact_root)
    assert store.read_text(model_span["attributes"]["tabcomplete.artifact.input.sha256"]) == prompt
    assert (
        store.read_text(model_span["attributes"]["tabcomplete.artifact.output.sha256"])
        == response_event["content"]
    )


def test_next_edit_reuses_one_model_hash_and_assigns_unique_request_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tinycomplete.observability import runs as runs_module
    from tinycomplete.observability.context import RunContext

    model = tmp_path / "model.gguf"
    model.write_bytes(b"pinned test model")
    runtime = tmp_path / "runtime" / "build" / "bin" / "llama-server"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"server")
    fixtures = tmp_path / "inputs.jsonl"
    rows = [
        {
            "case_id": case_id,
            "file_path": "sample.py",
            "current_content": "after\n",
            "original_content": "before\n",
            "context_files": {},
            "recent_diffs": [],
            "current_sha256": "c" * 64,
            "original_sha256": "d" * 64,
            "editable_start_byte": 0,
            "editable_end_byte": 0,
        }
        for case_id in sweep.EXPECTED_CASE_IDS
    ]
    fixtures.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    args = SimpleNamespace(
        next_edit_fixtures=fixtures,
        output=tmp_path / "out",
        runtime=runtime.parents[2],
    )
    plan = {
        "plan_sha256": "a" * 64,
        "comparison": {"next_edit": {"fixture_ids": list(sweep.EXPECTED_CASE_IDS)}},
    }
    request_ids: list[str | None] = []
    hash_calls: list[Path] = []

    @contextmanager
    def fake_run_scope(_path: Path, _phase: str):
        yield RunContext.new(campaign_id="campaign-test")

    class FakeServer:
        def __init__(self, _args, _model):
            self.argv = ["fake-server"]
            self.started_at = 0.0
            self.loaded_at = 1.0
            self.host_before = {}
            self.host_after = {}
            self.gpu_before = {}
            self.gpu_after_load = {}
            self.gpu_after_requests = {}
            self.backend_evidence = {"backend": "test"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def tokenize(self, _prompt: str) -> int:
            return 8

        def sample(self) -> None:
            return None

        def memory_report(self):
            return {"samples": 0}

    real_digest = sweep.digest_file

    def count_model_digest(path: Path) -> str:
        if path == model:
            hash_calls.append(path)
        return real_digest(path)

    def fake_generate(_server, _prompt, input_tokens, *, context, case_id, precision):
        request_ids.append(context.request_id)
        return {
            "input_tokens": input_tokens,
            "context_eligible": True,
            "explicit_terminal": False,
            "hit_token_cap": False,
            "observability_ids": {"request_id": context.request_id, "case_id": case_id},
            "precision": precision,
        }

    monkeypatch.setattr(runs_module, "run_scope", fake_run_scope)
    monkeypatch.setattr(sweep, "Server", FakeServer)
    monkeypatch.setattr(sweep, "_generate_sweep_case", fake_generate)
    monkeypatch.setattr(sweep, "digest_file", count_model_digest)
    monkeypatch.setattr(sweep, "_host_info", lambda: {"hostname": "synthetic-test"})

    sweep.run_next_edit(args, plan, model, "q8_0")
    assert len(hash_calls) == 1
    assert len(request_ids) == 48
    assert len(set(request_ids)) == 48

    sweep.run_next_edit(args, plan, model, "q8_0")
    assert len(hash_calls) == 2
    assert len(request_ids) == 48


def test_two_precision_dispatch_keeps_model_hashing_out_of_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib.metadata

    from tinycomplete.observability import bootstrap
    from tinycomplete.observability import runs as runs_module
    from tinycomplete.observability.context import RunContext

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    q8 = artifact_dir / sweep.MODEL_FILE
    q4 = artifact_dir / "sweep-next-edit-1.5b.q4_k_m.gguf"
    q8.write_bytes(b"test q8")
    q4.write_bytes(b"test q4")
    runtime = tmp_path / "runtime" / "build" / "bin" / "llama-server"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"test server")
    fixtures = tmp_path / "inputs.jsonl"
    rows = [
        {
            "case_id": case_id,
            "file_path": "sample.py",
            "current_content": "after\n",
            "original_content": "before\n",
            "context_files": {},
            "recent_diffs": [],
            "current_sha256": "c" * 64,
            "original_sha256": "d" * 64,
            "editable_start_byte": 0,
            "editable_end_byte": 0,
        }
        for case_id in sweep.EXPECTED_CASE_IDS
    ]
    fixtures.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("opentelemetry-api", "opentelemetry-sdk")
    }
    plan = {
        "plan_sha256": "b" * 64,
        "comparison": {"next_edit": {"fixture_ids": list(sweep.EXPECTED_CASE_IDS)}},
        "observability": {
            "required_worker_mode": "offline",
            "required_max_payload_bytes": 8 * 2**20,
            "required_offline_bundle_max_bytes": 128 * 2**20,
            "sdk_versions": versions,
        },
    }
    runtime_config = SimpleNamespace(
        enabled=True,
        mode="offline",
        capture_content=True,
        artifact_max_payload_bytes=8 * 2**20,
        offline_max_bytes=128 * 2**20,
    )
    args = SimpleNamespace(
        mode="next-edit",
        output=tmp_path / "out",
        artifact_dir=artifact_dir,
        runtime=runtime.parents[2],
        next_edit_fixtures=fixtures,
    )
    digest_counts: dict[Path, int] = {}
    request_ids: list[str | None] = []
    scope_paths: list[Path] = []

    @contextmanager
    def fake_run_scope(path: Path, phase: str):
        scope_paths.append(path)
        yield RunContext.new(campaign_id="campaign-test", run_id=f"run-{phase}")

    class FakeServer:
        def __init__(self, _args, _model):
            self.argv = ["fake-server"]
            self.started_at = 0.0
            self.loaded_at = 1.0
            self.host_before = {}
            self.host_after = {}
            self.gpu_before = {}
            self.gpu_after_load = {}
            self.gpu_after_requests = {}
            self.backend_evidence = {"backend": "test"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def tokenize(self, _prompt: str) -> int:
            return 8

        def sample(self) -> None:
            return None

        def memory_report(self):
            return {"samples": 0}

    real_digest = sweep.digest_file

    def fixed_model_digest(path: Path) -> str:
        if path == q8:
            digest = sweep.MODEL_SHA256
        elif path == q4:
            digest = sweep.Q4_SHA256
        else:
            digest = real_digest(path)
        if path in {q8, q4}:
            digest_counts[path] = digest_counts.get(path, 0) + 1
        return digest

    def fake_generate(_server, _prompt, input_tokens, *, context, case_id, precision):
        request_ids.append(context.request_id)
        return {
            "input_tokens": input_tokens,
            "context_eligible": True,
            "explicit_terminal": False,
            "hit_token_cap": False,
            "observability_ids": {"request_id": context.request_id, "case_id": case_id},
            "precision": precision,
        }

    monkeypatch.setattr(sweep, "load_and_verify_plan", lambda _args: plan)
    monkeypatch.setattr(
        bootstrap, "current_runtime", lambda: SimpleNamespace(config=runtime_config)
    )
    monkeypatch.setattr(runs_module, "run_scope", fake_run_scope)
    monkeypatch.setattr(sweep, "Server", FakeServer)
    monkeypatch.setattr(sweep, "_generate_sweep_case", fake_generate)
    monkeypatch.setattr(sweep, "digest_file", fixed_model_digest)
    monkeypatch.setattr(sweep, "_host_info", lambda: {"hostname": "synthetic-test"})

    sweep.run(args)
    assert len(request_ids) == 96
    assert len(set(request_ids)) == 96
    assert digest_counts == {q8: 1, q4: 1}
    assert set(scope_paths) == {
        args.output / "next-edit/q8_0/observability-run.json",
        args.output / "next-edit/q4_k_m/observability-run.json",
    }
    assert capsys.readouterr().out.count('"state": "complete"') == 2


def test_worker_observability_gate_and_strict_file_identity(tmp_path: Path) -> None:
    config = SimpleNamespace(
        enabled=True,
        mode="offline",
        capture_content=True,
        artifact_max_payload_bytes=8 * 2**20,
        offline_max_bytes=128 * 2**20,
    )
    plan_observability = {
        "required_worker_mode": "offline",
        "required_max_payload_bytes": 8 * 2**20,
        "required_offline_bundle_max_bytes": 128 * 2**20,
        "sdk_versions": {"opentelemetry-api": "1.44.0", "opentelemetry-sdk": "1.44.0"},
    }
    assert (
        sweep.validate_observability_runtime(
            config, plan_observability["sdk_versions"], plan_observability
        )["capture_content"]
        is True
    )
    config.mode = "live"
    with pytest.raises(RuntimeError, match="observability/capture"):
        sweep.validate_observability_runtime(
            config, plan_observability["sdk_versions"], plan_observability
        )

    fixture = tmp_path / "strict.jsonl"
    fixture.write_text("public source fixture\n", encoding="utf-8")
    digest = sweep.digest_file(fixture)
    sweep.verify_file_identity(fixture, digest, "strict causal suite")
    with pytest.raises(ValueError, match="strict causal suite identity"):
        sweep.verify_file_identity(fixture, "0" * 64, "strict causal suite")
