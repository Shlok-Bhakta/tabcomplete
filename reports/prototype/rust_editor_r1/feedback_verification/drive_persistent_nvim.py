#!/usr/bin/env python3
"""Drive the installed LazyVim prediction plugin through a persistent Nvim RPC.

This controller is intentionally separate from the frozen producer code and
the existing headless fixture. It starts the real user-configured Neovim with
``--listen`` on a private Unix socket, then sends actual key input through
``--remote-send``. Every prediction event is marked synthetic after plugin
startup. The test source is derived from the checked-in public MIT smoke
fixture and copied into a disposable Git repository. Its target line starts
without its trailing blank; the script types that blank through Neovim before
inference.

Do not run until the Rust engine and LazyVim deployment are ready. The required
``--deployment-ready`` switch is an extra guard against an accidental run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "reports/prototype/rust_editor_r1/smoke_fixture.json"
DEFAULT_NVIM = "/etc/profiles/per-user/shlok/bin/nvim"
DEFAULT_PLUGIN_SPEC = "/home/shlok/.config/lazyvim/lua/plugins/tabcomplete-trajectory.lua"
LUA_MODULE_FUNCTIONS = {
    "tabcomplete_trajectory.buffers": ("attach", "tabcomplete_trajectory/buffers.lua"),
    "tabcomplete_trajectory": ("emit", "tabcomplete_trajectory/init.lua"),
    "tabcomplete_trajectory.predict": ("setup", "tabcomplete_trajectory/predict.lua"),
    "tabcomplete_trajectory.sse": ("finish_rust", "tabcomplete_trajectory/sse.lua"),
}
PREDICTION_EVENT_TYPES = {
    "prediction_requested",
    "prediction_generated",
    "prediction_shown",
    "prediction_dismissed",
    "prediction_accepted",
    "prediction_partially_accepted",
    "prediction_rejected",
    "prediction_reviewed",
}
EXPECTED_CONTEXT = {
    "q25": ("cursor-last-v1", "single-line-cursor-last-context-v1"),
    "sweep": ("sweep-window-v1", "sweep-window-context-v1"),
}


class CheckFailure(RuntimeError):
    """A content-free controller failure suitable for a report."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise CheckFailure(code)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def installed_plugin_source(spec_path: pathlib.Path) -> dict[str, Any]:
    try:
        spec_bytes = spec_path.read_bytes()
        spec_text = spec_bytes.decode("utf-8")
    except (OSError, UnicodeError):
        raise CheckFailure("installed_plugin_spec_unavailable") from None
    match = re.search(r'(?m)^\s*local\s+plugin\s*=\s*"([^"]+)"\s*$', spec_text)
    if match is None:
        raise CheckFailure("installed_plugin_spec_source_path_missing")
    plugin_dir = pathlib.Path(match.group(1))
    lua_dir = plugin_dir / "lua"
    try:
        spec_resolved = spec_path.resolve(strict=True)
        plugin_resolved = plugin_dir.resolve(strict=True)
        lua_resolved = lua_dir.resolve(strict=True)
    except OSError:
        raise CheckFailure("installed_plugin_source_root_unavailable") from None
    expected_files = {}
    for module, (_function_name, relative_path) in LUA_MODULE_FUNCTIONS.items():
        expected_path = lua_resolved / relative_path
        try:
            expected_path = expected_path.resolve(strict=True)
            expected_hash = sha256(expected_path.read_bytes())
        except OSError:
            raise CheckFailure("installed_plugin_module_source_unavailable") from None
        expected_files[module] = {
            "path": str(expected_path),
            "sha256": expected_hash,
        }
    return {
        "spec_path": str(spec_resolved),
        "spec_sha256": sha256(spec_bytes),
        "plugin_dir": str(plugin_resolved),
        "lua_dir": str(lua_resolved),
        "expected_modules": expected_files,
    }


def vim_string(value: str) -> str:
    """Return a Vim single-quoted string literal."""
    return "'" + value.replace("'", "''") + "'"


def decode_remote_value(raw: str) -> Any:
    """Decode raw JSON or a JSON string returned by --remote-expr."""
    value: Any = raw.strip()
    for _ in range(3):
        if not isinstance(value, str):
            return value
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


class Nvim:
    def __init__(self, binary: pathlib.Path, socket_path: pathlib.Path):
        self.binary = binary
        self.socket_path = socket_path
        self.process: subprocess.Popen[bytes] | None = None

    def start(self, file_path: pathlib.Path, state_home: pathlib.Path | None) -> None:
        env = os.environ.copy()
        if state_home is not None:
            env["XDG_STATE_HOME"] = str(state_home)
        # Preserve the installed init and plugins; these flags only keep this
        # disposable headless run from creating swap or ShaDa files.
        self.process = subprocess.Popen(
            [
                str(self.binary),
                "--headless",
                "--listen",
                str(self.socket_path),
                "-i",
                "NONE",
                str(file_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise CheckFailure("nvim_exited_during_startup")
            if self.socket_path.exists():
                try:
                    self.remote_expr("v:version")
                    return
                except CheckFailure:
                    pass
            time.sleep(0.1)
        raise CheckFailure("nvim_rpc_startup_timeout")

    def _client(self, *args: str, timeout: float = 10) -> subprocess.CompletedProcess[bytes]:
        if self.process is None or self.process.poll() is not None:
            raise CheckFailure("nvim_process_not_running")
        try:
            return subprocess.run(
                [str(self.binary), "--server", str(self.socket_path), *args],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise CheckFailure("nvim_rpc_timeout") from None

    def remote_expr(self, expression: str) -> Any:
        result = self._client("--remote-expr", expression)
        if result.returncode != 0:
            raise CheckFailure(f"nvim_remote_expr_exit_{result.returncode}")
        try:
            return decode_remote_value(result.stdout.decode("utf-8"))
        except UnicodeError:
            raise CheckFailure("nvim_remote_expr_invalid_utf8") from None

    def lua(self, expression: str, args: list[str | int] | None = None) -> Any:
        call = f"luaeval({vim_string(expression)}"
        if args is not None:
            rendered = []
            for item in args:
                rendered.append(vim_string(item) if isinstance(item, str) else str(item))
            call += ", [" + ", ".join(rendered) + "]"
        call += ")"
        return self.remote_expr(call)

    def send(self, keys: str) -> None:
        result = self._client("--remote-send", keys)
        if result.returncode != 0:
            raise CheckFailure(f"nvim_remote_send_exit_{result.returncode}")

    def close(self) -> bool:
        if self.process is None or self.process.poll() is not None:
            return True
        try:
            # Graceful :qa! invokes the collector's VimLeavePre lifecycle hook.
            self.send("<Esc>:qa!<CR>")
            self.process.wait(timeout=20)
        except (CheckFailure, subprocess.TimeoutExpired):
            return self.process.poll() is not None
        self.socket_path.unlink(missing_ok=True)
        return True


def inspect_loaded_lua_sources(
    nvim: Nvim, installed_plugin: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    spec_items = ",".join(
        f"{{module={vim_string(module)},field={vim_string(function_name)}}}"
        for module, (function_name, _relative_path) in LUA_MODULE_FUNCTIONS.items()
    )
    sources = nvim.lua(
        "vim.json.encode((function() local out={}; "
        "local specs={" + spec_items + "}; for _,spec in ipairs(specs) do "
        "local ok,mod=pcall(require,spec.module); "
        "if not ok then out[spec.module]={error='module_require_failed'} else "
        "local fn=mod[spec.field]; if type(fn)~='function' then "
        "out[spec.module]={error='module_entrypoint_missing'} else "
        "local info=debug.getinfo(fn,'S'); out[spec.module]={source=info.source} "
        "end end end; return out end)())"
    )
    require(isinstance(sources, dict), "loaded_plugin_module_sources_invalid")
    provenance = []
    failures = []
    expected_modules = installed_plugin["expected_modules"]
    for module, (function_name, _relative_path) in LUA_MODULE_FUNCTIONS.items():
        loaded = sources.get(module)
        expected = expected_modules[module]
        if not isinstance(loaded, dict):
            provenance.append(
                {
                    "module": module,
                    "entrypoint": function_name,
                    "source_error": "loaded_plugin_module_source_missing",
                    "expected_path": expected["path"],
                    "expected_sha256": expected["sha256"],
                    "matched": False,
                }
            )
            failures.append("loaded_plugin_module_source_missing")
            continue
        source = loaded.get("source")
        if not isinstance(source, str):
            provenance.append(
                {
                    "module": module,
                    "entrypoint": function_name,
                    "source_error": loaded.get("error")
                    or "loaded_plugin_module_debug_source_missing",
                    "expected_path": expected["path"],
                    "expected_sha256": expected["sha256"],
                    "matched": False,
                }
            )
            failures.append("loaded_plugin_module_debug_source_missing")
            continue
        source_path = source[1:] if source.startswith("@") else source
        try:
            loaded_path = pathlib.Path(source_path).resolve(strict=True)
            loaded_hash = sha256(loaded_path.read_bytes())
            expected_path = pathlib.Path(expected["path"]).resolve(strict=True)
        except (OSError, KeyError, TypeError):
            provenance.append(
                {
                    "module": module,
                    "entrypoint": function_name,
                    "source_path": source_path,
                    "source_sha256": None,
                    "expected_path": expected["path"],
                    "expected_sha256": expected["sha256"],
                    "matched": False,
                }
            )
            failures.append("loaded_plugin_module_file_unavailable")
            continue
        path_matches = loaded_path == expected_path
        hash_matches = loaded_hash == expected["sha256"]
        if not path_matches:
            failures.append("loaded_plugin_module_source_path_mismatch")
        if not hash_matches:
            failures.append("loaded_plugin_module_source_hash_mismatch")
        provenance.append(
            {
                "module": module,
                "entrypoint": function_name,
                "source_path": str(loaded_path),
                "source_sha256": loaded_hash,
                "expected_path": str(expected_path),
                "expected_sha256": expected["sha256"],
                "matched": path_matches and hash_matches,
            }
        )
    return provenance, failures


class Controller:
    def __init__(
        self,
        nvim: Nvim,
        repo: pathlib.Path,
        fixture_path: pathlib.Path,
        target_row: int,
        timeout: float,
        quiet_window: float,
        installed_plugin: dict[str, Any],
    ):
        self.nvim = nvim
        self.repo = repo
        self.fixture_path = fixture_path
        self.target_row = target_row
        self.timeout = timeout
        self.quiet_window = quiet_window
        self.installed_plugin = installed_plugin
        self.module_sources_by_session: list[dict[str, Any]] = []
        self.module_source_failures_by_session: list[dict[str, Any]] = []
        self.session_ids: list[str] = []
        self.scenarios: list[dict[str, Any]] = []
        self.all_events: list[dict[str, Any]] = []
        self.closed_session_events: list[list[dict[str, Any]]] = []

    def setup_after_load(self) -> dict[str, Any]:
        self.nvim.lua(
            '(function() require("tabcomplete_trajectory.predict").setup({synthetic=true}); '
            "vim.o.swapfile=false; return true end)()"
        )
        snap = self.snapshot()
        module_sources, module_failures = inspect_loaded_lua_sources(
            self.nvim, self.installed_plugin
        )
        self.module_sources_by_session.append(
            {"session_id": snap["session_id"], "modules": module_sources}
        )
        self.module_source_failures_by_session.append(
            {"session_id": snap["session_id"], "failures": module_failures}
        )
        require(not module_failures, module_failures[0] if module_failures else "")

        # Inspect the module's original emit function first. The observer below
        # is created by luaeval(), so its debug source would be ``luaeval()``.
        self.nvim.lua(
            '(function() local c=require("tabcomplete_trajectory"); '
            "if not _G.__tc_feedback_recorded_emit then "
            "local old=c.emit; _G.__tc_feedback_recorded_emit=true; "
            "_G.__tc_feedback_events={}; c.emit=function(...) "
            "local event=old(...); if event then "
            "table.insert(_G.__tc_feedback_events, vim.deepcopy(event)) end; "
            "return event end end; return true end)()"
        )
        if snap["session_id"] and snap["session_id"] not in self.session_ids:
            self.session_ids.append(snap["session_id"])
        require(snap["status"]["mode"] == "automatic", "automatic_mode_not_enabled")
        require(snap["status"]["experimental_auto_opt_in"] is True, "automatic_opt_in_missing")
        require(
            snap["status"]["automatic_personalization_enabled"] is False,
            "automatic_personalization_not_disabled",
        )
        require(snap["status"]["acceptance_key"] == "<M-l>", "acceptance_key_not_alt_l")
        require(snap["accept_map"]["exists"], "alt_l_insert_mapping_missing")
        require(snap["accept_map"]["callback"], "alt_l_insert_mapping_has_no_callback")
        require(snap["status"]["backend"] == "rust-editor-v1", "rust_backend_not_installed")
        aliases = snap["aliases"]
        require("q25" in aliases and "sweep" in aliases, "both_model_aliases_not_available")
        return snap

    def snapshot(self) -> dict[str, Any]:
        value = self.nvim.lua(
            '(function() local p=require("tabcomplete_trajectory.predict"); '
            'local s=p.status(); local c=require("tabcomplete_trajectory"); '
            "local b=vim.api.nvim_get_current_buf(); "
            "local lines=vim.api.nvim_buf_get_lines(b,0,-1,false); "
            'local m=vim.fn.maparg("<M-l>","i",false,true); '
            "local aliases=p.model_aliases(); table.sort(aliases); "
            'local state_path=vim.fn.stdpath("state")'
            '.."/tabcomplete-rust-editor-mode.json"; '
            'local f=io.open(state_path,"rb"); local saved=nil; '
            'if f then local raw=f:read("*a"); f:close(); '
            "local ok,v=pcall(vim.json.decode,raw); "
            'if ok and type(v)=="table" then '
            "saved={mode=v.mode,experimental_auto_opt_in=v.experimental_auto_opt_in} end end; "
            "return vim.json.encode({status={mode=s.mode,backend=s.backend, "
            "protocol=s.protocol_version, "
            "model_alias=s.model_alias or vim.NIL, "
            "model_protocol=s.model_protocol or vim.NIL, "
            "context_layout=s.context_layout or vim.NIL, "
            "context_policy_version=s.context_policy_version or vim.NIL, "
            "acceptance_key=s.acceptance_key, "
            "experimental_auto_opt_in=s.experimental_auto_opt_in, "
            "automatic_quality_validated=s.automatic_quality_validated, "
            "automatic_personalization_enabled=s.automatic_personalization_enabled, "
            "in_flight=s.in_flight,proposal_active=s.proposal_active, "
            "state=s.state,counters=s.counters}, "
            "mode=vim.api.nvim_get_mode().mode, "
            "file=vim.api.nvim_buf_get_name(b),mode_state_path=state_path, "
            "line=lines[" + str(self.target_row + 1) + '] or "", '
            'buffer_hash=vim.fn.sha256(table.concat(lines,"\\n")), '
            'accept_map={exists=vim.fn.maparg("<M-l>","i")~="",'
            'callback=type(m.callback)=="function"}, '
            "aliases=aliases,session_id=c.session_id,saved_mode=saved}) end)()"
        )
        if not isinstance(value, dict):
            raise CheckFailure("nvim_snapshot_invalid")
        return value

    def recorded_events(self) -> list[dict[str, Any]]:
        events = self.nvim.lua("vim.json.encode(_G.__tc_feedback_events or {})")
        if not isinstance(events, list):
            raise CheckFailure("nvim_event_capture_invalid")
        self.all_events = events
        return events

    def wait_for(
        self, predicate: Callable[[dict[str, Any]], bool], label: str, timeout: float | None = None
    ) -> dict[str, Any]:
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while time.monotonic() < deadline:
            snap = self.snapshot()
            if predicate(snap):
                return snap
            if self.nvim.process is None or self.nvim.process.poll() is not None:
                raise CheckFailure(f"nvim_exited_while_waiting_{label}")
            time.sleep(0.08)
        snap = self.snapshot()
        raise CheckFailure(f"timeout_{label}_state_{snap['status']['state']}")

    def wait_for_show(self, base: dict[str, Any], name: str) -> dict[str, Any]:
        req0 = base["status"]["counters"]["requested"]
        shown0 = base["status"]["counters"]["displayed"]
        self.wait_for(
            lambda s: s["status"]["counters"]["requested"] >= req0 + 1,
            f"{name}_request",
        )
        self.wait_for(
            lambda s: not s["status"]["in_flight"] or s["status"]["proposal_active"],
            f"{name}_completion",
        )
        snap = self.snapshot()
        require(
            snap["status"]["counters"]["requested"] == req0 + 1, f"{name}_request_count_not_one"
        )
        require(
            snap["status"]["counters"]["displayed"] == shown0 + 1, f"{name}_display_count_not_one"
        )
        require(snap["status"]["proposal_active"], f"{name}_proposal_not_active")
        # Let any accidental repeat debounce run before accepting the first UI.
        time.sleep(self.quiet_window)
        quiet = self.snapshot()
        require(
            quiet["status"]["counters"]["requested"] == req0 + 1,
            f"{name}_repeat_request_after_pause",
        )
        require(
            quiet["status"]["counters"]["displayed"] == shown0 + 1,
            f"{name}_repeat_display_after_pause",
        )
        require(quiet["mode"].startswith("i"), f"{name}_not_in_real_insert_mode")
        return quiet

    def latest_event(self, event_type: str, after: int = 0) -> dict[str, Any] | None:
        events = self.recorded_events()
        for event in reversed(events[after:]):
            if event.get("event_type") == event_type:
                return event
        return None

    def event_for_prediction(
        self, event_type: str, prediction_id: str, after: int = 0
    ) -> dict[str, Any] | None:
        for event in reversed(self.recorded_events()[after:]):
            payload = event.get("payload") or {}
            if (
                event.get("event_type") == event_type
                and payload.get("prediction_id") == prediction_id
            ):
                return event
        return None

    def assert_context_policy(self, snapshot: dict[str, Any], alias: str) -> tuple[str, str]:
        expected = EXPECTED_CONTEXT.get(alias)
        require(expected is not None, "active_model_context_layout_unknown")
        layout, policy = expected
        require(
            snapshot["status"].get("context_layout") == layout, "rust_model_context_layout_mismatch"
        )
        require(
            snapshot["status"].get("context_policy_version") == policy,
            "rust_model_context_policy_mismatch",
        )
        return expected

    def assert_prediction_context(
        self, prediction_id: str, alias: str, after: int = 0
    ) -> tuple[str, str]:
        layout, policy = EXPECTED_CONTEXT[alias]
        for event_type in ("prediction_requested", "prediction_generated"):
            event = self.event_for_prediction(event_type, prediction_id, after)
            require(event is not None, f"{event_type}_context_event_missing")
            payload = event["payload"]
            require(
                payload.get("context_layout") == layout, f"{event_type}_context_layout_mismatch"
            )
            require(
                payload.get("context_policy_version") == policy,
                f"{event_type}_context_policy_mismatch",
            )
        return layout, policy

    def assert_prediction_events_synthetic(self) -> None:
        for event in self.recorded_events():
            if event.get("event_type") in PREDICTION_EVENT_TYPES:
                payload = event.get("payload") or {}
                require(payload.get("synthetic") is True, "prediction_event_missing_synthetic_true")

    def set_cursor(self) -> None:
        self.nvim.lua(
            "(function() vim.api.nvim_win_set_cursor(0,{_A[1],_A[2]}); return true end)()",
            # Normal-mode cursors clamp to the final byte of a line. Place it
            # safely, then use `A` to enter Insert mode at the true line end.
            [self.target_row + 1, 0],
        )

    def reload_fixture(self) -> None:
        self.nvim.send("<Esc>:edit!<CR>")
        self.wait_for(lambda s: s["file"] == str(self.fixture_path), "reload_fixture", 10)
        snap = self.snapshot()
        require(snap["line"] == "    return", "fixture_line_changed_on_disk")
        self.set_cursor()

    def begin_typed_request(self, name: str) -> tuple[dict[str, Any], int]:
        self.reload_fixture()
        base = self.snapshot()
        event_offset = len(self.recorded_events())
        self.nvim.send("A ")
        typed = self.snapshot()
        require(typed["mode"].startswith("i"), f"{name}_typing_did_not_enter_insert_mode")
        require(typed["line"] == "    return ", f"{name}_remote_send_did_not_type_expected_prefix")
        return base, event_offset

    def run_scenario(self, name: str, operation: Callable[[], dict[str, Any] | None]) -> None:
        began = time.monotonic()
        try:
            detail = operation() or {}
            self.scenarios.append(
                {
                    "name": name,
                    "status": "passed",
                    "elapsed_ms": round((time.monotonic() - began) * 1000, 1),
                    **detail,
                }
            )
        except CheckFailure as exc:
            self.scenarios.append(
                {
                    "name": name,
                    "status": "failed",
                    "code": str(exc),
                    "elapsed_ms": round((time.monotonic() - began) * 1000, 1),
                }
            )
        except Exception as exc:  # sanitize unexpected exceptions
            self.scenarios.append(
                {
                    "name": name,
                    "status": "failed",
                    "code": "unexpected_" + type(exc).__name__,
                    "elapsed_ms": round((time.monotonic() - began) * 1000, 1),
                }
            )

    def type_pause_accept_undo(self) -> dict[str, Any]:
        base, offset = self.begin_typed_request("accept_undo")
        shown = self.wait_for_show(base, "accept_undo")
        alias = shown["status"].get("model_alias")
        layout, policy = self.assert_context_policy(shown, alias)
        before_accept = self.snapshot()
        shown_event = self.latest_event("prediction_shown", offset)
        require(shown_event is not None, "accepted_proposal_show_event_missing")
        prediction_id = shown_event["payload"].get("prediction_id")
        event_layout, event_policy = self.assert_prediction_context(prediction_id, alias, offset)
        require(
            (event_layout, event_policy) == (layout, policy),
            "status_and_prediction_context_metadata_disagree",
        )
        self.nvim.send("<M-l>")
        accepted = self.wait_for(
            lambda s: (
                s["status"]["counters"]["accepted"] == base["status"]["counters"]["accepted"] + 1
            ),
            "installed_alt_l_acceptance",
        )
        require(not accepted["status"]["proposal_active"], "alt_l_left_proposal_active")
        accepted_line = accepted["line"]
        require(accepted_line != before_accept["line"], "alt_l_did_not_change_buffer")
        self.nvim.send("<Esc>u")
        undone = self.wait_for(
            lambda s: not s["status"]["proposal_active"], "undo_after_accept", 10
        )
        require(
            undone["buffer_hash"] == before_accept["buffer_hash"],
            "undo_did_not_restore_pre_accept_buffer",
        )
        require(undone["line"] == "    return ", "undo_removed_earlier_typed_prefix")
        events = self.recorded_events()
        accept_events = [
            e
            for e in events
            if e.get("event_type") == "prediction_accepted"
            and (e.get("payload") or {}).get("prediction_id") == prediction_id
        ]
        require(len(accept_events) == 1, "acceptance_not_recorded_once")
        self.assert_prediction_events_synthetic()
        return {
            "session_id": shown_event["session_id"],
            "prediction_id": prediction_id,
            "requested_delta": 1,
            "displayed_delta": 1,
            "accepted_delta": 1,
            "model_alias": alias,
            "context_layout": layout,
            "context_policy_version": policy,
            "undo_preserved_typed_prefix": True,
        }

    def divergent_typing(self) -> dict[str, Any]:
        self.set_mode("automatic")
        base, offset = self.begin_typed_request("divergent_typing")
        self.wait_for_show(base, "divergent_typing")
        before = self.snapshot()
        rejected0 = before["status"]["counters"]["rejected_implicit_typing"]
        self.nvim.send("x")
        dismissed = self.wait_for(
            lambda s: s["status"]["counters"]["rejected_implicit_typing"] > rejected0,
            "divergent_typing_classification",
            5,
        )
        require(not dismissed["status"]["proposal_active"], "divergent_input_kept_proposal")
        self.set_mode("off")
        self.assert_prediction_events_synthetic()
        event = self.latest_dismissal(offset, "rejected_implicit_typing")
        require(event is not None, "divergent_dismissal_event_missing")
        return {
            "prediction_id": event["payload"].get("prediction_id"),
            "outcome": "rejected_implicit_typing",
        }

    def latest_dismissal(self, after: int, outcome: str) -> dict[str, Any] | None:
        for event in reversed(self.recorded_events()[after:]):
            if (
                event.get("event_type") == "prediction_dismissed"
                and (event.get("payload") or {}).get("outcome") == outcome
            ):
                return event
        return None

    def typed_matching(self) -> dict[str, Any]:
        self.set_mode("automatic")
        base, offset = self.begin_typed_request("typed_matching")
        self.wait_for_show(base, "typed_matching")
        shown = self.latest_event("prediction_shown", offset)
        require(shown is not None, "typed_match_show_event_missing")
        proposed = shown["payload"].get("proposed_text")
        current = self.snapshot()["line"]
        require(
            isinstance(proposed, str) and proposed.startswith(current),
            "typed_match_proposal_does_not_preserve_current_prefix",
        )
        suffix = proposed[len(current) :]
        require(
            suffix != "" and "\n" not in suffix and "\r" not in suffix,
            "typed_match_suffix_not_single_line",
        )
        # Set a private editor register through RPC, then insert it through
        # actual Ctrl-R input. Neovim reports the register text as successive
        # raw deltas, so the first character dismisses as a partial match while
        # the remaining input can still complete the offered line.
        self.nvim.lua('(function() vim.fn.setreg("z",_A[1]); return true end)()', [suffix])
        partial0 = self.snapshot()["status"]["counters"]["typed_partial_match"]
        self.nvim.send("<C-r>z")
        self.set_mode("off")
        partial = self.wait_for(
            lambda s: (
                s["status"]["counters"]["typed_partial_match"] > partial0
                and not s["status"]["proposal_active"]
            ),
            "typed_partial_match_classification",
            8,
        )
        require(partial["line"] == proposed, "typed_input_did_not_reach_full_proposal")
        event = self.latest_dismissal(offset, "typed_partial_match")
        require(event is not None, "typed_partial_match_dismissal_event_missing")
        deltas = [e for e in self.recorded_events()[offset:] if e.get("event_type") == "edit_delta"]
        previous_line = current
        matching_deltas = []
        for delta in deltas:
            payload = delta.get("payload") or {}
            deleted = payload.get("deleted_text")
            inserted = payload.get("inserted_text")
            if deleted != previous_line or not isinstance(inserted, str):
                continue
            if not inserted.startswith(previous_line) or not proposed.startswith(inserted):
                continue
            matching_deltas.append(delta)
            previous_line = inserted
        require(len(matching_deltas) >= 2, "typed_match_raw_delta_chain_too_short")
        require(
            matching_deltas[0]["payload"]["inserted_text"] != proposed,
            "typed_match_first_delta_was_not_a_partial_prefix",
        )
        require(previous_line == proposed, "typed_match_raw_delta_chain_did_not_reach_proposal")
        self.assert_prediction_events_synthetic()
        return {
            "prediction_id": event["payload"].get("prediction_id"),
            "plugin_first_delta_outcome": "typed_partial_match",
            "derived_full_input_relation": "typed_match",
            "raw_delta_chain_count": len(matching_deltas),
            "raw_delta_first_sequence": matching_deltas[0].get("sequence_number"),
            "raw_delta_final_sequence": matching_deltas[-1].get("sequence_number"),
        }

    def set_mode(self, mode: str) -> None:
        if mode == "off":
            self.nvim.lua(
                '(function() require("tabcomplete_trajectory.predict").set_mode("off"); '
                "return true end)()"
            )
        else:
            self.nvim.send(f"<Esc>:TabCompleteMode {mode}<CR>")
        self.wait_for(lambda s: s["status"]["mode"] == mode, f"mode_{mode}", 8)

    def switch_model(self, alias: str) -> None:
        self.nvim.send(f"<Esc>:TabCompleteModel {alias}<CR>")
        switched = self.wait_for(
            lambda s: (
                s["status"]["model_alias"] == alias
                and s["status"]["mode"] == "automatic"
                and s["status"]["state"] == "model switched to " + alias
            ),
            f"model_switch_{alias}",
        )
        self.assert_context_policy(switched, alias)

    def model_picker_one_at_a_time(self) -> dict[str, Any]:
        self.set_mode("automatic")
        snap = self.snapshot()
        aliases = snap["aliases"]
        choices = self.nvim.lua(
            'vim.json.encode(vim.fn.getcompletion("TabCompleteModel ","cmdline"))'
        )
        require(isinstance(choices, list), "model_command_completion_not_available")
        require("q25" in choices and "sweep" in choices, "model_completion_missing_alias")
        current = snap["status"]["model_alias"]
        require(current in {"q25", "sweep"}, "active_model_alias_unknown")
        first = "sweep" if current == "q25" else "q25"
        second = current
        # These are two real user commands in one remote input batch. The first
        # starts an async model load; set_model must reject the second while the
        # first owns the model-switch operation.
        self.nvim.send(f"<Esc>:TabCompleteModel {first}<CR>:TabCompleteModel {second}<CR>")
        switched = self.wait_for(
            lambda s: (
                s["status"]["model_alias"] == first
                and s["status"]["mode"] == "automatic"
                and s["status"]["state"] == "model switched to " + first
            ),
            "model_switch_single_owner",
            180,
        )
        layout, policy = self.assert_context_policy(switched, first)
        require(
            switched["status"]["model_alias"] != second, "overlapping_model_switch_was_not_rejected"
        )
        time.sleep(2.0)
        settled = self.snapshot()
        require(
            settled["status"]["model_alias"] == first
            and settled["status"]["state"] == "model switched to " + first,
            "overlapping_model_switch_changed_final_alias",
        )
        self.assert_context_policy(settled, first)
        if first != "q25":
            self.switch_model("q25")
        self.assert_prediction_events_synthetic()
        return {
            "completion_aliases": [name for name in aliases if name in choices],
            "first_alias": first,
            "overlapping_alias_rejected": second,
            "final_alias": self.snapshot()["status"]["model_alias"],
            "context_layout": layout,
            "context_policy_version": policy,
        }

    def stale_response_on_file_switch(self) -> dict[str, Any]:
        self.set_mode("automatic")
        base, offset = self.begin_typed_request("stale_file_switch")
        req0 = base["status"]["counters"]["requested"]
        started = self.wait_for(
            lambda s: s["status"]["counters"]["requested"] >= req0 + 1 and s["status"]["in_flight"],
            "stale_request_in_flight",
            self.timeout,
        )
        require(not started["status"]["proposal_active"], "stale_case_already_displayed")
        # Change file while native generation is in flight. Use an RPC command,
        # not an API buffer mutation, so the normal BufLeave path invalidates it.
        self.nvim.lua(
            '(function() vim.api.nvim_command("edit! " .. '
            "vim.fn.fnameescape(_A[1])); return true end)()",
            [str(self.repo / "other.py")],
        )
        switched = self.wait_for(
            lambda s: s["file"] == str(self.repo / "other.py"), "stale_case_file_switch", 10
        )
        require(not switched["status"]["proposal_active"], "file_switch_retained_stale_proposal")
        self.wait_for(lambda s: not s["status"]["in_flight"], "stale_request_closed", 45)
        time.sleep(1.0)
        final = self.snapshot()
        require(not final["status"]["proposal_active"], "late_stale_response_was_displayed")
        events = self.recorded_events()[offset:]
        cancelled = [
            e
            for e in events
            if e.get("event_type") == "heartbeat"
            and (e.get("payload") or {}).get("prediction_lifecycle") == "cancelled_unseen"
        ]
        require(cancelled, "stale_request_did_not_record_unseen_cancellation")
        self.assert_prediction_events_synthetic()
        return {
            "cancelled_unseen": True,
            "late_display_count": 0,
            "prediction_id": cancelled[-1]["payload"].get("prediction_id"),
        }

    def off_cancels_pending(self) -> dict[str, Any]:
        self.nvim.lua(
            '(function() vim.api.nvim_command("edit! " .. '
            "vim.fn.fnameescape(_A[1])); return true end)()",
            [str(self.fixture_path)],
        )
        self.wait_for(lambda s: s["file"] == str(self.fixture_path), "off_case_fixture", 10)
        self.set_mode("automatic")
        base, offset = self.begin_typed_request("off_cancels_pending")
        req0 = base["status"]["counters"]["requested"]
        self.wait_for(
            lambda s: s["status"]["counters"]["requested"] >= req0 + 1 and s["status"]["in_flight"],
            "off_case_request_in_flight",
        )
        cancelled0 = self.snapshot()["status"]["counters"]["cancelled_unseen"]
        self.set_mode("off")
        final = self.wait_for(lambda s: not s["status"]["in_flight"], "off_case_request_closed", 45)
        time.sleep(1.0)
        final = self.snapshot()
        require(final["status"]["mode"] == "off", "off_mode_not_retained")
        require(not final["status"]["proposal_active"], "off_mode_left_proposal_visible")
        require(
            final["status"]["counters"]["cancelled_unseen"] > cancelled0,
            "off_mode_did_not_cancel_unseen_request",
        )
        self.assert_prediction_events_synthetic()
        return {"cancelled_unseen": True, "mode": "off"}

    def restart_persistent_automatic(self, state_home: pathlib.Path) -> dict[str, Any]:
        self.set_mode("automatic")
        saved = self.snapshot()["saved_mode"]
        require(
            saved == {"mode": "automatic", "experimental_auto_opt_in": True},
            "automatic_mode_not_persisted",
        )
        old_session = self.snapshot()["session_id"]
        self.closed_session_events.append(self.recorded_events())
        require(self.nvim.close(), "first_nvim_session_would_not_exit_cleanly")
        self.nvim = Nvim(self.nvim.binary, self.nvim.socket_path)
        self.nvim.start(self.fixture_path, state_home)
        second = self.setup_after_load()
        require(
            second["status"]["mode"] == "automatic",
            "automatic_mode_not_restored_after_restart",
        )
        require(second["session_id"] != old_session, "collector_session_id_reused_after_restart")
        base, offset = self.begin_typed_request("restart_persistent_automatic")
        shown = self.wait_for_show(base, "restart_persistent_automatic")
        alias = shown["status"].get("model_alias")
        layout, policy = self.assert_context_policy(shown, alias)
        shown_event = self.latest_event("prediction_shown", offset)
        require(shown_event is not None, "restart_auto_prediction_show_event_missing")
        prediction_id = shown_event["payload"].get("prediction_id")
        self.assert_prediction_context(prediction_id, alias, offset)
        self.set_mode("off")
        require(
            not self.snapshot()["status"]["proposal_active"], "restart_auto_prediction_not_cleared"
        )
        self.assert_prediction_events_synthetic()
        return {
            "old_session_id": old_session,
            "new_session_id": second["session_id"],
            "saved_mode": "automatic",
            "restored_mode": second["status"]["mode"],
            "automatic_request_after_restart": True,
            "automatic_display_after_restart": True,
            "requested_delta": shown["status"]["counters"]["requested"]
            - base["status"]["counters"]["requested"],
            "displayed_delta": shown["status"]["counters"]["displayed"]
            - base["status"]["counters"]["displayed"],
            "prediction_id": prediction_id,
            "context_layout": layout,
            "context_policy_version": policy,
        }

    def raw_delta_events(self) -> list[dict[str, Any]]:
        raw: list[dict[str, Any]] = []
        event_sets = list(self.closed_session_events)
        try:
            event_sets.append(self.recorded_events())
        except CheckFailure:
            if not event_sets:
                raise
        for event in (item for group in event_sets for item in group):
            if event.get("event_type") != "edit_delta":
                continue
            payload = event.get("payload") or {}
            path_value = payload.get("path")
            if isinstance(path_value, str):
                payload = dict(payload)
                payload["path"] = pathlib.Path(path_value).name
            raw.append(
                {
                    "event_id": event.get("event_id"),
                    "session_id": event.get("session_id"),
                    "sequence_number": event.get("sequence_number"),
                    "timestamp_ms": event.get("timestamp_ms"),
                    "event_type": "edit_delta",
                    "file_id": "repo:public-mit-disposable:"
                    + pathlib.Path(str(event.get("file_id", "")).rsplit(":", 1)[-1]).name,
                    "cursor": event.get("cursor"),
                    "mode": event.get("mode"),
                    "payload": payload,
                }
            )
        return raw


def verify_global_user_mode_persistence(
    binary: pathlib.Path,
    fixture_path: pathlib.Path,
    repo: pathlib.Path,
    target_row: int,
    timeout: float,
    quiet_window: float,
    installed_plugin: dict[str, Any],
    expected_mode_file: pathlib.Path,
) -> dict[str, Any]:
    socket_path = pathlib.Path("/tmp") / ("tc-fb-global-" + uuid.uuid4().hex[:12] + ".sock")
    first_nvim = Nvim(binary, socket_path)
    active_nvim = first_nvim
    controller: Controller | None = None
    try:
        first_nvim.start(fixture_path, None)
        controller = Controller(
            first_nvim,
            repo,
            fixture_path,
            target_row,
            timeout,
            quiet_window,
            installed_plugin,
        )
        first = controller.setup_after_load()
        state_file = pathlib.Path(first["mode_state_path"])
        require(state_file == expected_mode_file, "global_mode_file_path_mismatch")
        initial_saved_mode = first.get("saved_mode")

        # This is the real LazyVim state path. Setup marks any plugin decisions
        # synthetic, and the mode command itself does not start inference.
        controller.set_mode("automatic")
        persisted = controller.snapshot()
        require(
            persisted.get("saved_mode") == {"mode": "automatic", "experimental_auto_opt_in": True},
            "global_automatic_mode_not_written",
        )
        first_session_id = persisted.get("session_id")
        require(first_nvim.close(), "global_mode_first_nvim_exit_failed")

        second_nvim = Nvim(binary, socket_path)
        active_nvim = second_nvim
        second_nvim.start(fixture_path, None)
        second_controller = Controller(
            second_nvim,
            repo,
            fixture_path,
            target_row,
            timeout,
            quiet_window,
            installed_plugin,
        )
        second = second_controller.setup_after_load()
        require(
            pathlib.Path(second["mode_state_path"]) == expected_mode_file,
            "global_mode_restart_path_mismatch",
        )
        require(
            second["status"]["mode"] == "automatic",
            "global_automatic_mode_not_restored",
        )
        require(
            second.get("saved_mode") == {"mode": "automatic", "experimental_auto_opt_in": True},
            "global_automatic_mode_state_changed_after_restart",
        )
        require(
            second.get("session_id") != first_session_id,
            "global_mode_restart_reused_collector_session",
        )
        require(second_nvim.close(), "global_mode_second_nvim_exit_failed")
        return {
            "status": "passed",
            "synthetic": True,
            "mode_state_file": str(expected_mode_file),
            "initial_saved_mode": initial_saved_mode,
            "written_mode": persisted["saved_mode"],
            "restored_mode": second["status"]["mode"],
            "first_session_id": first_session_id,
            "second_session_id": second["session_id"],
            "module_sources_by_session": (
                controller.module_sources_by_session + second_controller.module_sources_by_session
            ),
            "module_source_failures_by_session": (
                controller.module_source_failures_by_session
                + second_controller.module_source_failures_by_session
            ),
        }
    finally:
        if active_nvim.process is not None and active_nvim.process.poll() is None:
            active_nvim.close()
        socket_path.unlink(missing_ok=True)


def create_disposable_public_repo(root: pathlib.Path) -> tuple[pathlib.Path, int, int, str]:
    try:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise CheckFailure("public_mit_fixture_unavailable") from None
    require(fixture.get("source_license") == "MIT", "fixture_license_not_mit")
    state = fixture.get("state")
    require(isinstance(state, dict), "fixture_state_missing")
    source = state.get("source")
    row = state.get("target_row")
    col = state.get("cursor_col")
    require(
        isinstance(source, str) and type(row) is int and type(col) is int, "fixture_source_invalid"
    )
    source_lines = source.splitlines(keepends=True)
    require(0 <= row < len(source_lines), "fixture_target_row_out_of_range")
    target = source_lines[row]
    ending = "\r\n" if target.endswith("\r\n") else "\n" if target.endswith("\n") else ""
    target_text = target[: -len(ending)] if ending else target
    require(target_text == "    return ", "fixture_target_line_unexpected")
    source_lines[row] = "    return" + ending
    derived_source = "".join(source_lines)
    path = root / "sample.py"
    path.write_bytes(derived_source.encode("utf-8"))
    (root / "other.py").write_text("def unrelated(value):\n    return value\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "TabComplete synthetic test",
            "GIT_AUTHOR_EMAIL": "synthetic@example.invalid",
            "GIT_COMMITTER_NAME": "TabComplete synthetic test",
            "GIT_COMMITTER_EMAIL": "synthetic@example.invalid",
        }
    )
    commands = [
        ["git", "init", "--quiet", str(root)],
        ["git", "-C", str(root), "add", "sample.py", "other.py"],
        ["git", "-C", str(root), "commit", "--quiet", "-m", "synthetic public fixture"],
    ]
    for command in commands:
        completed = subprocess.run(
            command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )
        require(completed.returncode == 0, "disposable_repo_setup_failed")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
    )
    require(commit.returncode == 0, "disposable_repo_commit_missing")
    return path, row, col, commit.stdout.strip()


def write_report(
    directory: pathlib.Path, report: dict[str, Any], deltas: list[dict[str, Any]]
) -> tuple[pathlib.Path, pathlib.Path]:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    result_path = directory / f"persistent_nvim_{timestamp}_{suffix}.json"
    delta_path = directory / f"persistent_nvim_{timestamp}_{suffix}_raw_deltas.jsonl"
    report["result_path"] = str(result_path)
    report["raw_delta_artifact"] = delta_path.name
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with delta_path.open("w", encoding="utf-8") as stream:
        for delta in deltas:
            stream.write(json.dumps(delta, sort_keys=True, ensure_ascii=False) + "\n")
    return result_path, delta_path


def run(args: argparse.Namespace) -> int:
    require(args.deployment_ready, "deployment_ready_flag_required")
    binary_argument = pathlib.Path(args.nvim).expanduser()
    require(
        binary_argument.is_file() and os.access(binary_argument, os.X_OK),
        "installed_nvim_binary_unavailable",
    )
    binary = binary_argument.resolve(strict=True)
    installed_plugin = installed_plugin_source(args.expected_plugin_spec)
    require(args.timeout >= 15 and args.timeout <= 300, "timeout_outside_15_to_300_seconds")
    require(0 <= args.quiet_window <= 10, "quiet_window_outside_0_to_10_seconds")

    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="tabcomplete-feedback-")).resolve()
    repo = temp_root / "public-mit-disposable"
    repo.mkdir()
    state_home = temp_root / "xdg-state"
    socket_path = pathlib.Path("/tmp") / ("tc-fb-" + uuid.uuid4().hex[:12] + ".sock")
    fixture_path, row, col, commit = create_disposable_public_repo(repo)
    nvim = Nvim(binary, socket_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
        "synthetic": True,
        "installed_nvim_binary": str(binary),
        "nvim_wrapper_path": str(binary_argument),
        "installed_plugin_spec": installed_plugin,
        "transport": "persistent headless Neovim over a private Unix socket",
        "typing_transport": "nvim --remote-send",
        "mode_override_used": False,
        "fixture": {
            "id": "public-return-prefix-smoke",
            "license": "MIT",
            "repository_kind": "disposable local Git copy",
            "commit": commit,
            "source_sha256": sha256(fixture_path.read_bytes()),
            "origin_fixture_sha256": sha256(FIXTURE.read_bytes()),
            "fixture_cursor_byte_column_before_typing": col,
            "starting_line_variant": (
                "removed fixture target trailing blank; actual remote-send A-space restores it"
            ),
        },
        "scenarios": [],
        "session_ids": [],
        "raw_delta_count": 0,
    }
    controller: Controller | None = None
    clean_exit = True
    deltas: list[dict[str, Any]] = []
    try:
        nvim.start(fixture_path, state_home)
        controller = Controller(
            nvim,
            repo,
            fixture_path,
            row,
            args.timeout,
            args.quiet_window,
            installed_plugin,
        )
        initial = controller.setup_after_load()
        require(initial["file"] == str(fixture_path), "nvim_did_not_open_public_fixture")
        require(initial["line"] == "    return", "nvim_loaded_unexpected_fixture_line")
        require(initial["session_id"], "collector_session_not_started")
        report["initial_model_alias"] = initial["status"].get("model_alias")
        report["backend"] = initial["status"]["backend"]
        report["model_protocol"] = initial["status"]["model_protocol"]
        report["module_sources_by_session"] = controller.module_sources_by_session
        report["module_source_failures_by_session"] = controller.module_source_failures_by_session

        controller.run_scenario(
            "type_pause_one_request_one_display", controller.type_pause_accept_undo
        )
        controller.run_scenario("divergent_typing_classification", controller.divergent_typing)
        controller.run_scenario("typed_matching_classification", controller.typed_matching)
        controller.run_scenario(
            "model_completion_and_one_at_a_time_switch", controller.model_picker_one_at_a_time
        )
        controller.run_scenario(
            "stale_response_after_file_switch", controller.stale_response_on_file_switch
        )
        controller.run_scenario("off_cancels_unseen_inference", controller.off_cancels_pending)

        controller.run_scenario(
            "restart_restores_persistent_automatic_mode",
            lambda: controller.restart_persistent_automatic(state_home),
        )
        global_mode_result: dict[str, Any] = {}

        def verify_global_mode() -> dict[str, Any]:
            global_mode_result.update(
                verify_global_user_mode_persistence(
                    binary,
                    fixture_path,
                    repo,
                    row,
                    args.timeout,
                    args.quiet_window,
                    installed_plugin,
                    args.expected_global_mode_file,
                )
            )
            return global_mode_result

        controller.run_scenario(
            "global_user_mode_persists_across_lazyvim_restart",
            verify_global_mode,
        )
        report["global_mode_persistence"] = global_mode_result
        report["session_ids"] = controller.session_ids
        report["scenarios"] = controller.scenarios
        report["module_sources_by_session"] = controller.module_sources_by_session
        report["module_source_failures_by_session"] = controller.module_source_failures_by_session
        deltas = controller.raw_delta_events()
        report["raw_delta_count"] = len(deltas)
        report["raw_delta_sha256"] = sha256(
            json.dumps(deltas, sort_keys=True, ensure_ascii=False).encode("utf-8")
        )
        report["nvim_exit_clean"] = controller.nvim.close()
        clean_exit = report["nvim_exit_clean"]
    except CheckFailure as exc:
        report["fatal_code"] = str(exc)
        if controller is not None:
            report["session_ids"] = controller.session_ids
            report["scenarios"] = controller.scenarios
            report["module_sources_by_session"] = controller.module_sources_by_session
            report["module_source_failures_by_session"] = (
                controller.module_source_failures_by_session
            )
            try:
                deltas = controller.raw_delta_events()
                report["raw_delta_count"] = len(deltas)
            except CheckFailure:
                pass
        active_nvim = controller.nvim if controller is not None else nvim
        clean_exit = active_nvim.close()
        report["nvim_exit_clean"] = clean_exit
    finally:
        active_nvim = controller.nvim if controller is not None else nvim
        if active_nvim.process is not None and active_nvim.process.poll() is None:
            if controller is not None:
                try:
                    deltas = controller.raw_delta_events()
                    report["raw_delta_count"] = len(deltas)
                except CheckFailure:
                    pass
            clean_exit = active_nvim.close()
            report["nvim_exit_clean"] = clean_exit
        report["finished_at"] = dt.datetime.now(dt.UTC).isoformat()
        if not clean_exit:
            report["temporary_repo_retained"] = str(repo)
            report["nvim_socket_retained"] = str(socket_path)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)
        result_path, _delta_path = write_report(args.output_dir, report, deltas)
        if clean_exit:
            socket_path.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "result_path": report.get("result_path"),
                "scenario_statuses": [
                    {
                        "name": scenario.get("name"),
                        "status": scenario.get("status"),
                        "code": scenario.get("code"),
                    }
                    for scenario in report.get("scenarios", [])
                ],
                "fatal_code": report.get("fatal_code"),
                "session_count": len(report.get("session_ids", [])),
                "raw_delta_count": report.get("raw_delta_count", 0),
                "nvim_exit_clean": report.get("nvim_exit_clean"),
            },
            sort_keys=True,
        )
    )
    required = {
        "type_pause_one_request_one_display",
        "divergent_typing_classification",
        "typed_matching_classification",
        "model_completion_and_one_at_a_time_switch",
        "stale_response_after_file_switch",
        "off_cancels_unseen_inference",
        "restart_restores_persistent_automatic_mode",
        "global_user_mode_persists_across_lazyvim_restart",
    }
    passed = {
        scenario.get("name")
        for scenario in report.get("scenarios", [])
        if scenario.get("status") == "passed"
    }
    return 0 if required <= passed and not report.get("fatal_code") and clean_exit else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deployment-ready",
        action="store_true",
        help="confirm the parent has completed ThinkPad deployment",
    )
    parser.add_argument("--nvim", default=os.environ.get("TABCOMPLETE_NVIM_BIN", DEFAULT_NVIM))
    parser.add_argument(
        "--timeout",
        type=float,
        default=90,
        help="seconds allowed for a model request/switch (15..300)",
    )
    parser.add_argument(
        "--quiet-window",
        type=float,
        default=2,
        help="seconds to watch for a duplicate automatic request (0..10)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--expected-global-mode-file",
        type=pathlib.Path,
        default=pathlib.Path("/home/shlok/.local/state/lazyvim/tabcomplete-rust-editor-mode.json"),
        help="expected global LazyVim automatic-mode state path",
    )
    parser.add_argument(
        "--expected-plugin-spec",
        type=pathlib.Path,
        default=pathlib.Path(DEFAULT_PLUGIN_SPEC),
        help="installed LazyVim plugin spec that declares the Nix source root",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except CheckFailure as exc:
        print(json.dumps({"status": "failed", "code": str(exc)}, sort_keys=True))
        raise SystemExit(1) from exc
