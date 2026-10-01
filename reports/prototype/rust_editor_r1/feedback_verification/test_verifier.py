import importlib.util
import json
import pathlib
import unittest
from unittest.mock import patch

SCRIPT = pathlib.Path(__file__).with_name("verify_synthetic_session.py")
SPEC = importlib.util.spec_from_file_location("verify_synthetic_session", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


Q25 = audit.MODEL_REGISTRY["q25"]


def sse_frame(event: dict) -> bytes:
    return b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\n\n"


class VerifierTests(unittest.TestCase):
    @staticmethod
    def partial_match_fixture(post_events):
        path = "/synthetic/main.py"
        prediction_id = "prediction-partial"
        source = b"return \n"
        action_blob = b'{"kind":"replace_line","text":"return value"}'
        content_hash = audit.sha256(source)
        action_hash = audit.sha256(action_blob)

        def event(event_id, sequence, event_type, payload):
            return {
                "event_id": event_id,
                "sequence_number": sequence,
                "event_type": event_type,
                "payload": payload,
            }

        rows = [
            event(
                "anchor",
                1,
                "buffer_open",
                {
                    "path": path,
                    "content_hash": content_hash,
                    "blob_uploaded": True,
                    "fileformat": "unix",
                    "eol": True,
                    "line_count": 1,
                },
            ),
            event(
                "request",
                2,
                "prediction_requested",
                {
                    "prediction_id": prediction_id,
                    "file": path,
                    "editable_range": {"start_row": 0},
                },
            ),
            event(
                "generated",
                3,
                "prediction_generated",
                {
                    "prediction_id": prediction_id,
                    "action_blob_hash": action_hash,
                },
            ),
            event(
                "causal-edit",
                4,
                "edit_delta",
                {
                    "path": path,
                    "start_row": 0,
                    "old_end_row": 1,
                    "new_end_row": 1,
                    "deleted_text": "return ",
                    "inserted_text": "return val",
                },
            ),
            event(
                "dismissal",
                5,
                "prediction_dismissed",
                {
                    "prediction_id": prediction_id,
                    "outcome": "typed_partial_match",
                    "ended_by_event_id": "causal-edit",
                    "ended_by_sequence": 4,
                },
            ),
        ]
        rows.extend(
            event(event_id, sequence, event_type, payload)
            for sequence, (event_id, event_type, payload) in enumerate(post_events, start=6)
        )
        return rows, {content_hash: source, action_hash: action_blob}, prediction_id, path

    def test_q25_accepts_only_declared_policies_and_preserves_exact_value(self):
        policies = (
            "single-line-context-v2",
            "single-line-cursor-last-context-v1",
        )
        for policy in policies:
            with self.subTest(policy=policy):
                self.assertEqual(audit.require_context_policy(Q25, policy), policy)

        with self.assertRaisesRegex(audit.AuditError, "request_context_policy_mismatch"):
            audit.require_context_policy(Q25, "single-line-context-v3")

    def test_sweep_policy_allowlist_stays_distinct(self):
        sweep = audit.MODEL_REGISTRY["sweep"]
        self.assertEqual(
            audit.require_context_policy(sweep, "sweep-window-context-v1"),
            "sweep-window-context-v1",
        )
        with self.assertRaisesRegex(audit.AuditError, "request_context_policy_mismatch"):
            audit.require_context_policy(sweep, "single-line-cursor-last-context-v1")

    def test_backend_timings_tolerate_only_float_json_round_trip_noise(self):
        stored = {
            "cache_n": 339,
            "prompt_n": 76,
            "prompt_ms": 478.055135,
            "predicted_n": 11,
            "predicted_ms": 287.153784,
            "total_ms": 766.30781,
        }
        terminal = dict(stored)
        terminal["predicted_ms"] = 287.15378400000003
        comparison = audit.compare_backend_timings(stored, terminal)
        self.assertTrue(comparison["matches"])
        self.assertFalse(comparison["exact_match"])
        self.assertEqual(
            comparison["float_fields_with_round_trip_difference"][0]["field"],
            "predicted_ms",
        )

        terminal["predicted_n"] = 12
        self.assertFalse(audit.compare_backend_timings(stored, terminal)["matches"])
        terminal = dict(stored)
        terminal["predicted_ms"] += 0.001
        self.assertFalse(audit.compare_backend_timings(stored, terminal)["matches"])

    def test_raw_q25_sse_validates_terminal_and_sampled_token_ids(self):
        raw = sse_frame({"content": "R\tvalue", "stop": False, "tokens": [19]})
        raw += sse_frame(
            {
                "content": "",
                "stop": True,
                "stop_type": "eos",
                "tokens_predicted": 1,
                "canonical_action": {"kind": "replace_line", "text": "value"},
                "model_protocol": Q25["model_protocol"],
                "model_sha256": Q25["sha256"],
                "timings": {
                    "cache_n": 1,
                    "prompt_n": 10,
                    "prompt_ms": 1.0,
                    "predicted_n": 1,
                    "predicted_ms": 2.0,
                    "total_ms": 3.0,
                },
            }
        )
        wire, details = audit.parse_sse(raw, Q25, Q25["sha256"])
        self.assertEqual(audit.decode_single_line(wire), details["canonical_action"])
        self.assertEqual(details["token_id_count"], 1)
        self.assertTrue(details["sampled_token_ids_present"])

    def test_raw_sse_rejects_wrong_model_digest(self):
        raw = sse_frame({"content": "N", "stop": False, "tokens": [1]})
        raw += sse_frame(
            {
                "content": "",
                "stop": True,
                "stop_type": "eos",
                "tokens_predicted": 1,
                "canonical_action": {"kind": "keep", "text": None},
                "model_protocol": Q25["model_protocol"],
                "model_sha256": "0" * 64,
                "timings": {
                    "cache_n": 0,
                    "prompt_n": 1,
                    "prompt_ms": 1,
                    "predicted_n": 1,
                    "predicted_ms": 1,
                    "total_ms": 2,
                },
            }
        )
        with self.assertRaisesRegex(audit.AuditError, "raw_sse_model_digest_mismatch"):
            audit.parse_sse(raw, Q25, Q25["sha256"])

    def test_sweep_wire_maps_only_the_target_line(self):
        source = b"head\r\nold\r\ntail\r\n"
        self.assertEqual(
            audit.map_sweep(b"head\r\nnew\r\ntail\r\n", source, 1),
            {"kind": "replace_line", "text": "new"},
        )
        self.assertIsNone(audit.map_sweep(b"changed\r\nnew\r\ntail\r\n", source, 1))

    def test_canonical_action_blob_rejects_duplicate_keys(self):
        with self.assertRaisesRegex(audit.AuditError, "canonical_action_blob_invalid"):
            audit.decode_action_blob(b'{"kind":"keep","kind":"delete_line","text":null}')

    def test_delta_replay_matches_later_anchor_and_undo(self):
        path = "/synthetic/sample.py"
        before = b"first\nold\n"
        accepted = b"first\nnew\n"
        blobs = {audit.sha256(before): before, audit.sha256(accepted): accepted}
        rows = [
            {
                "event_id": "anchor-1",
                "sequence_number": 1,
                "event_type": "buffer_open",
                "payload": {
                    "path": path,
                    "content_hash": audit.sha256(before),
                    "blob_uploaded": True,
                    "fileformat": "unix",
                    "eol": True,
                    "line_count": 2,
                },
            },
            {
                "event_id": "delta-1",
                "sequence_number": 2,
                "event_type": "edit_delta",
                "payload": {
                    "path": path,
                    "start_row": 1,
                    "old_end_row": 2,
                    "new_end_row": 2,
                    "deleted_text": "old",
                    "inserted_text": "new",
                },
            },
            {
                "event_id": "delta-undo",
                "sequence_number": 3,
                "event_type": "edit_delta",
                "payload": {
                    "path": path,
                    "start_row": 1,
                    "old_end_row": 2,
                    "new_end_row": 2,
                    "deleted_text": "new",
                    "inserted_text": "old",
                },
            },
            {
                "event_id": "anchor-2",
                "sequence_number": 4,
                "event_type": "buffer_write",
                "payload": {
                    "path": path,
                    "content_hash": audit.sha256(before),
                    "blob_uploaded": True,
                    "fileformat": "unix",
                    "eol": True,
                    "line_count": 2,
                },
            },
        ]
        # The first segment is deliberately checked after the inverse edit.
        result = audit.replay_session(rows, blobs)
        self.assertEqual(result["delta_count"], 2)
        self.assertEqual(result["replay_mismatch_count"], 0)
        self.assertEqual(result["unanchored_delta_count"], 0)
        self.assertEqual(result["replay_segments"][0]["byte_equal"], True)

    def test_contiguous_per_key_typing_verifies_eventual_match_after_partial(self):
        path = "/synthetic/main.py"
        post_events = [
            ("key-1", "key", {}),
            (
                "edit-1",
                "edit_delta",
                {
                    "path": path,
                    "start_row": 0,
                    "old_end_row": 1,
                    "new_end_row": 1,
                    "deleted_text": "return val",
                    "inserted_text": "return valu",
                },
            ),
            ("key-2", "key", {}),
            (
                "edit-2",
                "edit_delta",
                {
                    "path": path,
                    "start_row": 0,
                    "old_end_row": 1,
                    "new_end_row": 1,
                    "deleted_text": "return valu",
                    "inserted_text": "return value",
                },
            ),
        ]
        rows, blobs, prediction_id, _path = self.partial_match_fixture(post_events)
        replay = audit.replay_session(rows, blobs)
        trace = replay["post_dismissal_trajectories"][0]
        self.assertEqual(trace["eventual_exact_match"]["event_id"], "edit-2")
        self.assertTrue(trace["eventual_exact_match_continuity_verified"])
        self.assertEqual(len(trace["continuity_key_edit_pairs"]), 2)

        prediction = {
            "prediction_id": prediction_id,
            "outcome": "typed_partial_match",
            "dismissal_comparison": {
                "event_id": "dismissal",
                "causal_edit_event_id": "causal-edit",
                "causal_edit_sequence": 4,
                "outcome": "typed_partial_match",
                "derived_input_relation": "matching_prefix_only",
                "post_dismissal_trajectory": trace,
            },
        }
        coverage = audit.coverage_summary(
            [prediction],
            {
                "verified_file_jump_count": 0,
                "distinct_navigation_identity_count": 0,
            },
            {
                "sequence_contiguous": True,
                "replay_mismatch_count": 0,
                "unanchored_delta_count": 0,
            },
        )
        self.assertEqual(coverage["outcome_counts"], {"typed_partial_match": 1})
        self.assertTrue(coverage["typed_match_causally_verified"])
        self.assertTrue(coverage["eventual_typed_match_after_partial"])
        self.assertFalse(coverage["divergent_typing_causally_compared"])
        evidence = coverage["typed_match_evidence"][0]
        self.assertEqual(evidence["raw_dismissal_outcome"], "typed_partial_match")
        self.assertEqual(evidence["causal_edit_event_id"], "causal-edit")
        self.assertEqual(evidence["exact_match_event_id"], "edit-2")

    def test_navigation_breaks_eventual_typed_match_continuity(self):
        path = "/synthetic/main.py"
        rows, blobs, _prediction_id, _path = self.partial_match_fixture(
            [
                ("navigation", "cursor_move", {}),
                ("key", "key", {}),
                (
                    "edit",
                    "edit_delta",
                    {
                        "path": path,
                        "start_row": 0,
                        "old_end_row": 1,
                        "new_end_row": 1,
                        "deleted_text": "return val",
                        "inserted_text": "return value",
                    },
                ),
            ]
        )
        trace = audit.replay_session(rows, blobs)["post_dismissal_trajectories"][0]
        self.assertIsNone(trace["eventual_exact_match"])
        self.assertFalse(trace["eventual_exact_match_continuity_verified"])
        self.assertEqual(trace["continuity_break"]["event_type"], "cursor_move")

    def test_reload_updates_replay_metadata_and_is_not_typed_matching(self):
        path = "/synthetic/reloaded.py"
        before = b"old\r\n"
        after = b"new"
        before_hash = audit.sha256(before)
        after_hash = audit.sha256(after)
        rows = [
            {
                "event_id": "anchor-before",
                "sequence_number": 1,
                "event_type": "buffer_open",
                "payload": {
                    "path": path,
                    "content_hash": before_hash,
                    "blob_uploaded": True,
                    "fileformat": "dos",
                    "eol": True,
                    "line_count": 1,
                },
            },
            {
                "event_id": "reload-delta",
                "sequence_number": 2,
                "event_type": "edit_delta",
                "payload": {
                    "path": path,
                    "start_row": 0,
                    "old_end_row": 1,
                    "new_end_row": 1,
                    "deleted_text": "old",
                    "inserted_text": "new",
                    "change_origin": "buffer_reload",
                    "fileformat_after": "unix",
                    "eol_after": False,
                },
            },
            {
                "event_id": "anchor-after",
                "sequence_number": 3,
                "event_type": "buffer_open",
                "payload": {
                    "path": path,
                    "content_hash": after_hash,
                    "blob_uploaded": True,
                    "fileformat": "unix",
                    "eol": False,
                    "line_count": 1,
                },
            },
        ]
        replay = audit.replay_session(rows, {before_hash: before, after_hash: after})
        self.assertEqual(replay["replay_mismatch_count"], 0)
        self.assertTrue(replay["replay_segments"][0]["byte_equal"])

        trace_rows, trace_blobs, _prediction_id, _path = self.partial_match_fixture(
            [
                (
                    "reload",
                    "edit_delta",
                    {
                        "path": "/synthetic/main.py",
                        "start_row": 0,
                        "old_end_row": 1,
                        "new_end_row": 1,
                        "deleted_text": "return val",
                        "inserted_text": "return value",
                        "change_origin": "buffer_reload",
                        "fileformat_after": "unix",
                        "eol_after": True,
                    },
                )
            ]
        )
        trace = audit.replay_session(trace_rows, trace_blobs)["post_dismissal_trajectories"][0]
        self.assertEqual(trace["continuity_break"]["reason"], "editor_buffer_reload")
        self.assertIsNone(trace["eventual_exact_match"])

    def test_classify_group_uses_full_session_rows_to_link_undo(self):
        prediction_id = "prediction-1"
        request_id = "request-1"
        path = "/synthetic/main.py"
        source = b"old\n"
        accepted_bytes = b"new\n"
        context = b"synthetic prompt"
        action_blob = b'{"kind":"replace_line","text":"new"}'
        raw_response = b"synthetic SSE"
        action = {"kind": "replace_line", "text": "new"}
        context_hash = audit.sha256(context)
        action_hash = audit.sha256(action_blob)
        raw_hash = audit.sha256(raw_response)
        q25 = audit.MODEL_REGISTRY["q25"]
        runtime_hash = "a" * 64
        context_policy = "single-line-context-v2"
        timings = {
            "cache_n": 0,
            "prompt_n": 12,
            "prompt_ms": 1.0,
            "predicted_n": 1,
            "predicted_ms": 1.0,
            "total_ms": 2.0,
        }
        request_range = {
            "start_row": 0,
            "start_col": 0,
            "end_row": 0,
            "end_col": 3,
        }
        accepted_range = audit.action_range(source, 0, action)

        def event(event_id, sequence, event_type, payload):
            return {
                "event_id": event_id,
                "sequence_number": sequence,
                "event_type": event_type,
                "payload": payload,
            }

        def prediction_payload(**fields):
            return {
                "prediction_id": prediction_id,
                "request_id": request_id,
                "synthetic": True,
                **fields,
            }

        request = event(
            "request-event",
            1,
            "prediction_requested",
            prediction_payload(
                model_alias="q25",
                model_protocol=q25["model_protocol"],
                model_gguf_sha256=q25["sha256"],
                model_revision=q25["sha256"],
                runtime_config_hash=runtime_hash,
                context_policy_version=context_policy,
                context_hash=context_hash,
                context_blob_hash=context_hash,
                wire_version=q25["wire_version"],
                precision=q25["precision"],
                max_output_tokens=q25["output_tokens"],
                pre_state_hash=audit.sha256(source),
                file_identity=path,
                editable_range=request_range,
            ),
        )
        generated = event(
            "generated-event",
            2,
            "prediction_generated",
            prediction_payload(
                model_alias="q25",
                model_protocol=q25["model_protocol"],
                model_sha256=q25["sha256"],
                runtime_config_hash=runtime_hash,
                context_hash=context_hash,
                context_policy_version=context_policy,
                wire_version=q25["wire_version"],
                max_output_tokens=q25["output_tokens"],
                canonical_action=action["kind"],
                stop_type="eos",
                action_blob_hash=action_hash,
                raw_response_hash=raw_hash,
                first_token_observation="sampled_token_ids",
                first_token_at_ms=1.0,
                first_text_at_ms=1.0,
                prompt_tokens=12,
                backend_timings=timings,
            ),
        )
        shown = event(
            "shown-event",
            3,
            "prediction_shown",
            prediction_payload(
                action="replace_line",
                action_blob_hash=action_hash,
                proposed_text="new",
                context_hash=context_hash,
                wire_version=q25["wire_version"],
                active_buffer=True,
                focused=True,
                proposed_start={"row": 0, "col": 0},
                proposed_end={"row": 0, "col": 3},
                proposed_start_byte=0,
                proposed_end_byte=3,
                proposed_range_end_exclusive=True,
                proposed_range_includes_terminator=False,
            ),
        )
        accepted = event(
            "accepted-event",
            4,
            "prediction_accepted",
            prediction_payload(
                shown_event_id=shown["event_id"],
                applied_through_sequence=5,
                editable_range=accepted_range,
                proposed_start_byte=0,
                proposed_end_byte=3,
            ),
        )
        applied_delta = {
            "start_row": 0,
            "old_end_row": 1,
            "new_end_row": 1,
            "deleted_text": "old",
            "inserted_text": "new",
        }
        accepted_delta = event(
            "accepted-delta",
            5,
            "edit_delta",
            {"path": path, **applied_delta},
        )
        undo_delta = event(
            "undo-delta",
            6,
            "edit_delta",
            {
                "path": path,
                "start_row": 0,
                "old_end_row": 1,
                "new_end_row": 1,
                "deleted_text": "new",
                "inserted_text": "old",
            },
        )
        event_rows = [request, generated, shown, accepted, accepted_delta, undo_delta]
        group = [request, generated, shown, accepted]
        blobs = {
            context_hash: context,
            action_hash: action_blob,
            raw_hash: raw_response,
        }
        replay = {
            "request_sources": {
                prediction_id: {"source": source, "path": path},
            },
            "seq_to_id": {5: accepted_delta["event_id"]},
            "event_states": {
                accepted_delta["event_id"]: {
                    "sequence": 5,
                    "path": path,
                    "after": accepted_bytes,
                    "payload": applied_delta,
                },
                undo_delta["event_id"]: {
                    "sequence": 6,
                    "path": path,
                    "after": source,
                    "payload": undo_delta["payload"],
                },
            },
            "post_dismissal_trajectories": [],
        }
        projection = {
            "prediction_id": prediction_id,
            "request_event_id": request["event_id"],
            "shown_event_id": shown["event_id"],
            "last_outcome_event_id": accepted["event_id"],
            "outcome": "accepted",
            "model_gguf_sha256": q25["sha256"],
            "model_revision": q25["sha256"],
            "runtime_config_hash": runtime_hash,
            "context_policy_version": context_policy,
            "context_blob_hash": context_hash,
            "action_blob_hash": action_hash,
            "review_status": "unreviewed",
            "review_event_id": None,
            "file_identity": path,
            "resolved_file_id": None,
            "updated_through_sequence": 4,
        }
        raw_details = {
            "canonical_action": action,
            "terminal": {
                "timings": timings,
                "model_sha256": q25["sha256"],
                "model_protocol": q25["model_protocol"],
            },
            "nonempty_text_observed": True,
        }
        with (
            patch.object(audit, "parse_sse", return_value=(b"R\tnew", raw_details)),
            patch.object(audit, "_CURRENT_PROJECTIONS", [projection]),
            patch.object(audit, "_file_ids_by_repo_path", {}),
        ):
            result = audit.classify_group(
                prediction_id,
                group,
                event_rows,
                blobs,
                replay,
                None,
                None,
            )

        self.assertEqual(result["accepted_edit_delta_linked"], True)
        self.assertEqual(result["undo_event_id"], undo_delta["event_id"])
        self.assertEqual(result["undo_restores_pre_state"], True)

    def test_action_range_keeps_byte_coordinates_and_newline(self):
        source = "α = 1\nnext\n".encode()
        result = audit.action_range(source, 0, {"kind": "replace_line", "text": "α = 2"})
        self.assertEqual(result["start_byte"], 0)
        self.assertEqual(result["end_byte"], len("α = 1".encode()))
        self.assertEqual(
            audit.apply_action_to_source(source, 0, {"kind": "replace_line", "text": "α = 2"}),
            "α = 2\nnext\n".encode(),
        )


if __name__ == "__main__":
    unittest.main()
