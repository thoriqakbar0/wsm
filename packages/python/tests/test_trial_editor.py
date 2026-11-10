"""Proxy editor contracts through a local HTTP SSE server, not a live provider."""

import http.server
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from wsm.errors import WeightlessError
from wsm.trial_editor import ProxyEditor, validate_config


class ProxyEditorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.requests, self.mode, self.reservations = [], "normal", []
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append({"body": request, "authorization": self.headers.get("Authorization")})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if owner.mode == "disconnect":
                    return
                output = [{"type": "message", "content": [{"type": "output_text", "text": '{"action":"finish"}'}]}]
                if owner.mode == "tool":
                    output = [{"type": "function_call", "name": "shell"}]
                completion = {"id": "resp-fixture", "status": "completed", "model": request["model"],
                              "tools": [], "tool_choice": "none", "output": output,
                              "usage": {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17}}
                if owner.mode == "wrong_model":
                    completion["model"] = "different-route"
                self.wfile.write(b"data: " + json.dumps({"type": "response.completed", "response": completion}).encode() + b"\n\n")

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.config = {"provider": "responses_proxy", "base_url": f"http://127.0.0.1:{self.server.server_port}/v1",
                       "model_env": "EDITOR_MODEL", "api_key_env": "EDITOR_KEY",
                       "max_tokens": 512, "max_turns": 2, "request_seconds": 2}

    def editor(self, name="editor"):
        with patch.dict(os.environ, {"EDITOR_MODEL": "opaque-editor-route", "EDITOR_KEY": "fixture-secret"}, clear=True):
            return ProxyEditor(self.config, directory=self.root / name,
                               reserve=lambda operation, units: self.reservations.append((operation, units)))

    def test_exact_prompt_and_source_are_retained_without_route_or_credentials(self):
        editor = self.editor()
        prompt = "Inspect the public history.\nUse Python if needed."
        result = editor.sample(prompt)
        self.assertEqual(result["prompt"], prompt)
        self.assertEqual(result["text"], '{"action":"finish"}')
        self.assertEqual(result["teacher"], "Codex")
        self.assertEqual(result["provider"], "responses_proxy")
        self.assertEqual(result["response_id"], "resp-fixture")
        self.assertEqual(self.requests[0]["body"]["input"], [{"role": "user", "content": prompt}])
        self.assertEqual(self.requests[0]["authorization"], "Bearer fixture-secret")
        self.assertEqual(self.reservations, [("teacher_request", {"requests": 1, "max_output_tokens": 512})])
        records = "".join(path.read_text() for path in (self.root / "editor").rglob("*.json"))
        self.assertNotIn("opaque-editor-route", records)
        self.assertNotIn("fixture-secret", records)
        self.assertIn("Codex", records)
        self.assertNotIn("opaque-editor-route", json.dumps(result))

    def test_failed_stream_and_tool_attempt_are_not_automatically_retried(self):
        for mode in ("disconnect", "tool", "wrong_model"):
            with self.subTest(mode=mode):
                self.mode = mode
                editor = self.editor(mode)
                before = len(self.requests)
                with self.assertRaises(WeightlessError):
                    editor.sample("One public prompt")
                with self.assertRaisesRegex(ValueError, "uncertain"):
                    editor.sample("Another public prompt")
                self.assertEqual(len(self.requests), before + 1)
                with self.assertRaises(FileExistsError):
                    self.editor(mode)
                failure, = (self.root / mode).rglob("failure.json")
                self.assertEqual(json.loads(failure.read_text())["automatic_retry"], False)
                records = "".join(path.read_text() for path in (self.root / mode).rglob("*.json"))
                self.assertNotIn("opaque-editor-route", records)
                self.assertNotIn("different-route", records)

    def test_turn_and_request_bounds_apply_before_network(self):
        editor = self.editor()
        with self.assertRaises(ValueError):
            editor.sample("x" * 262145)
        self.assertEqual(self.requests, [])
        self.assertEqual(self.reservations, [])
        editor.sample("first")
        editor.sample("second")
        with self.assertRaisesRegex(ValueError, "limit"):
            editor.sample("third")
        self.assertEqual(len(self.requests), 2)

    def test_remote_plain_http_and_missing_credentials_fail_before_network(self):
        for endpoint in ("http://example.invalid/v1", "https://user:key@example.invalid/v1", "https://example.invalid?key=secret"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_config({**self.config, "base_url": endpoint})
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            ProxyEditor(self.config, directory=self.root / "missing", reserve=lambda *_: None)
        self.assertEqual(self.requests, [])
