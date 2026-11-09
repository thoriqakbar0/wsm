"""Public workspace configuration boundaries, independent of sandbox availability."""
import tempfile
import unittest
from pathlib import Path

from wsm import Experiment, WeightlessError
from wsm.client import make_runner, read_json, write_json


class WorkspaceConfigurationTests(unittest.TestCase):
    def test_ambiguous_or_nonfinite_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "config.json"
            for raw in ('{"target":1,"target":2}', '{"x":NaN}', '{"x":1e999}', '{broken'):
                with self.subTest(raw=raw):
                    path.write_text(raw)
                    with self.assertRaises(WeightlessError) as caught:
                        read_json(path)
                    self.assertEqual(caught.exception.code, "invalid_json")

    def test_artifact_write_never_overwrites_user_data(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "proposal.json"
            path.write_text("keep this")
            with self.assertRaises(FileExistsError):
                write_json(path, {"edits": []})
            self.assertEqual(path.read_text(), "keep this")

    def test_open_missing_workspace_does_not_create_database(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(WeightlessError) as caught:
                Experiment.open(root)
            self.assertEqual(caught.exception.code, "workspace_missing")
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_invalid_config_is_rejected_before_creating_workspace(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "public"
            for config in ([], {}, {"format": "unknown"}):
                with self.subTest(config=config), self.assertRaises(WeightlessError):
                    Experiment.create(path, config)
            self.assertFalse(path.exists())

    def test_runner_must_be_selected_explicitly_without_fallback(self):
        for config in ({}, {"backend": "host"}, {"backend": "cloud"}, []):
            with self.subTest(config=config), self.assertRaises(WeightlessError) as caught:
                make_runner(config)
            self.assertEqual(caught.exception.code, "invalid_config")
