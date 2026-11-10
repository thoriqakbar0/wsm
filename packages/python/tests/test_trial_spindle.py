"""Spindle GPU reservation and deployment topology contracts."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

from wsm.binary64 import WeightlessError
from wsm.trial_poc import Budget


class SpindleBudgetTests(unittest.TestCase):
    def test_gpu_envelope_includes_prior_spend_and_does_not_token_price_spindle(self):
        prices = {"max_usd": 10, "prior_reserved_usd": 1.38, "modal_hourly_usd": 5,
                  "modal_build_usd": 0.5, "modal_egress_per_gib": 0.04,
                  "spindle_hourly_usd": 16, "spindle_seconds": 1200}
        with tempfile.TemporaryDirectory() as temporary:
            budget = Budget(prices, Path(temporary) / "budget.jsonl")
            with self.assertRaises(WeightlessError):
                budget.reserve("spindle_sample", {"sample_tokens": 1})
            budget.reserve("spindle_create", {})
            prepaid = budget.reserved
            self.assertAlmostEqual(prepaid, 1.38 + 16 / 3)
            budget.reserve("spindle_sample", {"sample_tokens": 1_000_000})
            budget.reserve("spindle_train", {"train_tokens": 2_000_000})
            self.assertEqual(budget.reserved, prepaid)
            with self.assertRaises(ValueError):
                budget.reserve("spindle_create", {})
            with self.assertRaises(ValueError):
                budget.reserve("tinker_create", {})
            with self.assertRaises(WeightlessError):
                budget.reserve("modal_sandbox", {"seconds": 3600})
            self.assertEqual(budget.reserved, prepaid)


@unittest.skipUnless(importlib.util.find_spec("spindle"), "Spindle deployment dependency unavailable")
class TopologyTests(unittest.TestCase):
    def test_recipe_overrides_multigpu_preset_before_modal_provisioning(self):
        from spindle.deployments import DeploymentConfig
        path = Path(__file__).resolve().parents[3] / "examples/spindle/qwen3_4b.py"
        spec = importlib.util.spec_from_file_location("wsm_poc_recipe", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        deployment = DeploymentConfig.create(module.config)
        recipe = deployment.recipe
        self.assertEqual(recipe.model, "Qwen/Qwen3-4B")
        self.assertEqual(recipe.trainer_nodes * recipe.trainer_gpus_per_node, 1)
        self.assertEqual(recipe.inference_gpus_per_node * recipe.inference_max_replicas, 1)
        self.assertEqual(recipe.trainer_max_instances, 1)
        self.assertEqual(recipe.trainer_max_clients_per_instance, 1)
        self.assertLessEqual(recipe.trainer_timeout_s, 900)
        self.assertEqual(recipe.miles_cfg["tensor_model_parallel_size"], 1)
        self.assertEqual(recipe.sglang_cfg["tp_size"], 1)
