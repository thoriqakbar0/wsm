"""Exact-token RL datum contracts and local training-receipt failure behavior.

These tests construct real SDK datums. The training transport checks are local
simulations and do not establish a hosted Tinker training update.
"""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from wsm.trial_training import TinkerPolicy, build_datums, validate_config


HAS_TINKER = importlib.util.find_spec("tinker") is not None


def policy_config(provider="spindle"):
    return {"provider": provider, "base_url": "https://training.example.modal.run",
            "model": "test-model", "rank": 8, "loss_fn": "importance_sampling",
            "learning_rate": 0.0001, "reward_baseline": 0.0, "max_tokens": 16, "max_turns": 2}


def turn(prompt=None, response=None, logprobs=None):
    return {
        "policy": "policy-one", "prompt_tokens": [10, 20, 30] if prompt is None else prompt,
        "tokens": [40, 50] if response is None else response,
        "logprobs": [-0.5, -1.25] if logprobs is None else logprobs,
        "text": '{"action":"finish"}',
    }


class BackendConfigTests(unittest.TestCase):
    def test_provider_and_credential_free_https_endpoint_are_explicit(self):
        config = policy_config()
        self.assertEqual(validate_config(config), config)
        for endpoint in ("", "http://training.example", "https://key@training.example",
                         "https://training.example?api_key=example", "https://training.example/#key",
                         "https://training.example:bad", "https://training.example\n"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_config({**config, "base_url": endpoint})
        for missing in ("provider", "base_url"):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                validate_config({key: value for key, value in config.items() if key != missing})
        for seconds in (0, float("inf"), True):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                validate_config({**config, "request_seconds": seconds})


@unittest.skipUnless(HAS_TINKER, "tinker SDK unavailable")
class BackendTests(unittest.TestCase):
    def test_spindle_uses_its_own_key_and_reserves_gpu_provider_operations(self):
        config = policy_config()
        reserve = Mock()
        with patch.dict("os.environ", {"WSM_SPINDLE_API_KEY": "tml-local-spindle-fixture",
                                       "TINKER_API_KEY": "tml-hosted-fixture"}, clear=True):
            with patch("tinker.ServiceClient") as service:
                policy = TinkerPolicy(config, reserve=reserve)
                self.assertEqual(service.call_args.kwargs,
                                 {"base_url": config["base_url"], "api_key": "tml-local-spindle-fixture"})
                self.assertEqual([call.args[0] for call in reserve.call_args_list],
                                 ["spindle_create", "spindle_snapshot"])
                policy.close()

    def test_missing_provider_key_never_falls_back_or_connects(self):
        for provider, environment, expected in (
                ("spindle", {"TINKER_API_KEY": "tml-hosted-fixture"}, "WSM_SPINDLE_API_KEY"),
                ("tinker", {"WSM_SPINDLE_API_KEY": "tml-local-spindle-fixture"}, "TINKER_API_KEY")):
            with self.subTest(provider=provider), patch.dict("os.environ", environment, clear=True):
                with patch("tinker.ServiceClient") as service, self.assertRaisesRegex(ValueError, expected):
                    TinkerPolicy(policy_config(provider), reserve=Mock())
                service.assert_not_called()


@unittest.skipUnless(HAS_TINKER, "tinker SDK unavailable")
class DatumTests(unittest.TestCase):
    def test_targets_shift_once_and_only_responses_receive_final_reward(self):
        first, second = build_datums([
            turn(), turn(prompt=[1, 2, 3, 4], response=[5], logprobs=[-2.0]),
        ], reward=0.75, baseline=0.25)
        self.assertEqual(first.model_input.to_ints(), [10, 20, 30, 40])
        self.assertEqual(first.loss_fn_inputs["target_tokens"].data, [20, 30, 40, 50])
        self.assertEqual(first.loss_fn_inputs["advantages"].data, [0.0, 0.0, 0.5, 0.5])
        self.assertEqual(first.loss_fn_inputs["logprobs"].data, [0.0, 0.0, -0.5, -1.25])
        self.assertEqual(second.model_input.to_ints(), [1, 2, 3, 4])
        self.assertEqual(second.loss_fn_inputs["target_tokens"].data, [2, 3, 4, 5])
        self.assertEqual(second.loss_fn_inputs["advantages"].data, [0.0, 0.0, 0.0, 0.5])
        self.assertEqual(second.loss_fn_inputs["logprobs"].data, [0.0, 0.0, 0.0, -2.0])

    def test_single_prompt_token_preserves_first_response_loss(self):
        datum, = build_datums([turn(prompt=[7])], reward=-0.5, baseline=0)
        self.assertEqual(datum.model_input.to_ints(), [7, 40])
        self.assertEqual(datum.loss_fn_inputs["target_tokens"].data, [40, 50])
        self.assertEqual(datum.loss_fn_inputs["advantages"].data, [-0.5, -0.5])

    def test_invalid_sampled_trajectories_cannot_become_training_data(self):
        invalid = [[], [turn(prompt=[])], [turn(response=[])], [turn(logprobs=[-0.5])],
                   [turn(logprobs=[float("nan"), -1.0])], [turn(logprobs=[0.1, -1.0])],
                   [turn(response=[True, 2])], [turn(), {**turn(), "policy": "other"}]]
        for trajectory in invalid:
            with self.subTest(trajectory=trajectory), self.assertRaises(ValueError):
                build_datums(trajectory, 1.0, 0.0)
        for reward, baseline in [(float("nan"), 0.0), (1.0, float("inf")), (True, 0.0)]:
            with self.subTest(reward=reward, baseline=baseline), self.assertRaises(ValueError):
                build_datums([turn()], reward, baseline)


@unittest.skipUnless(HAS_TINKER, "tinker SDK unavailable")
class ReceiptTests(unittest.TestCase):
    def test_unknown_optimizer_outcome_retains_intent_and_prevents_resubmission(self):
        policy = object.__new__(TinkerPolicy)
        policy.config = policy_config()
        policy.policy_id = "policy-one"
        policy.training_started = False
        policy.reserve = Mock()
        policy.training = Mock()
        policy.training.forward_backward.return_value.result.return_value.metrics = {"loss:sum": -0.5}
        policy.training.optim_step.return_value.result.side_effect = TimeoutError("simulated lost response")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "training"
            with self.assertRaises(TimeoutError):
                policy.train([turn()], 0.5, output=output)
            receipt = json.loads((output / "receipt.json").read_text())
            self.assertEqual(receipt["state"], "outcome_unknown")
            self.assertEqual(receipt["last_state"], "gradients_computed")
            self.assertEqual(receipt["training_tokens"], 4)
            self.assertEqual(receipt["response_tokens"], 2)
            with self.assertRaisesRegex(ValueError, "already attempted"):
                policy.train([turn()], 0.5, output=Path(temporary) / "retry")
            self.assertFalse((Path(temporary) / "retry").exists())
            policy.training.save_state.assert_not_called()

    def test_foreign_policy_and_existing_receipt_fail_before_training(self):
        policy = object.__new__(TinkerPolicy)
        policy.config = {"reward_baseline": 0.0}
        policy.policy_id = "different-policy"
        policy.training_started = False
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "training"
            with self.assertRaisesRegex(ValueError, "not sampled"):
                policy.train([turn()], 1.0, output=output)
            self.assertFalse(output.exists())
            policy.policy_id = "policy-one"
            output.mkdir()
            (output / "receipt.json").write_text("previous uncertainty")
            with self.assertRaises(FileExistsError):
                policy.train([turn()], 1.0, output=output)
            self.assertEqual((output / "receipt.json").read_text(), "previous uncertainty")


if __name__ == "__main__":
    unittest.main()
