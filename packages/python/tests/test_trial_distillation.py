"""Teacher distillation uses student tokens and real SDK cross-entropy datums."""

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from wsm.trial_distillation import StudentTrainer, build_teacher_datums, validate_student_config


HAS_TINKER = importlib.util.find_spec("tinker") is not None


def config():
    return {"provider": "spindle", "base_url": "https://student.example.modal.run",
            "model": "student-model", "rank": 8, "loss_fn": "cross_entropy", "learning_rate": 1e-5}


def trajectory():
    return [{"teacher": "Codex", "prompt": "public task", "text": '{"action":"finish"}',
             "tokens": [800, 900], "logprobs": [-99.0, -99.0]}]


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking, return_dict):
        return [10, 20]

    def encode(self, text, *, add_special_tokens):
        return [30, 40]


class ConfigTests(unittest.TestCase):
    def test_student_requires_spindle_cross_entropy_and_credential_free_endpoint(self):
        self.assertEqual(validate_student_config(config()), config())
        for changes in ({"provider": "tinker"}, {"loss_fn": "importance_sampling"},
                        {"base_url": "http://example.com"}, {"base_url": "https://key@example.com"},
                        {"base_url": "https://example.com?key=value"}, {"rank": True},
                        {"learning_rate": float("nan")}, {"max_context_tokens": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_student_config({**config(), **changes})


@unittest.skipUnless(HAS_TINKER, "Tinker SDK unavailable")
class DatumTests(unittest.TestCase):
    def test_student_retokenizes_teacher_text_masks_prompt_and_appends_eos(self):
        datum, = build_teacher_datums(trajectory(), Tokenizer(), -2)
        self.assertEqual(datum.model_input.to_ints(), [10, 20, 30, 40])
        self.assertEqual(datum.loss_fn_inputs["target_tokens"].data, [20, 30, 40, 99])
        self.assertEqual(datum.loss_fn_inputs["weights"].data[0], 0.0)
        self.assertTrue(all(math.isclose(value, math.exp(-2), rel_tol=1e-6)
                            for value in datum.loss_fn_inputs["weights"].data[1:]))
        self.assertEqual(set(datum.loss_fn_inputs), {"target_tokens", "weights"})

    def test_existing_eos_is_not_duplicated_and_quality_weight_is_bounded(self):
        tokenizer = Tokenizer()
        tokenizer.encode = lambda text, **kwargs: [30, 99]
        for reward, expected in [(2, 1.0), (-1000, math.exp(-20))]:
            with self.subTest(reward=reward):
                datum, = build_teacher_datums(trajectory(), tokenizer, reward)
                self.assertEqual(datum.loss_fn_inputs["target_tokens"].data, [20, 30, 99])
                self.assertEqual(datum.loss_fn_inputs["weights"].data[0], 0.0)
                self.assertTrue(all(math.isclose(value, expected, rel_tol=1e-6)
                                    for value in datum.loss_fn_inputs["weights"].data[1:]))

    def test_empty_nonliteral_and_oversized_trajectories_are_rejected(self):
        for turns in ([], [{"teacher": "other", "prompt": "a", "text": "b"}],
                      [{"teacher": "Codex", "prompt": "", "text": "b"}],
                      [{"teacher": "Codex", "prompt": "a", "text": []}]):
            with self.subTest(turns=turns), self.assertRaises(ValueError):
                build_teacher_datums(turns, Tokenizer(), 0)
        with self.assertRaisesRegex(ValueError, "no text was truncated"):
            build_teacher_datums(trajectory(), Tokenizer(), 0, max_context_tokens=4)
        for reward in (float("nan"), float("inf"), True):
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                build_teacher_datums(trajectory(), Tokenizer(), reward)


@unittest.skipUnless(HAS_TINKER, "Tinker SDK unavailable")
class TrainerTests(unittest.TestCase):
    def test_constructor_is_lazy_and_hosted_credentials_never_substitute(self):
        with patch("tinker.ServiceClient") as service:
            student = StudentTrainer(config(), reserve=Mock())
            service.assert_not_called()
            with patch.dict("os.environ", {"TINKER_API_KEY": "hosted-fixture"}, clear=True):
                with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "WSM_SPINDLE_API_KEY"):
                    student.train(trajectory(), -2, output=Path(temporary) / "training")
            service.assert_not_called()

    def test_uncertain_optimizer_retains_exact_datums_and_prevents_retraining(self):
        reserve = Mock()
        student = StudentTrainer(config(), reserve=reserve)
        with patch("tinker.ServiceClient") as service, patch.dict(
                "os.environ", {"WSM_SPINDLE_API_KEY": "tml-spindle-fixture"}, clear=True):
            training = service.return_value.create_lora_training_client.return_value.result.return_value
            training.get_info.return_value.model_id = "student-run"
            training.get_tokenizer.return_value = Tokenizer()
            training.forward_backward.return_value.result.return_value.metrics = {"loss:sum": 1.0}
            training.optim_step.return_value.result.side_effect = TimeoutError("simulated uncertain update")
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "training"
                with self.assertRaises(TimeoutError):
                    student.train(trajectory(), -2, output=output)
                receipt = json.loads((output / "receipt.json").read_text())
                datums = json.loads((output / "datums.json").read_text())
                self.assertEqual(receipt["state"], "outcome_unknown")
                self.assertEqual(receipt["last_state"], "update_dispatched")
                self.assertEqual(receipt["training_tokens"], 4)
                self.assertEqual(receipt["response_tokens"], 3)
                self.assertEqual(datums[0]["loss_fn_inputs"]["target_tokens"]["data"], [20, 30, 40, 99])
                with self.assertRaisesRegex(ValueError, "already attempted"):
                    student.train(trajectory(), -2, output=Path(temporary) / "retry")
                training.save_weights_and_get_sampling_client.assert_not_called()
                training.save_state.assert_not_called()
                self.assertEqual([call.args[0] for call in reserve.call_args_list], ["spindle_create", "spindle_train"])

    def test_local_tokenization_failure_is_known_before_training(self):
        reserve = Mock()
        student = StudentTrainer(config(), reserve=reserve)
        with patch("tinker.ServiceClient") as service, patch.dict(
                "os.environ", {"WSM_SPINDLE_API_KEY": "tml-spindle-fixture"}, clear=True):
            training = service.return_value.create_lora_training_client.return_value.result.return_value
            training.get_info.return_value.model_id = "student-run"
            training.get_tokenizer.side_effect = ImportError("missing template renderer")
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "training"
                with self.assertRaises(ImportError):
                    student.train(trajectory(), -2, output=output)
                receipt = json.loads((output / "receipt.json").read_text())
                self.assertEqual(receipt["state"], "failed_before_training")
                self.assertEqual(receipt["last_state"], "client_created")
                self.assertEqual(receipt["training_run"], "student-run")
                training.forward_backward.assert_not_called()
                training.optim_step.assert_not_called()
                self.assertEqual([call.args[0] for call in reserve.call_args_list], ["spindle_create"])

    def test_completed_update_saves_student_state_without_sampling(self):
        student = StudentTrainer(config(), reserve=Mock())
        with patch("tinker.ServiceClient") as service, patch.dict(
                "os.environ", {"WSM_SPINDLE_API_KEY": "tml-spindle-fixture"}, clear=True):
            training = service.return_value.create_lora_training_client.return_value.result.return_value
            training.get_info.return_value.model_id = "student-run"
            training.get_tokenizer.return_value = Tokenizer()
            training.forward_backward.return_value.result.return_value.metrics = {"loss:sum": 1.0}
            training.optim_step.return_value.result.return_value.metrics = {"step": 1}
            training.save_state.return_value.result.return_value.path = "tinker://fixture/weights/step1"
            with tempfile.TemporaryDirectory() as temporary:
                receipt = student.train(trajectory(), -2, output=Path(temporary) / "training")
            self.assertEqual(receipt["state"], "complete")
            self.assertEqual(receipt["checkpoint"], "tinker://fixture/weights/step1")
            service.return_value.create_sampling_client.assert_not_called()
            training.save_weights_and_get_sampling_client.assert_not_called()
            self.assertEqual(training.forward_backward.call_args.args[1], "cross_entropy")
            training.optim_step.assert_called_once()
