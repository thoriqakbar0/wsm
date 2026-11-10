from .binary64 import NumericalEvaluation, WeightlessError, __version__
from .core import MetricPolicy, SearchRun, evaluation_result, evaluator_config, replay
from .session import Session, replay_session, verify_session
from .verifier import Verifier
from .sandbox import DockerRunner, stage_model, stage_smollm
from .local_sandbox import BubblewrapRunner
from .teacher import ResponsesTeacher
from .client import Experiment, addition_config, create_example_verifier

__all__ = ["MetricPolicy", "NumericalEvaluation", "SearchRun", "WeightlessError",
           "evaluation_result", "evaluator_config", "replay", "Session", "Verifier",
           "DockerRunner", "BubblewrapRunner", "stage_model", "stage_smollm", "ResponsesTeacher",
           "Experiment", "addition_config", "create_example_verifier", "replay_session", "verify_session",
           "export_checkpoint", "WeightModel", "execute_program"]


def __getattr__(name):
    if name == "export_checkpoint":
        from .checkpoint_export import export_checkpoint
        return export_checkpoint
    if name == "WeightModel":
        from .editing import WeightModel
        return WeightModel
    if name == "execute_program":
        from .program import execute_program
        return execute_program
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
