"""Python tools for editing, evaluating, and saving model weights."""

__version__ = "0.1.0"
__all__ = ["WeightModel", "execute_program"]


def __getattr__(name):
    if name == "WeightModel":
        from .editing import WeightModel
        return WeightModel
    if name == "execute_program":
        from .program import execute_program
        return execute_program
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
