"""Errors raised by model editing and program execution."""


class WeightlessError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
